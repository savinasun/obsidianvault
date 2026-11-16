#!/usr/bin/env python3
r"""Assemble the public Quartz `content/` folder from the vault.

Allowlist only: the MOC (as the home page), every note in _01_notes/, and the
images those pages actually embed. Root-level notes, templates, .claude/,
.obsidian/ and unreferenced images are never copied.

Only the published copies are rewritten (the vault files are untouched):
  - the `### [[dl_moc]]` back-link line is dropped (the site header links home),
  - the `# **Title**` heading becomes frontmatter `title` (Quartz renders it),
  - the `**Tags**::` line becomes frontmatter `tags` (so stray "#word" text is not a tag),
  - ```dataview blocks become links to the matching tag pages,
  - `$$…$$` display math glued to surrounding text (fine in Obsidian, broken in
    remark-math) is moved onto its own lines, and `x_\min`-style subscripts get the
    braces KaTeX requires,
  - non-breaking / zero-width characters from pasted text become plain spaces.
File mtimes are set from git history so "last modified" dates are real.

usage: prepare_content.py <vault_dir> <quartz_content_dir>
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

vault, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
NOTES = vault / "_01_notes"
MOC = vault / "_00_maps_of_contents" / "dl_moc.md"
IMAGES = vault / "_04_images"

MOC_LINE = re.compile(r"^`?###\s*\[\[dl_moc\]\]\s*$")
TITLE_LINE = re.compile(r"^#{1,2}\s+(.+?)\s*$")
TAGS_LINE = re.compile(r"^\*\*Tags\*\*::(.*)$")
DATAVIEW = re.compile(r"```dataview\n(.*?)```[ \t]*\n?", re.S)
FENCED_CODE = re.compile(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$")
INLINE_CODE = re.compile(r"(`+)(?!`).*?(?<!`)\1(?!`)")
KATEX_OPERATOR_SCRIPT = re.compile(r"([_^])\s*\\(min|max|sup|inf|lim|log|exp|det|arg)(?![A-Za-z])")
LIST_ITEM = re.compile(r"^([ \t]*)([-+*]|\d+[.)])([ \t]+)")
# Pasted text carries non-breaking / zero-width characters that markdown does not treat as
# whitespace (e.g. " $$" is not a math fence); render them as plain spaces / nothing.
INVISIBLES = {0x00A0: " ", 0x202F: " ", 0x2060: None, 0x200B: None, 0xFEFF: None}


def enclosing_item_indent(before):
    """Content indent of the list item that the next line lazily continues, if any
    (Obsidian keeps a `$$` line under a list item inside that item; CommonMark would not)."""
    for line in reversed(before.split("\n")[:-1]):
        if not line.strip() or line.lstrip().startswith("#"):
            return None
        item = LIST_ITEM.match(line)
        if item:
            return item.group(1) + " " * len(item.group(2) + item.group(3))
    return None


def normalize_math(text):
    """Re-flow display math the way Obsidian reads it: `$` and `$$` are scanned left to
    right, inline `$…$` closes at the next `$`, and each `$$…$$` block is rewritten onto
    its own lines, indented to stay inside an enclosing list item."""
    saved = []

    def mask(m):
        saved.append(m.group(0))
        return f"\x00{len(saved) - 1}\x00"

    text = INLINE_CODE.sub(mask, FENCED_CODE.sub(mask, text))
    fix = lambda s: KATEX_OPERATOR_SCRIPT.sub(r"\1{\\\2}", s)
    out, i, n = "", 0, len(text)
    while i < n:
        j = text.find("$", i)
        if j < 0:
            out += text[i:]
            break
        out += text[i:j]
        if j > 0 and text[j - 1] == "\\":                       # escaped \$
            out, i = out + "$", j + 1
            continue
        if text.startswith("$$", j):                            # display math
            k = text.find("$$", j + 2)
            if k < 0:
                out += text[j:]
                break
            prefix = out[out.rfind("\n") + 1:]
            item = LIST_ITEM.match(prefix)
            indent = (item.group(1) + " " * len(item.group(2) + item.group(3))) if item \
                else re.match(r"[ \t]*", prefix).group(0)
            if not prefix.strip():
                ctx = enclosing_item_indent(out[:len(out) - len(prefix)])
                if ctx is not None and len(ctx.expandtabs(4)) > len(indent.expandtabs(4)):
                    out, indent = out[:len(out) - len(prefix)] + ctx, ctx
            if prefix.strip():
                out = out.rstrip(" \t") + "\n" + indent
            lines = [ln.strip() for ln in fix(text[j + 2:k]).strip().split("\n")]
            out += "$$\n" + "".join(indent + ln + "\n" for ln in lines) + indent + "$$"
            i = k + 2
            eol = text.find("\n", i)
            rest = text[i:eol if eol >= 0 else n]
            if rest.strip():
                out += "\n" + indent
                i += len(rest) - len(rest.lstrip(" \t"))
            continue
        k = j + 1                                               # inline math
        while True:
            k = text.find("$", k)
            if k < 0 or "\n\n" in text[j:k]:
                k = -1
                break
            if text[k - 1] == "\\":
                k += 1
                continue
            break
        if k < 0:
            out, i = out + "$", j + 1
            continue
        out, i = out + "$" + fix(text[j + 1:k]) + "$", k + 1
    return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], out)


def git_mtime(path):
    try:
        ts = subprocess.run(
            ["git", "-C", str(vault), "log", "-1", "--format=%ct", "--", str(path.relative_to(vault))],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(ts) if ts else None
    except (OSError, subprocess.CalledProcessError):
        return None


def dataview_to_tag_links(match):
    tags = re.findall(r'contains\(\s*tags\s*,\s*"([^"]+)"\s*\)', match.group(1))
    if not tags:
        return ""
    return "*Browse all notes tagged* " + " · ".join(f"[#{t}](tags/{t})" for t in tags) + "\n"


def to_page(text, extra_frontmatter=None):
    lines = text.translate(INVISIBLES).split("\n")
    if lines and MOC_LINE.match(lines[0]):
        lines = lines[1:]
    title, tags = None, None
    for i, line in enumerate(lines[:6]):          # the title heading sits right under the MOC line
        m = TITLE_LINE.match(line)
        if m:
            title = m.group(1).replace("**", "").strip()
            del lines[i]
            break
    for i, line in enumerate(lines[:8]):
        m = TAGS_LINE.match(line)
        if m:
            tags = re.findall(r"#([\w/-]+)", m.group(1))
            del lines[i]
            break
    body = DATAVIEW.sub(dataview_to_tag_links, "\n".join(lines)).lstrip("\n")
    body = normalize_math(body)
    fm = {}
    if title:
        fm["title"] = title
    if tags:
        fm["tags"] = tags
    fm.update(extra_frontmatter or {})
    if not fm:
        return body
    head = "".join(f"{k}: {json.dumps(v, ensure_ascii=False)}\n" for k, v in fm.items())
    return f"---\n{head}---\n{body}"


def write(src, dst, text):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    ts = git_mtime(src)
    if ts:
        os.utime(dst, (ts, ts))


if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

pages = []
index = to_page(MOC.read_text(encoding="utf-8"), {"aliases": ["dl_moc"]})
write(MOC, out / "index.md", index)
pages.append(index)
for note in sorted(NOTES.glob("*.md")):
    page = to_page(note.read_text(encoding="utf-8"))
    write(note, out / note.name, page)
    pages.append(page)

refs = set()
for page in pages:
    refs |= {m.split("|")[0].strip() for m in re.findall(r"!\[\[([^\]]+)\]\]", page)}
    refs |= {unquote(os.path.basename(m)) for m in re.findall(r"!\[[^\]]*\]\(<?([^)>\s]+)", page) if "://" not in m}
    refs |= {unquote(os.path.basename(m)) for m in re.findall(r'<img[^>]+src="([^"]+)"', page) if "://" not in m}
missing = []
(out / "images").mkdir()
for name in sorted(refs):
    src = IMAGES / name
    if src.is_file():
        shutil.copy2(src, out / "images" / name)
    else:
        missing.append(name)
print(f"published {len(pages)} pages and {len(refs) - len(missing)} images")
if missing:
    print("referenced images not found in _04_images:", ", ".join(missing))
