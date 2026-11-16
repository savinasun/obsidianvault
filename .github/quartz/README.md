# Published notes site

The knowledge notes are published as a [Quartz](https://quartz.jzhao.xyz) v4.5.2 site at
**https://savinasun.github.io/obsidianvault/**.

## How it works

- `.github/workflows/publish.yml` runs on every push to `main` that touches the MOC,
  `_01_notes/`, `_04_images/` or this folder (i.e. obsidian-git's `vault backup` pushes),
  and can be run by hand from the repo's Actions tab ("Publish notes site" → Run workflow).
- `prepare_content.py` copies an **allowlist** into Quartz's `content/`: the MOC (as the home
  page), every note in `_01_notes/`, and only the images those pages embed. Root-level notes,
  templates, `.claude/`, `.obsidian/` and unreferenced images are not part of the site.
  It rewrites the published copies only (never the vault): title heading → page title,
  `**Tags**::` → page tags, dataview blocks → tag-page links, Obsidian-only math layouts
  and pasted non-breaking spaces → forms the site's KaTeX renderer accepts.
- `quartz.config.ts`, `quartz.layout.ts`, `custom.scss` replace Quartz's defaults
  (site title, no analytics, readable key-term color in light mode).
- The built site is deployed with GitHub's official Pages actions; nothing is committed back.

## One-time setup

Repo → Settings → Pages → Build and deployment → Source: **GitHub Actions**.
Then push (or wait for the next vault backup), or run the workflow from the Actions tab.

## Preview locally

```sh
git clone --depth 1 --branch v4.5.2 https://github.com/jackyzha0/quartz.git && cd quartz && npm ci
cp ../obsidianvault/.github/quartz/quartz.{config,layout}.ts . && cp ../obsidianvault/.github/quartz/custom.scss quartz/styles/
python3 ../obsidianvault/.github/quartz/prepare_content.py ../obsidianvault content
npx quartz build --serve        # http://localhost:8080
```
