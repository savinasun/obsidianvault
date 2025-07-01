### [[dl_moc]]  
## **Byte Pair Encoding**

**Tags**::  
**Links**::

---
The BPE algorithm was first introduced by Philip Gage in 1994 for data compression. It was adapted to NLP for neural machine translation. BPE was then used by GPT-2 [@radford].

Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.