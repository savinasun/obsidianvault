### [[dl_moc]]  
## **Mixture of Experts**
**Tags**::  #llm
**Links**::

+ It's a technique where a large model is broken down into smaller, specialized sub-models (experts) that handle different aspects of the input. 
+ A [gating network](https://www.google.com/search?sca_esv=9edba2248acd0033&sxsrf=AE3TifMf94P51qT8K2-oo7XlS_dEqNXl1g%3A1748390221235&q=gating+network&sa=X&sqi=2&ved=2ahUKEwix5MKQ7cSNAxUHKVkFHWv9DPYQxccNegQIHBAB&mstk=AUtExfAWvz1G98LG1YwrsKB4B2aJLY910pDJoaCEpmz99068B3Wnc0C-hMcNZeuJAsK-urTQykH3lz0yiutKAeQsGTddbLdz4Cgj5r6F4Ucs2wNNYO9CJpq_Y9W93OgtuZlM4ZknA7AvDWP62XHSPwq9OIK4yJk9--t0Xo7MRqdutoRxTTWvaoAE4qH0tzsbmnsaxb9relSwWSLfurQ7j-OXNzb4X-iw7-Jp8ZzSAm8uAPV-LPzUHuZfVPlpG37YeqN6bFkRmjCXe1iK9EarWkYhDpIH&csui=3) then selects which experts to activate for a given input, allowing for efficient and specialized processing.