### Basics
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "optimisation")
SORT file.name ASC
```


### Transformers
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "transformer")
SORT file.name ASC
```



### VAE
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "vae")
SORT file.name ASC
```


