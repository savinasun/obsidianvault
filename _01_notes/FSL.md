### [[dl_moc]]  
## **Few-Shot Learning**
**Tags**::  #llm
**Links**:: 

| ****Aspect****                      | ****Zero-Shot Learning (ZSL)****                                              | ****One-Shot Learning (OSL)****                           | ****Few-Shot Learning (FSL)****                                         |
| ----------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| ****Number of Training Examples**** | No training examples for new classes                                          | Exactly one example per new class                         | Few examples (typically a few to several dozen) per new class           |
| ****Approach to Learning****        | Uses semantic descriptions or attributes                                      | Uses similarity metrics or metric learning                | Uses meta-learning techniques                                           |
| ****Training Data Requirements****  | Relies on indirect information for prediction                                 | Requires one example for each new class during evaluation | Requires a small number of examples for each new class                  |
| ****Applications****                | Useful when examples for new classes are impractical                          | Ideal for scenarios with only one example available       | Effective when a few examples can be collected                          |
| ****Challenges****                  | Depends on quality of semantic information; struggles with dissimilar classes | Performance relies on the quality of the single example   | Risk of overfitting; sensitive to quality and diversity of few examples |
| ****Methodologies****               | Semantic embedding models, attribute-based methods                            | Similarity-based networks, metric learning                | Meta-learning approaches like MAML                                      |
