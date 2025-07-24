# Annotations  
(7/5/2025, 5:10:19 PM)

![](llms_01.png)  
[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=2&annotation=MY6QXD7H)  
([Xiong et al., 2020, p. 2](zotero://select/library/items/PGMN3GAU)) pre-LN v.s. post-LN architecture

![](llms_02.png)  
[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=4&annotation=PF6U74UT)  
([Xiong et al., 2020, p. 4](zotero://select/library/items/PGMN3GAU))

[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=5&annotation=DERMPN7P) “that puts the layer normalization between the residual blocks, the Pre-LN Transformer puts the layer normalization inside the residual connection and places it before all other non-linear transformations.” ([Xiong et al., 2020, p. 5](zotero://select/library/items/PGMN3GAU))

[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=5&annotation=VQ4CVNHF) “Additionally, the Pre-LN Transformer uses a final layer normalization right before the prediction.” ([Xiong et al., 2020, p. 5](zotero://select/library/items/PGMN3GAU))

[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=6&annotation=5CTZGREL) “For the Pre-LN Transformer, the scale of the gradients is much smaller.” ([Xiong et al., 2020, p. 6](zotero://select/library/items/PGMN3GAU))

[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=6&annotation=EQMKXY48) “in the Post-LN Transformer, the norm of xpost l,i is √d and thus we study the norm of xpost,5 l,i instead.” ([Xiong et al., 2020, p. 6](zotero://select/library/items/PGMN3GAU))

[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=6&annotation=5GVNCHM9) “The main idea is that the layer normalization will normalize the gradients. In the Post-LN Transformer, the scale of the inputs to the layer normalization is independent of L, and thus the gradients of parameters in the last layer are independent of L. While in the Pre-LN Transformer, the scale of the input to the final layer normalization is linear in L, and thus the gradients of all parameters will be normalized by √L.” ([Xiong et al., 2020, p. 6](zotero://select/library/items/PGMN3GAU))

[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=6&annotation=T5QJ7ECU) “the gradient norm in the Post-LN Transformer is large for the parameters near the output and will be likely to decay as the layer index l decreases. On the contrary, the gradient norm in the PreTransformer will be likely to stay the same for any layer l.” ([Xiong et al., 2020, p. 6](zotero://select/library/items/PGMN3GAU))

![](llms_03.png)  
[Go to annotation](zotero://open-pdf/library/items/LSKHVHCG?page=7&annotation=BI25J9W3)  
([Xiong et al., 2020, p. 7](zotero://select/library/items/PGMN3GAU)) experiments on gradient norms across layers for pre-LN and post-LN architectures