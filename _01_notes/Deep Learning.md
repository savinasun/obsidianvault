### [[dl_moc]]  
## **Basics of Neural Network**
**Tags**:: #architecture 
**Links**:: 


SIMD: single instruction multiple data → GPU are much better at SIMD computations than CPU
### Forward Propagation

- Derivations of dL/dz

    ![[Pasted image 20250513105438.png|500]]
    
    [https://community.deeplearning.ai/t/derivation-of-dl-dz/165](https://community.deeplearning.ai/t/derivation-of-dl-dz/165)
    

### Backward Propagation

Joint State Method: Optimal Control 1960s - Backpropgation

### Activation Functions

- Sigmoid
    
- Softmax: a function converts a set of numbers to a set of probabilities sum up to 1
    
    Softmax tends to amplify the variances of high variance vectors
    
- Tanh
    - Tanh is superior than Sigmoid in near all aspects, so don’t use sigmoid unless it is the output layer
    - $tanh’(x) = 1 - tanh^2(x)$
- ReLu
    - The most commonly used activation function, if not sure what to use, just use it; it is fast for learning since there is no gradient vanish as the output closes to 1
    - LeNet uses sigmoid, AlexNet uses ReLu

## [[Neural Network Initialization]]

## Improve Deep Neural Network

### [[Double Descent]]


### **Regularization Techniques**

[Regularization techniques for training deep neural networks | AI Summer](https://theaisummer.com/regularization/)

- **Loss Function Modifications**
    - L2 regularization:
        - Weights end up smaller ("weight decay"):
            - Weights are pushed to smaller values.
    - L1 regularization
    - Elastic Net
- **Entropy Regularization**
    - bayesian models: forces the output probability distribution towards uniform distribution to reduce bias
    - applies in reinforcement learning
- **Label Smoothing**
    - inject randomness/softness to the output labels, especially in classification tasks
- **Dropout (Invert Dropout)**
    + []
    + [[Hinton et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html), **Dropout: A Simple Way to Prevent Neural Networks from Overfitting]**
    - A **common mistake** when using dropout is to use it both in training and testing. You should **use dropout** (randomly eliminate nodes) **only in training**.
    - Apply dropout both during forward and backward propagation.
    - During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations.
    - By using a larger mini batch size, the regularization effect is reduced.
    - Inverted Dropout, Gaussian dropout, Dropout Connect, Variational Dropout, etc.
    
- **Stochastic Depth**
    
- **Data Augmentation**
    
    - by applying transformations like rotation, flipping, cropping, or scaling to images (or similar manipulations in other data types), data augmentation creates a more diverse training set, forcing the model to learn robust features that are not specific to the original data orientation or appearance.
- **Early stopping / Orthogonalization**
    
    ![[Pasted image 20250513110300.png]]
    
- **Batch Normalization**
    
    ![[Pasted image 20250513110318.png]]
    

### Batch Gradient Descent V.S. Mini-Batch Gradient Descent

![[Pasted image 20250513110349.png]]

- If mini batch size is 1: it is called **Stochastic Gradient Descent** and it is very floppy
    
    - loss all speed up from vectorization
- If mini batch size is equal to the full training set size: it is called Batch Gradient Descent
    
    - → small data set: less than 2000 training samples
- Typical mini batch size: 64, 128, 256, 512
    
    ![[Pasted image 20250513110403.png]]
    

### **Optimization Algorithms**

- **Momentum**
	- Mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.
    - exponential moving average
    - Bias correction
- RMSprop: Jeffrey Hinton
- Adam: Adaptive moment estimation
- Adam-W
- Learning rate scheduler/decay
    - During the first part of training, your model can get away with taking large steps, but over time, using a fixed value for the learning rate alpha can cause your model to get stuck in a wide oscillation that never quite converges. But if you were to slowly reduce your learning rate alpha over time, you could then take smaller, slower steps that bring you closer to the minimum. This is the idea behind learning rate decay.
    - Can be achieved by using either adaptive methods or pre-defined learning rate schedules, such as:
        - linear decay
        - inverse square-root decay
        - forced decay at particular iterations

Local Optima & Saddle Point

## Convolutional Neural Network

### Convolutional Networks

- **LeNet-5 (1989)**:
    
    + [LeCun et al., 1998. Gradient-based learning applied to document recognition]
    
    - ~60k parameters
        
    - Architecture of CNN
        
        ![[Pasted image 20250513110442.png|500]]
        
        - Convolutional Layer
        - Pooling Layer
            - f: filter size
            - s: stride size
            - input size: $n_H \times n_W \times n_C$ 
            - output size: $[\frac{n_H-f}{s}+1] \times [\frac{n_W-f}{s}+1] \times n_C$
            - **Max pooling or average pooling**
            - No parameters to learn
            
        - Fully-Connected Layer (Dense Layer)
            - **$1 \times 1$ convolutions:** [https://www.facebook.com/yann.lecun/posts/10152820758292143](https://www.facebook.com/yann.lecun/posts/10152820758292143)
                
                Suppose we have a conv layer which outputs an $(𝑁,𝐹,𝐻,𝑊)$ shaped tensor,
                
                and is fed into a conv layer with $𝐹_1$ $1\times 1$ filters, zero padding and stride 1
                
                then the output of this $1 \times 1$ conv layer will have shape $(𝑁,𝐹_1,𝐻,𝑊)$
                
                $1 \times 1$ conv is used to change the dimensionality in the filter space.
                
                If $𝐹_1>𝐹$, then we are increasing dimensionality;
                
                if $𝐹_1<𝐹$, we are decreasing dimensionality, in the filter dimension.
                
    - **Inductive Bias in CNN**
        
        [https://iclr-blogposts.github.io/2023/blog/2023/how-does-the-inductive-bias-influence-the-generalization-capability-of-neural-networks/](https://iclr-blogposts.github.io/2023/blog/2023/how-does-the-inductive-bias-influence-the-generalization-capability-of-neural-networks/)
        
        - **Locality**
            - **Parameter Sharing:** A feature detector (such as a vertical edge detector) that’s useful in one part of the image is probably useful in another part of the image
            - **Sparsity of Connections:** In each layer, each output layer depends only on a small region (patch) of the inputs
        - **Translation Invariance**
            - **"Translation Invariance"** in a Convolutional Neural Network (CNN) means that **the network can identify an object in an image regardless of its position within the image**, essentially meaning the output of the network remains the same even if the object is shifted or translated to a different location; this property is primarily achieved through the use of convolution operations and pooling layers, especially max pooling, which focus on the most salient features irrespective of their exact position within the image.
    
- **AlexNet (2012)**:
    
    [[Krizhevesky et al., 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), ImageNet classification with deep convolutional neural networks]
    
    - similar to LeNet but much bigger: ~60m parameters
    - ReLu activation function
    - training on multiple GPUs
    - Local Response Normalization (LRN)
    
- **Visualizing ConvNet (2013)**:
    
    [[Zeiler & Fergus., 2014](https://arxiv.org/abs/1311.2901), Visualizing and Understanding Convolutional Networks]
    
    - set a one-hot feature map
        
    - back-prop to pixels
        
    - receptive field
        
        - larger RF size correlated to improved recognition performance
            
            ![[Pasted image 20250513110512.png|500]]
            
        - [_Araujo et al., 2019,_ [Computing receptive fields of convolutional neural networks](https://distill.pub/2019/computing-receptive-fields)]
            
            ![[Pasted image 20250513110529.png]]
            
        
	        ![[Pasted image 20250513110544.png|300]]
        
        
- **VGG-16 (2014)**:
    
    [[Simonyan & Zisserman 2015](https://arxiv.org/abs/1409.1556), Very deep convolutional networks for large-scale image recognition]
    
    - large model: ~138 million parameters
    - 16 hidden layers with weights
    - stage-wise training: not end-to-end training
- **GoogLeNet / Inception (2014)**
    
- **ResNet**
    
    [[He et al., 2015](https://arxiv.org/abs/1512.03385), Deep Residual Learning for Image Recognition**]**
    
    - [https://www.doc.ic.ac.uk/~bkainz/teaching/DL/notes/ResNet.pdf](https://www.doc.ic.ac.uk/~bkainz/teaching/DL/notes/ResNet.pdf)
    - 18, 38, 50, 101, 152 layers
    - intuition: skip connection reduced the complexity of the model
    
    ![[Pasted image 20250513110600.png]]
    
- **MobileNet**
    
- **EfficientNet**
    
    - Compound Scaling: r - resolution, d - depth, w - width
- **ImageNet performances**
    
    ![[Pasted image 20250513110615.png]]
    

### Neural Style Transfer

[[Gatys et al., 2015](https://arxiv.org/abs/1508.06576), A neural algorithm of artistic style]

- Define style as **correlation** between activations across channels
    
    style: learned features across channels tend to appear simultaneously
    
    **style matrix: feature correlations →** **Gram matrix** (the determinant is a metric to compute linear independence of the column vectors)
    
    ![[Pasted image 20250513110707.png]]
    
- Cost function:
    
    ![[Pasted image 20250513110720.png]]
    
    - use hidden layer l to compute the content cost
    - use pre-trained ConvNet (E.g., VGG network)
    - let a_[l](C) and a_[l](G) be the activation of layer l on the content image and generated image
    - if a_[l](C) and a_[l](G) are similar, both images have similar content
    - use **Frobenius norm** to define the cost function

### Object Detection

- Object Localization
    
    - Sliding Windows Detection Algorithm: implement convolutionally to save computation cost
        
        [[Sermanet et al., 2014](https://arxiv.org/abs/1312.6229), OverFeat: Integrated recognition, localization and detection using convolutional networks]
        
    - Evaluation: Intersection over Union (IoU) correct if ≥ 0.5
        
- Landmark Detection
    
- YOLO Algorithm: object detection algorithm → predict multiple bounding boxes
    
    [[Redmon et al., 2015](https://arxiv.org/abs/1506.02640), You Only Look Once: Unified real-time object detection]
    
    - Non-Max Suppression Algorithm:
        - Each output prediction is a vector like (p_c, b_x, b_y, b_h, b_w), same setup as YOLO algo
        - Discard all boxes with p_c ≤ 0.6
        - While there are any remaining boxes:
            - pick the box with the largest p_c, output that as a prediction
            - discard any remaining box with IoU≥0.5 with the output box in the previous step
    - Anchor Box Algorithm: used when there are overlapping objects
- Region Proposal: R-CNN
    
    - R-CNN: [Girshik et al. 2013, Rich feature hierarchies for accurate object detection and semantic segmentation]
        
        Propose regions by segmentation algorithms. Classify proposed regions one at a time. Output label + bounding box.
        
    - Faster algorithms:
        
        - Fast R-CNN: [Girshik. 2015, Fast R-CNN]
            
            Propose regions. Use convolution implementation of sliding windows to classify all the proposed regions.
            
        - Faster R-CNN: [Ren et. al, 2016. Faster R-CNN: Towards real-time object detection with region proposal networks]
            
            Use convolutional network to propose regions.
            
- Semantic Segmentation
    
    - U-Net Architecture
        
        [Ronneberger et al., 2015, U-Net: Convolutional Networks for Biomedical Image Segmentation]
        
        ![[Pasted image 20250513110816.png]]
        
    - Transpose convolution
        

### Face Recognition

- Learn a “Similarity” Function
    
    d(img1, img2) = degree of difference between images
    
    for face verification task: if d≤threshold, “same”; else “different”
    
- Siamese Network
    
    [Taigman et, al., 2014. DeepFace closing the gap to human level performance]
    
    parameters of NN define an encoding of image f(x(i))
    
    learn parameters such that if x(i) and x(j) represent the same person, then ||f(x(i)), f(x(j))||^2 is small;
    
    ```
                                            if x(i) and x(j) represent different persons, then ||f(x(i)), f(x(j))||^2 is large.
    ```
    
- TripLet Loss Function
    
    [Schroff et, al., 2015, FaceNet: A unified embedding for face recognition and clustering]
    
    Given a triplet image sample (A,P,N)
    
    loss(A,P,N) := max(d(A,P) - d(A,N) + a, 0)
    
    objective function J = sum_{i=1}^n loss(A_i, P_i, N_i)
    
- training dataset: 10k pictures of 1k persons
    
    we need to create triplet samples drawn from the dataset satisfies patterns like (**A**nchor, **P**ositive, **N**egative)
    
    during training, if (A,P,N) are chosen randomly, d(A,P) + a ≤ d(A,N) is easily satisfied;
    
    so we need to choose triplets that are hard to train on, i.e. d(A,P) \approx d(A,N)
    

**Chi-Square Similarity**

[https://www.geeksforgeeks.org/chi-square-distance-in-python/](https://www.geeksforgeeks.org/chi-square-distance-in-python/)

![[Pasted image 20250513110833.png]]

### Scale Invariant Feature Detection (SIFT)

- A method to find features in images

### **Data Augmentation Techniques**

- Mirroring
- Random Cropping
    - Multi-Crop at test time: run trained model on multiple versions of test images and average results
- Rotation, Shearing, Local warping
- Color Shifting
    - PCA color augmentation: AlexNet paper

Implementation of Data Distortions during Training

- multiple CPU threads doing distortions for data
- training on CPU/GPU
- (parellel)

Two sources of Knowledge

- Labeled data
- Hand engineered features/network architecture/other components

## Sequence Models

### Sequence Models

- **RNN (Recurrent Neural Network)**
    
    - Sequence to Sequence Architecture:
        
        [Cho, et al., 2014. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.]
        
        [Sutskever, et al. 2014. Sequence to sequence learning with neural networks.]
        
        - emporally local connections/loops
        - temporally weight-sharing
    - Recurrent Unit/Cell:
        
        ![image.png](attachment:5c0bad9c-f133-4833-b78c-5c9869fe99e8:image.png)
        
        ![image.png](attachment:78d9c6ce-3690-443c-baf1-0efb65b99183:image.png)
        
        - RNN Cell Input:
            - The hidden state from the previous cell 𝑎⟨𝑡-1⟩
            - The current time step's input data 𝑥⟨𝑡⟩
        - RNN Cell Output:
            - A hidden state 𝑎⟨𝑡⟩
            - A prediction 𝑦⟨𝑡⟩
        - the weights and biases parameters (𝑊𝑎𝑎,𝑊𝑎𝑥,𝑏𝑎,𝑊𝑎𝑦,𝑏𝑦) are re-used each time step
    - Cons of RNN:
        
        - **suffer from gradient vanishing problems**
        - not good at catching long-term dependencies and sequencies
    - RNN works best when each output 𝑦̂ ⟨𝑡⟩ can be estimated using "local" context.
        
        "Local" context refers to information that is close to the prediction's time step .
        
- **LSTM (Long Short-Term Memory Network)**
    
    - [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
        
        use gates to control the flow of information
        
        ![image.png](attachment:b4485728-9530-4fd4-9494-194ec584916b:image.png)
        
    - **Forget gate 𝚪𝑓:** is a tensor containing values between 0 and 1
        
        ![image.png](attachment:9f37f72d-4e81-45bf-bab0-aeedaa7f5b9e:image.png)
        
    - **Candidate value 𝐜̃⟨𝑡⟩:** The candidate value is a tensor containing information from the current time step that **may** be stored in the current cell state 𝐜⟨𝑡⟩
        
        values range from -1 to 1
        
        ![image.png](attachment:73e09d38-48c0-4b63-97bb-0265d75ef508:image.png)
        
    - **Update gate 𝚪𝑖:** decide what aspects of the candidate 𝐜̃ ⟨𝑡⟩ to add to the cell state 𝑐⟨𝑡⟩
        
        ![image.png](attachment:980d2a3a-b764-4ce7-a764-817e440e9066:image.png)
        
    - **Cell state 𝐜⟨𝑡⟩:** the cell state is the "memory" that gets passed onto future time steps
        
        ![image.png](attachment:04af7e60-646a-4db5-8b3c-93f91480b5b8:image.png)
        
    - **Output gate 𝚪𝑜:** decides what gets sent as the prediction (output) of the time step
        
        ![image.png](attachment:e7ae48c2-8860-4dd5-9cc5-7e7fd2db3e7c:image.png)
        
    - **Hidden state 𝐚⟨𝑡⟩:** gets passed to the LSTM cell's next time step and determines the prediction 𝐲⟨𝑡⟩_𝑝𝑟𝑒𝑑
        
        ![image.png](attachment:8a37563d-5d27-4032-8c6a-7c3abb53a21d:image.png)
        
        ![image.png](attachment:86fe0685-6850-4c8c-9d01-06dfd333f79b:image.png)
        
    - Examples:
        
        - (2016) Google’s Neural Machine Translation System
            
            [Wu, et al., 2016. Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation]
            
            - It uses residual connections to going deep without degradation
        - (2016) DeepMind’s WaveNet for Audio Generation
            
- GRU (Gated Recurrent Unit)
    
- Bidirectional RNN
    

### Word Embedding Models

- Word Representation
    
    - One-Hot Representation: does not do a good job of capturing the level of similarities between words
    - **Embedding Representation**
    - Visualization: [van der Maatan and Hinton., 2008. Visualizing data using t-SNE]
        - t-SNE: [nonlinear dimensionality reduction](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction) technique
- (context, target)
    
- one-hot vectors → embedding matrix → embedding vectors e_j = E \cdot o_j
    
- Word2Vec
    
    - Skip-Gram Model
        - Hierarchical Softmax Classifier: tree structure, not necessary a balanced tree or binary tree, the most common words are at the top, the less common words are buried deep
    - CBOW (Continuous Backward Model)
- GloVe (Global Vectors for Word Representation)
    
- Debias Word Embeddings
    
    [[Boliukbasi et al., 2016](https://arxiv.org/abs/1607.06520). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings]
    
    - Identify Bias Direction
        
        the bias direction could be defined by computing
        
        𝑔1=𝑒_𝑤𝑜𝑚𝑎𝑛−𝑒_𝑚𝑎𝑛
        
        𝑔2=𝑒_𝑚𝑜𝑡ℎ𝑒𝑟−𝑒_𝑓𝑎𝑡ℎ𝑒𝑟
        
        𝑔3=𝑒_𝑔𝑖𝑟𝑙−𝑒_𝑏𝑜𝑦
        
        and so on and averaging over them, which obtains an estimate of the "gender" dimension in the 50 dimensional word embedding space
        
    - Neutralize: for every word that is not definitional, project to get rid of bias
        
        ![image.png](attachment:b70494cb-e729-445c-bbcd-834bf48d73cc:image.png)
        
        ![image.png](attachment:91700198-c428-4ef0-bf70-688386c9e69d:image.png)
        
    - Equalize pairs for gender-specific words
        
        ![image.png](attachment:2d5baa79-718c-4e16-9fa6-159cad09e077:image.png)
        
        ![image.png](attachment:b7f9702e-b69b-4647-b59d-a80a950dfd7b:image.png)
        

**Mahalanobis distance**

### Beam Search Algorithm

- Beam Width: number of nodes considered at each level
    
    - k=1: beam search is equal to greedy search
        
        ![image.png](attachment:76f7c285-7f98-4cb2-b792-3c7ecbe47e4f:image.png)
        
- Length Penalty (length normalization):
    
    - beam search will tend to promote shorter hypotheses
    - length penalty parameter _**α**_: normally set to between 0.5 and 1.0
    - [Wu et al. (2016)](https://arxiv.org/pdf/1609.08144.pdf) **`lp(Y) = (5 + |Y|)α / (5 + 1)α`**
    - **_k_ and _α_ should be fine-tuned independently on your target task**

### Machine Translation Evaluation Metrics

- BLEU Score
    
    - Bilingual Evaluation Understudy Score: ranges from 0 to 1
        
        - a weighted geometric mean of all the **modified n-gram precisions**,
        - multiplied by the **brevity penalty**
        
        ![image.png](attachment:c1f503ed-8441-4f01-a96e-529b33b943c9:image.png)
        
    - Brevity Penalty
        
        - BLEU is penalized for producing shorter answers
            
            ![image.png](attachment:47c8dd9c-92e3-4676-b60b-890a52bbc99d:image.png)
            
    - the most typical choice, the one recommended in the original paper, is w1=⋯=w4=1/4
        
- **PPL Score (Perplexity)**
    
    exponentiated average log-likelihood of the predicted words in a sequence;
    
    or cross-entropy with the exponential function applied
    
    - **Entropy**
        
        ![Screenshot 2025-05-06 at 16.10.17.png](attachment:d25e7d83-5173-4f96-bdbf-0811cbf0c493:Screenshot_2025-05-06_at_16.10.17.png)
        
    - **Cross-Entropy**
        
        ![Screenshot 2025-05-06 at 16.11.13.png](attachment:7afb5064-a52d-4ea0-bc29-62805d18d5c9:Screenshot_2025-05-06_at_16.11.13.png)
        
    - **KL-Divergence**
        
        ![Screenshot 2025-05-06 at 16.12.15.png](attachment:05dfaf82-5fac-4a9c-9960-4012dc2efce5:Screenshot_2025-05-06_at_16.12.15.png)
        
        **D(P||Q) = H(P,Q) - H(P)**
        
    - Perplexity
        
        ![Screenshot 2025-05-06 at 21.49.36.png](attachment:462a66a1-6393-4d7f-b299-a4e24e6dc241:Screenshot_2025-05-06_at_21.49.36.png)
        


Manifold Hypothesis
Local distance is meaningful, global distance is meaningless

ventral pathway v.s. dorsal pathway

Orientation selectivity is **expressed by cells within the visual cortex, when such cells increase impulse or signal activity for specific oriented degree of shape presented within the visual field**.

## References

- Dive Into Deep Learning
    
    [https://d2l.ai/chapter_preface/index.html](https://d2l.ai/chapter_preface/index.html)
    
- NYU DL
    
    [https://atcold.github.io/NYU-DLSP20/en/week01/01-1/](https://atcold.github.io/NYU-DLSP20/en/week01/01-1/)
    
- Deep Learning Fundamentals
    
    [Fundamentals | AI Summer](https://theaisummer.com/learn-ai/fundamentals/)
    
- PyTorch resources
    
    [PyTorch documentation — PyTorch 2.6 documentation](https://pytorch.org/docs/stable/index.html)
    
    [PyTorch Cheat Sheet — PyTorch Tutorials 2.6.0+cu124 documentation](https://pytorch.org/tutorials/beginner/ptcheat.html)
    
- Papers With Code
    
    [Papers with Code - Browse the State-of-the-Art in Machine Learning](https://paperswithcode.com/sota)
    
