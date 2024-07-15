# Overview
This project showcases the complete architecture of a Transformer-based Large Language Model, developed using PyTorch. The model was evaluated using a Mental Health Dataset, which includes two thousand client-psychologist interactions. It is important to note that this testing was not aimed at mimicking the pretraining of a base Transformer model, as my available hardware would not support such an extensive task. Additionally, the dataset used is not suitable for pretraining, which typically involves unsupervised learning on much larger datasets. This project was undertaken solely for self-learning purposes and was not intended to create a full-scale model. Below, I will outline the key insights I gained from each component of the Transformer during the implementation of the model.


# Encoder
### Tokenizing and Embedding
To begin with, words cannot be directly represented in their readable form for machine processing. Instead, we need to convert them into a machine-interpretable format. This process starts with tokenization, which involves assigning unique integers to each word in the input space. These tokens serve as indices for an Artificial Neural Network (ANN).
The ANN uses these indices to look up corresponding weights, which act as the dense vector representations (embeddings) of the words. Importantly, these weights are trainable parameters, meaning they are adjusted during the training process to better capture the linguistic characteristics of the words.

### Positional Encoding
The sequence of words in a sentence is a crucial component of language, meaning Natural Language Processing (NLP) models must account for word order. Traditionally, Recurrent Neural Networks (RNNs) were used to handle sequencing, but they posed significant challenges in terms of training time. The sequential nature of RNNs hinders parallel computation, making the training process slower.
To address this issue, Positional Encoding was introduced. Positional Encoding assigns each word embedding a positional embedding using a combination of sine and cosine functions, along with dimension and word indices. This technique enables models to consider word order without relying on the sequential processing inherent to RNNs, thus allowing for more efficient parallel computation during training.

### Attention
In the Transformer model, the Attention mechanism enhances word embeddings within the context of the sentence by adding additional embeddings. This is crucial because words can carry different meanings depending on their context. To achieve this, the input sentence is segmented into three components: the query, key, and value. The query matrix consists of the original input embeddings, while the key matrix is the transpose of these embeddings. Cosine similarity is then computed between the query and key matrices, followed by applying a softmax function. The resulting matrix indicates how each word in the sentence relates to every other word, capturing their contextual similarities. 

Furthermore, the value matrix retains the original input embeddings. Multiplying the value matrix by the resulting matrix yields a new matrix with the original dimensions of the input embeddings. Each word's embedding now incorporates context embeddings from other words based on their similarity, determined by cosine similarity. Thus, the degree of contextual influence each word receives from others is governed by their similarity.

### Multi-Headed Attention
A single Attention head is insufficient for capturing all contextual embeddings of a sentence. To address this limitation, we partition the input embeddings into N matrices. Each matrix undergoes the Attention mechanism independently, enabling each to capture distinct linguistic contexts within the sentence.

### Concatenate
After all partitioned matrices undergo the Attention mechanism independently, the matrices are concatenated to its original embedding dimensions. 

### Linear Layer, Add and Norm, Feedforward
After concatenation, the matrix is processed through a Linear Layer followed by an Add and Norm component. Adding the original embeddings preserves their linguistic characteristics, while normalizing each feature dimension accelerates convergence. Post Add and Norm, the matrix enters a feed-forward network. This step allows the model to grasp nonlinear transformations necessary for predicting future tokens based on the embeddings' evolving representation. 
The matrix is then passed into another Add and Norm component. This new matrix is now regarded as the output of the Encoder Block.

### Encoder Blocks
In the "Attention is All You Need" paper, each encoder block constitutes one iteration in a series. The output from each encoder block serves as the input for the next block, a process repeated typically eight times. This stacking is crucial because each encoder block can extract varying levels of abstraction from the input data. By sequentially layering multiple blocks, the model effectively captures progressively intricate patterns and relationships within the data.



