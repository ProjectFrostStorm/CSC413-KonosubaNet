# CSC413-KonosubaNet

# Task

The goal of the project is to make a generative text model. The model is fed excerpts from the web novel series, KonoSuba, and tries to generate text mimicking the writing style. 

# Model
We decided to use a transformer model. The model takes in a sequence of characters as inputs, and outputs a sequence of probabilities for the next character. The implementation of our model is based off of the implementation from http://peterbloem.nl/blog/transformers. However, instead of using the self attention module written in the blog, we use pytorch's MultiheadAttention, and we use positional encoding (implementation from https://pytorch.org/tutorials/beginner/transformer_tutorial.html) instead of positional embedding. 

This is a overview of the model
![Model](Transformer_Architecture.jpeg)

The characters first get passed through an embedding layer, and then the positional encoding is added to the embeddings. The inputs are then passed through a series of transformer blocks before the final layer converts the inputs to logits, which is then soft maxed to get the probabilities of the next character.

This is the architecture of each transformer block.
![Transformer Block](Transformer_Block.jpeg)

There is a skipped connection between the multiheaded attention layer, and between the MLP layer. The MLP is two layers of size **4k**, where **k** is the size of the multiheaded attention's output. 

# Data

# Training
For model5, we tuned the following hyperparameters: batch size, learning rate, depth of model and heads and k.

For learning rate, we initially use 0.001, default from adam optimizer. Then we want to try a larger learning rate to speed up the trainin process. With 0.01, the model did not learn well and output garbage words such as (hell# w!fld). We also tried smaller learning rate 0.0008, which perform equally well as 0.001, slightly better then 0.001. It seems like increasing the learning rate from 0.0008 to 0.001 did not affect much on the model training speed and performance.

For depth, larger depth will result into the model requiring more time to train for each epoch, and more computing memory.  Our model is using depth 1, but the largest we could try is 2, otherwise we run into memory issue.

For head and k, heads needs to be divisible by k. The largest heads we could try is 256. Larger heads means more complex model, which should yield better output.

For batch size, we tried 1,2,4 and 8. With heads 256, we run out of memory when batch size equals to 8. So the largest batch size we could have is 4. More heads means more trainable units, with concumes more memory. When we reduced heads to 128, we ere able to use larger batch size. The larger the batch size, the faster will be the training speed (parallel training).

# Results

# Ethical concerns

