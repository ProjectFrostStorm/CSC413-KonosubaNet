# CSC413-KonosubaNet

# Task

The goal of the project is to make a generative text model. The model is fed excerpts from the web novel series, KonoSuba, and tries to generate text mimicking the writing style. 

# Model
We decided to use a transformer model. The model takes in a sequence of characters as inputs, and outputs a sequence of probabilities for the next character. The implementation of our model is based off of the implementation from http://peterbloem.nl/blog/transformers. However, instead of using the self attention module written in the blog, we use pytorch's MultiheadAttention, and we use positional encoding (implementation from https://pytorch.org/tutorials/beginner/transformer_tutorial.html) instead of positional embedding. 

This is a overview of the model

! [Model](Transformer Architecture.jpeg)

# Data

# Results

# Ethical concerns

