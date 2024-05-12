# Karpathy-makemore
This repository is my implementation for `Andrej Karpathy's` [makemore](https://github.com/karpathy/makemore/blob/master/makemore.py)

[makemore_origin.py](https://github.com/Zhaohr1990/Karpathy-makemore/blob/main/makemore_origin.py) is my implementation following the original code, where Bigram, Ngram, MLP models are incorporated.
[makemore_hz.py](https://github.com/Zhaohr1990/Karpathy-makemore/blob/main/makemore_hz.py) is an updated implementation, where the dataset and dataloader are revised. Instead of organizing the tensor at the word level, i.e., batch_size, max_word_length, feature_dimension, I construct the dataset based on markov_order/block_size, i.e., batch_size, markov_order, feature_dimension. The model objects are updated accordingly. 

[To-do]: Add Transformers
