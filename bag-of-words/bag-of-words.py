import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    vector = np.zeros(len(vocab), dtype=int)
    
    for token in tokens:
        if token in vocab:
            index = vocab.index(token)
            vector[index] += 1
    
    return vector