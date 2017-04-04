import model
import data

import argparse
import torch
import sys
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--vocab_size', type=int,  default=10000,
                    help='threshold for vocabulary')
parser.add_argument('--data', type=str, default='./data/penn/',
                    help='location of the data corpus')
args = parser.parse_args()

def visualize_embeddings():
    #load model checkpoint
    the_model = torch.load('model_checkpoints/2_layers.pt')

    #convert embedding weights to numpy matrix
    embedding_matrix = the_model.encoder.weight.data.numpy()
    #print embedding_matrix
    
    corpus = data.Corpus(args.data)
    word_to_id = corpus.dictionary.word2idx
    
    #visualize embeddings of a given word list
    word_list = ['a','an','document','in','of','picture','nation','country','end','books','novel','almost', 'work','job']
    """'institutions','organizations','big','assets','portfolio','down',"'",'quite','finance','acquisition','seems','good','great','minutes']"""

    index_list = []
    for word in word_list:
        index_list.append(word_to_id[word])
    
    #reduce dimensionality
    model = TSNE(n_components=2, random_state=0)
    lower_dim_embeddings = model.fit_transform(embedding_matrix)
    
    lower_dim_word_list = lower_dim_embeddings[index_list]
    
    x = lower_dim_word_list[:,0]
    y = lower_dim_word_list[:,1]

    #create scatter plot with labels
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x,y)

    for i, txt in enumerate(word_list):
        ax.annotate(txt, (x[i],y[i]))

    plt.title("2-Dimensional Representation of Word Embeddings")

    plt.savefig("word_list_embeddings.png")

    #visualize embeddings of random words
    n = 25

    random_word_list = random.sample(word_to_id.keys(), n)
    
    random_index_list = []
    for word in random_word_list:
        random_index_list.append(word_to_id[word])

    lower_dim_random_list = lower_dim_embeddings[random_index_list]

    x = lower_dim_random_list[:,0]
    y = lower_dim_random_list[:,1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x,y)

    for i, txt in enumerate(random_word_list):
        ax.annotate(txt, (x[i],y[i]))

    plt.title("2-Dimensional Representation of Random Word Embeddings")

    plt.savefig("random_word_embeddings.png")

visualize_embeddings()
