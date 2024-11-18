# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2024-11-18 16:28:54

import numpy as np
import os

data_file = "./data/pokedex_fr.txt"

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]   
    print ('%s' % (txt, ), end='')

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length


def initialize_parameters(n_a, n_x, n_y):

    Wax = np.random.randn(n_a, n_x)*0.01 
    Waa = np.random.randn(n_a, n_a)*0.01 
    Wya = np.random.randn(n_y, n_a)*0.01 
    b = np.zeros((n_a, 1)) 
    by = np.zeros((n_y, 1)) 
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) 
    p_t = softmax(np.dot(Wya, a_next) + by)  
    
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] 
    daraw = (1 - a * a) * da 
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def rnn_forward(X, Y, a0, parameters, vocab_size = 27):

    x, a, y_hat = {}, {}, {}
    
    a[-1] = np.copy(a0)
    
    loss = 0
    
    for t in range(len(X)):
        
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1
        
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
        loss -= np.log(y_hat[t][Y[t],0])
        
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_backward(X, Y, parameters, cache):

    gradients = {}
    
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    

    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    


    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])

    
    return gradients, a


def clip(gradients, maxValue):
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   


    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient)

    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients

def sample(parameters, char_to_ix):
 
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]    
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))   
    indices = []
    idx = -1 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):       
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        idx = np.random.choice(vocab_size, p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        a_prev = a
        
    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, maxValue = 5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X)-1]

def model(data, ix_to_char, char_to_ix, num_iterations = 50001, n_a = 50, names_number = 10, vocab_size = 27, parameters = None, load_saved_weights = False, save_new_weights=False):

    n_x, n_y = vocab_size, vocab_size
    

    if (parameters == None and load_saved_weights == False):
        parameters = initialize_parameters(n_a, n_x, n_y)
    elif (parameters == None and load_saved_weights == True):
        parameters = np.load('./saved_weights/weights_rnn.npy', allow_pickle='TRUE')
        parameters = parameters.tolist()

    loss = get_initial_loss(vocab_size, names_number)
    

    with open(data_file) as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    np.random.shuffle(examples)
    

    a_prev = np.zeros((n_a, 1))
    

    for j in range(num_iterations):

        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 5e-4)

        loss = smooth(loss, curr_loss)

        if j % 50000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            

            for name in range(names_number):
                

                sampled_indices = sample(parameters, char_to_ix)
                print_sample(sampled_indices, ix_to_char)

            print('\n')
    if (save_new_weights == True):
        np.save('./saved_weights/weights_rnn.npy', dict(parameters))
    return parameters

def main():
    data = open(data_file, 'r').read()
    data= data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
    parameters = model(data, ix_to_char, char_to_ix,num_iterations=100001, load_saved_weights=True)

if __name__ == "__main__":
    main()