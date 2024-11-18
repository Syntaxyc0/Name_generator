# Name_generator
This Python project implements a Recurrent Neural Network (RNN) from scratch only using the numpy library to generate names. It trains on a dataset of existing names and learns to predict the next character in a sequence, resulting in the creation of new, plausible names.


## How it works:
The code implements a specific type of RNN called a Long Short-Term Memory (LSTM) network.LSTMs are powerful for sequence prediction tasks like name generation.

Here's a simplified breakdown of the process:

### Data Preprocessing:

Loads the name data from a text file.
Converts all names to lowercase characters.
Creates a dictionary to map each unique character to a unique index.
Represents each name as a sequence of these indices.

### Network Architecture:

The network consists of multiple layers that process the sequence of character indices one by one.
Each layer learns the relationships between characters and predicts the probability of the next character in the sequence.
Training:

The network is trained by iterating through the name data multiple times.
During each iteration, it compares its predictions for the next character in a name with the actual character, and adjusts its internal weights to minimize the error.

### Generating Names:

Once trained, the network can be used to generate new names by starting with a random character and predicting the next one based on its internal knowledge.
It continues this process, predicting the next character based on the previous ones, until it generates a complete name that ends with a newline character ("\n").
