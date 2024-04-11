import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

num_layers = 6
layer_size = 256

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.hidden_size = h
        self.numOfLayer = num_layers
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.fc = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        h0 = torch.zeros(self.numOfLayer, 1, self.hidden_size).to(device)
        
        out, _ = self.rnn(inputs,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        
        return out

def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        testing = json.load(test_f)

    tra = []
    val = []
    tes = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in testing:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val, tes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) 

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))
    last_train_accuracy = 0
    last_validation_accuracy = 0
    stopping_condition = False
    epoch = 0
    training_results = []
    validation_results = []

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16#I HAVE CHANGED THIS LINE I HAVE CHANGED THIS LINE I HAVE CHANGED THIS LINE I HAVE CHANGED THIS LINE I HAVE CHANGED THIS LINE 
        N = len(train_data)


        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                vectors = np.array(vectors)
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)

                output = model(vectors)

                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]).to(device))

                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]
            vectors = np.array(vectors)
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total
        training_results.append(trainning_accuracy)
        validation_results.append(validation_accuracy)

        if (validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy) or epoch > 5:
                    stopping_condition=True
                    print("Training done to avoid overfitting!")
                    print("Best validation accuracy is:", last_validation_accuracy)
                    
                    model.eval()
                    correct = 0
                    total = 0
                    random.shuffle(test_data)
                    print("Validation started for epoch {}".format(epoch + 1))
                    valid_data = valid_data

                    for input_words, gold_label in tqdm(test_data):
                        input_words = " ".join(input_words)
                        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]
                        vectors = np.array(vectors)
                        vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
                        output = model(vectors)
                        predicted_label = torch.argmax(output)
                        correct += int(predicted_label == gold_label)
                        total += 1
                    print("Validation completed for test")
                    print("Validation accuracy for test: {}".format(correct / total))
                    print(training_results)
                    print(validation_results)
                    
                    import matplotlib.pyplot as plt

                    # Sample data (replace with your actual data)
                    epochs = list(range(0, len(training_results)))  # Example epochs
                    test_result = correct / total  # Example test result

                    # Plotting
                    plt.plot(epochs, training_results, label='Training Results', marker='o')
                    plt.plot(epochs, validation_results, label='Validation Results', marker='o')
                    plt.scatter(epochs[-1], test_result, color='red', label='Test Result')  # Plotting test result at the last epoch
                    plt.xlabel('Epochs')
                    plt.ylabel('Results')
                    plt.title("Training and Validation Results Over Epochs\nLayer Size: {} Number of Layers: {}".format(layer_size,num_layers))
                    plt.legend()
                    plt.grid(True)
                    plt.show()
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy
            
        epoch += 1
