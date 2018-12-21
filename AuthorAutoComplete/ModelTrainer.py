# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:56:08 2017

@author: nhj
"""

import nltk
from __future__ import absolute_import, division, print_function

# %matplotlib nbagg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
tf.set_random_seed(42)

import os
import sys
sys.path.append(os.path.join('.', '..')) 

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import pickle
import sys
import heapq




DATA_DIR = "C:/Users/erona/Desktop/nlpplayground/AuthorAutocomplete/austen/"
DATA_DIR = "C:/Users/NHJ/Desktop/playground/AuthorAutoComplete/austen/"

import glob
filenames = glob.glob(DATA_DIR + "*.txt")
text = ""

for filename in filenames:
    with open(filename, 'r', encoding="utf-8") as file:
        text = text + file.read()
    print('corpus length:', len(text))


chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')


SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print(f'num training examples: {len(sentences)}')




X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1



model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))


optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history




model.save(DATA_DIR + 'AustenAutoComplete_model.h5')
pickle.dump(history, open(DATA_DIR + "AustenAutoComplete_history.p", "wb"))


model = load_model(DATA_DIR + 'AustenAutoComplete_model.h5')




plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');
plt.show()




def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
        
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]



quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]


quotes = [
    "Bake at 350 degrees for 30 to 32 minutes. Test corners to see if done, as center will ",
    "Whip 1 pint of heavy cream. Add 4 Tbsp. brandy or rum to",
    "Cook over a hot grill, or over",
    "With blender on high speed, add ice cubes, one at a time, making certain each cube",
    "Dice the pulp of the eggplant and put it in a bowl with the",
    "As this is a tart rather than a cheesecake, you should",
    "This may be one of the most exceptional souffles you’ll ever serve. The beet color spreads",
    "Coat apple slices with "
        ]


for q in quotes:
    seq = q[:40].lower()
    print(seq)
    pred = [predict_completions(seq, 1)][0][0]
    predictions = [pred]
    newseq = seq[len(pred):] + pred
    for i in range(1,10):
        pred = predict_completions(newseq, 1)[0]
        newseq = newseq[len(pred):] + pred
        predictions.append(pred)
    print(seq + " ".join(predictions))
    print()











#tokenize and clean data
tokenized_text = nltk.word_tokenize(data.lower())
data = tokenized_text

# import word features from FastText
# and map words to word features

N_WORD_FEATURES = 200
SEQ_LENGTH = 4


X = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, N_WORD_FEATURES))
y = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, N_WORD_FEATURES))
for i in range(0, len(data)/SEQ_LENGTH):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, N_WORD_FEATURES))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, N_WORD_FEATURES))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

# Define validation data


# Set up hyperparameters
# Build the input layer(s)
# Build the hidden layer(s)
# Define the output layer


HIDDEN_DIM = 100
SAVE_FOLDER = "C:/Users/nhj/Desktop/playground/Author AutoComplete/"

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, N_WORD_FEATURES), return_sequences=True))
model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(N_WORD_FEATURES)))
model.add(Activation('softmax'))
# Define the loss and validation function
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")



# (Optional) Test the forward pass

# Train model

nb_epoch = 0
while True:
    print('\n\n')
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        model.save_weights(SAVE_FOLDER + 'checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))



# Manually test results of model

def generate_text(model, length):
    ix = [np.random.randint(N_WORD_FEATURES)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, N_WORD_FEATURES))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


# Plot results of model


# Save model






