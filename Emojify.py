# # Emojify!  
# 
# Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that. 
# So rather than writing:
# >"Congratulations on the promotion! Let's get coffee and talk. Love you!"   
# 
# The emojifier can automatically turn this into:
# >"Congratulations on the promotion! 👍 Let's get coffee and talk. ☕️ Love you! ❤️"
# 
# * You will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️).


import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


#importing data into the python variables
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

# maximum Length of the Sentences
maxLen = len(max(X_train, key=len).split())


# converting to One Hot Vector
Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)


# importing glove pretrained dictionaries
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


# Model

import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)


# Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
#    The output shape should be such that it can be given to `Embedding()`

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               
        sentence_words = X[i].lower().split()
        j = 0
        
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    return X_indices


# #### Inputs and outputs to the embedding layer
# 
# * The figure shows the propagation of two example sentences through the embedding layer. 
#     * Both examples have been zero-padded to a length of `max_len=5`.
#     * The word embeddings are 50 units in length.
#     * The final dimension of the representation is  `(2,max_len,50)`. 
#

# Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    embedding_layer.build((None,))
    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

# checking if function is working or not
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


# ### 2.1 - Overview of the model
# 
# Here is the Emojifier-v2 you will implement:
# 


def Emojify(input_shape, word_to_vec_map, word_to_index):
    
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings =  embedding_layer(sentence_indices)
    
    X = LSTM(units=128, return_sequences=True)(embeddings)
    
    X = Dropout(0.5)(X)
    
    X = LSTM(units=128)(X)
    
    X = Dropout(0.5)(X)
    
    X = Dense(units=5)(X)
    
    X = Activation('softmax')(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model


model = Emojify((maxLen,), word_to_vec_map, word_to_index)
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)


model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)


X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)


# This code allows you to see the mislabelled examples
print("Wrong Test values: ")
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

print()

# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
x_test = np.array(['I wrote a research paper'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))



# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
x_test = np.array(['i like her'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))


# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
x_test = np.array(['you messed up Just leave me alone'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))




