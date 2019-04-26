from __future__ import print_function
import keras
from keras_preprocessing import sequence
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.layers.embeddings import Embedding
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

dataset = pd.read_csv('heart.csv',index_col=0)
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X = (X - X.mean()) / (X.max() - X.min())
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

np.random.seed(155)
model = Sequential() # create model
model.add(Dense(40, input_dim=12, activation='relu')) # hidden layer
model.add(Dense(20, input_dim=40, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # output layer

tbCallBack= keras.callbacks.TensorBoard(log_dir='./Graph3', write_images=True)

model.compile(loss= keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adamax(),
              metrics=['accuracy'])

#fit the model
history = model.fit(X_train, Y_train,batch_size=256,epochs=30,verbose=1,
          validation_data=(X_test, Y_test), callbacks=[tbCallBack])

# my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
# my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
#                                      initial_epoch=0)
cvscores = []
print(model.summary())
scores = model.evaluate(X_test, Y_test, verbose=0)
Y_pred = model.predict(X_test)

average_precision = average_precision_score(Y_test,Y_pred)
print(average_precision)