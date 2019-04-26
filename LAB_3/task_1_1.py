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
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)
top_words = 5000
# load dataset with top words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
#padding dataset to a maxium review lenght in words
max_words = 500
x_train = sequence.pad_sequences(x_train,maxlen=max_words)
x_test = sequence.pad_sequences(x_test,maxlen=max_words)

#Creating the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
# model.add(Conv1D(filters=32,kernel_size= 3, padding='same',activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32,kernel_size= 3, padding='same',activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32,kernel_size= 3, padding='same',activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
model.add(Dense(250,activation='relu'))
model.add(Dense(125,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

tbCallBack= keras.callbacks.TensorBoard(log_dir='./Graph', write_images=True)

model.compile(loss= keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(lr= 0.1),
              metrics=['accuracy'])

#fit the model
model.fit(x_train, y_train,batch_size=128,epochs=2,verbose=1,
          validation_data=(x_test, y_test), callbacks=[tbCallBack])

model.save('mnist.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


