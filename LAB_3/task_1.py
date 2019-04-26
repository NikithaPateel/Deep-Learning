from __future__ import print_function
import keras
from keras_preprocessing import sequence
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.callbacks import TensorBoard
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
model.add(Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dense(1,activation= 'relu'))
# model.add(Conv1D(filters=32,kernel_size= 3, padding='same',activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(250,input_dim = 3,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

#tbCallBack= keras.callbacks.TensorBoard(log_dir='./Graph', write_images=True)
tensorboard = TensorBoard(log_dir="logs/{}",histogram_freq= 0,write_graph=True, write_images=True)

model.compile(loss= keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

#fit the model
history = model.fit(x_train, y_train,batch_size=1024,epochs=30,verbose=1,
          validation_data=(x_test, y_test), callbacks=[tensorboard])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

model.save('mnist.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
