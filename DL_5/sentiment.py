from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
import numpy as np

model = load_model('sentiment.h5')

new_data = ["A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"]
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(new_data)
new_text = tokenizer.texts_to_sequences(new_data)

new_text = pad_sequences(new_text,maxlen= 28)
#print(new_text)

pred = model.predict(new_text)
print(np.argmax(pred))

