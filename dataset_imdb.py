from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import models, layers
import matplotlib.pyplot as plt


word = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=word)
x_test = pad_sequences(x_test, maxlen=word)

model = models.Sequential()
model.add(layers.Embedding(input_dim=10000, output_dim=32))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_train,epochs=10,batch_size=512,validation_split=0.2)


results = model.evaluate(x_test, y_test)
print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}')


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
