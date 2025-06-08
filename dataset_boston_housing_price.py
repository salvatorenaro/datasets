from keras.datasets import boston_housing
from keras import models, layers
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


network = models.Sequential()
network.add(layers.Dense(32, activation='relu', input_shape=(13,)))
network.add(layers.Dense(1)) 
network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = network.fit(x_train,y_train, epochs=10, batch_size=1, validation_split=0.2, verbose=0)

plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label = 'Mae')
plt.plot(history.history['val_mae'], label = 'Val Mae')
plt.xlabel('Epochs')
plt.ylabel('Mae')
plt.legend()
plt.show()