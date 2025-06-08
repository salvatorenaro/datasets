from keras.datasets import mnist
from keras import models, layers, optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer=optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


network.fit(x_train, y_train, epochs=10, batch_size=128)


loss, acc = network.evaluate(x_test, y_test)
print(f"acc: {acc}, loss: {loss}")

idx = np.random.randint(0, y_test.shape[0])
numeri = x_test[idx].reshape(28,28)
plt.imshow(numeri, cmap=plt.cm.binary,label = "Numero")
plt.title("Dataset Mnist")
plt.xlabel("Dataset Mnist")
plt.ylabel("Dataset Mnist")
plt.show()