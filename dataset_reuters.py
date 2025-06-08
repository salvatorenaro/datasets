from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

def decode_newswire(sequence):
    return ' '.join([reverse_word_index.get(i-3, '*') for i in sequence])
  

sample_index = 0
print("Testo:")
print(decode_newswire(x_train[sample_index]))
print("\nEtichetta della categoria:", y_train[sample_index])
