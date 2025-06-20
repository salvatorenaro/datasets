from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#  SALVATORE NARO



# Caricamento del Dataset Mnist

#   Set di addestramento             Set di test  
#Featuers       Labels         featuers     Labels
   
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()#->Contiene imaggini di cifre scritte a mano dallo 0 al 9

# Preprocessing dei dati
#Set di addestramento:60000 immagini di 28x28 pixel a scala di grigi con un intensivita di pixel tra 0 e 255
train_images = train_images.reshape(60000,28,28,1).astype('float32')/255 #-> Disponiamo le imaggini in un array di 60000 righe e 784 colonne (28*28) e le normalizziamo tra 0 e 1
train_labels = to_categorical(train_labels,num_classes=10)#-> Ogni etichetta viene convertita in un vettore con 10 elementi(0-9)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Set di test:10000 immagini di 28x28 pixel a scala di grigi con un intensivita di pixel tra 0 e 255
test_images = test_images.reshape(10000,28,28,1).astype('float32')/255 #-> Disponiamo le imaggini in un array di 10000 righe e 784 colonne (28*28) e le normalizziamo tra 0 e 1
test_labels = to_categorical(test_labels,num_classes=10)#-> Ogni etichetta viene convertita in un vettore con 10 elementi(0-9)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creazione del Modello
model = models.Sequential()#>Creiamo un modello sequenziale
model.add(layers.Conv2D(32,(3,3),(1,1),activation='relu',padding='same',input_shape=(28,28,1)))#->Aggiungiamo un layer convuluzionale con 32 filtri di dimensione 3,3 (standard), stride di 1 (sposta di 1 pixel in modo da analizzare ogni pixel), attivazione ReLu, padding 'same' (standard), e input_shape di 28x28 pixel con 1 canale (scala di grigi)
model.add(layers.MaxPooling2D((2,2),padding='valid'))#->Aggiungiamo un layer di pooling che riduce le dimesioni dell'immagine di 2x2 (standard)
model.add(layers.Conv2D(64,(3,3),(1,1), activation='relu', padding='same'))#->Aggiungiamo un'altro layers convuluzionale con 64 filtri
model.add(layers.MaxPooling2D((2,2),padding='valid'))#->Aggiungiamo un'altro layer di pooling che riduce le dimesioni dell'immagine di 2x2 (standard)
model.add(layers.Conv2D(128,(3,3),(1,1), activation='relu', padding='same'))#->Aggiungiamo un'altro layers convuluzionale con 128 filtri praticamente stiamo aumentando il doppio ogni volta di filtri
model.add(layers.Flatten())#->Aggiungiamo un layer  flatten che appiattisce il vettore 2d in un vettore 1d cosi possiamo passare a layer densamente connessi tra loro
model.add(layers.Dense(128,activation='relu'))#->Aggiungiamo un layer densamente connesso con 128 neuroni(unita nascoste) e attivazione ReLu
model.add(layers.Dropout(0.5))#->Aggiungiamo un layer di dropout che elimina il 50% dei neuroni per evitare l'overfitting(Il fenome in cui il nostro modello apprende benissimo i dati di addestramento ma non impara i dati che non ha mai visto prima)
model.add(layers.Dense(10,activation='softmax'))#->Aggiungiamo un layer  con 10 neuroni (uno per ogni cifra) e attivazione softmax che restituisce una probabilita per ogni cifra
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Compilazione del modello
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])#->Utilizziamo come ottimizatore(Il meccanisco con la quale la rete si aggiornera su se stessa in base alla funzione obiettivo) rmsprop , funzione obiettivo (serve per valutare le proprie prestazioni sui dati di addestramento) categorical_crossentropy, metrica accuray 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Archittettura del modello
model.summary()#->Visualizziamo l'architettura del modello
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Addestramento del modello
history = model.fit(train_images,train_labels,epochs=10,batch_size=64,validation_split=0.2)#->Epochs(numero di iterazioni complete nella quale aggiorna i propri pesi) 10, batch_size (i dati che la rete utilizza per apprendere) 64, verbose(per visualizzare l'andamento) 1 
result = model.evaluate(test_images,test_labels)#->Valutiamo il modello sui dati di test , loss e la vicinanza della rete sui dati di addestramento mentre acc Ã¨ l'accuratezza del modello sui dati di addestramento
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(f"Test Loss: {result[0]}, Test Accuracy: {result[1]}")#->Visualizziamo 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Visualizzazione dei risultati

#visualizziamo l'andamento dell'accuratezza
plt.plot(history.history['accuracy'],label = 'accuracy')#->Visualizziamo l'accuratezza 
plt.plot(history.history['val_accuracy'],label = 'val_accuracy')#->Visualizziamo il val_accuracy 
plt.title('Model Accuracy')#-> Titolo
plt.xlabel('Epochs')#->Etichetta asse x
plt.ylabel('Accuracy')#->Etichetta asse y
plt.show()#->Visualizziamo il grafico
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Visualizziamo l'andamento del loss
plt.plot(history.history['loss'],label = 'loss')#->Visualizziamo il loss 
plt.plot(history.history['val_loss'],label = 'val_loss')#->Visualizziamo il val_loss 
plt.title('Model Loss')#-> Titolo
plt.xlabel('Epochs')#->Etichetta asse x
plt.ylabel('Loss')#->Etichetta asse y
plt.show()#->Visualizziamo il grafico
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Cose il dropout?
#Il dropout Ã¨ una tecnica di regolarizzazione utilizzata per prevenire l'overfitting . Il dropout si applica su un layers,consiste nell'azzerare durante l'addestramento un certo numero di caratteristiche di output del layer in modo casuale.
#Per mitigare l'overfitting bisogna imporre dei vincoli alla rete costringendo ai suoi pesi ad assumere valori piccoli.
#La regolarizzazione dei pesi viene effettuata aggiungendo alla funzione obiettivo un costo.
#Esistono 2 tipi:
#Regolarizzazione L1: Il costo aggiunto Ã¨ proporzionale al valore assoluto dei coeficienti dei pesi
#Regolarizzazione L2: Il costo aggiunto Ã¨ proporzionale al quadrato dei coeficienti dei pesi