import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np

EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

#Lancement de la session de suivi MLflow
with mlflow.start_run():
	#Enregistrement des param tres
	mlflow.log_param("epochs", EPOCHS)
	mlflow.log_param("batch_size", BATCH_SIZE)
	mlflow.log_param("dropout_rate", DROPOUT_RATE)

#Chargement du jeu de donnees MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Normalisation des donnees
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#Redimensionnement des images pour les reseaux fully-connected
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#Construction du modele
model = keras.Sequential([keras.layers.Dense(512, activation='relu', input_shape=(784,)), keras.layers.Dropout(0.2), keras.layers.Dense(10, activation='softmax')])

#Compilation du modele
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Entrainement du modele
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

#evaluation du modele
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Precision sur les donnees de test: {test_acc:.4f}")

#Sauvegarde du modele
model.save("mnist_model.h5")
print(" Modele sauvegarde sous mnist_model.h5")

#Enregistrement des metriques
mlflow.log_metric("test_accuracy", test_acc)

#Enregistrement du mod le complet
mlflow.keras.log_model(model, "mnist-model")
