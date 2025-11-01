import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Hyperparamètres globaux
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
L2_LAMBDA = 0.001

# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Séparation en training / validation (90% / 10%)
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

# Normalisation et mise en forme
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((x_train.shape[0], 784))
x_val = x_val.reshape((x_val.shape[0], 784))
x_test = x_test.reshape((x_test.shape[0], 784))

# Liste des optimiseurs à comparer
optimizers = {
    "SGD_with_momentum": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam": keras.optimizers.Adam(learning_rate=0.001)
}

# Fonction pour créer le modèle avec régularisation et batch normalization
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,),
                           kernel_regularizer=keras.regularizers.l2(L2_LAMBDA)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Boucle d’expériences MLflow
for opt_name, optimizer in optimizers.items():
    with mlflow.start_run(run_name=f"Optimizer_Comparison_{opt_name}"):
        print(f"\n🔹 Entraînement avec optimiseur : {opt_name}")

        # Création du modèle
        model = create_model()

        # Compilation
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Enregistrement des paramètres dans MLflow
        mlflow.log_param("optimizer", opt_name)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("dropout_rate", DROPOUT_RATE)
        mlflow.log_param("l2_lambda", L2_LAMBDA)

        # Entraînement du modèle
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=2
        )

        # Évaluation sur les données de test
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ Précision sur les données de test ({opt_name}): {test_acc:.4f}")

        # Log des métriques
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)

        # Sauvegarde du modèle et du run
        model_path = f"mnist_model_{opt_name}.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        mlflow.keras.log_model(model, f"mnist-model-{opt_name}")

print("\nEntraînements terminés pour tous les optimiseurs !")
