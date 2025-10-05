from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Création de l'application Flask
app = Flask(__name__)

# Chargement du modèle Keras sauvegardé
model = keras.models.load_model('mnist_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Vérification des données envoyées
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Conversion de l'image en tableau NumPy
    image_data = np.array(data['image'])

    # Assurez-vous que l'image est au bon format (1, 784) et normalisée
    image_data = image_data.reshape(1, 784)
    image_data = image_data.astype("float32") / 255.0

    # Prédiction du modèle
    prediction = model.predict(image_data)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    # Retour du résultat sous forme JSON
    return jsonify({
        'prediction': predicted_class,
        'probabilities': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
