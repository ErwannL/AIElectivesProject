import os
import numpy as np
import cv2
from skimage import io, transform
from tensorflow.keras.models import load_model

# Chemins
input_directory = 'rotated_images/file_1'  # Dossier contenant les images à traiter
output_directory = 'output'  # Dossier où enregistrer les images redressées
model_path = 'model_orientation.h5'

# Charger le modèle
model = load_model(model_path)

# Fonction pour charger et prétraiter les images
def load_and_preprocess_image(file_path):
    image = io.imread(file_path)
    if len(image.shape) == 2:  # Image en niveaux de gris
        image = np.stack((image,) * 3, axis=-1)  # Convertir en RGB
    image = transform.resize(image, (128, 128))  # Redimensionner à 128x128
    return image

# Fonction pour redresser une image en fonction de la prédiction
def correct_image_orientation(image):
    # Prétraitement de l'image
    processed_image = load_and_preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Ajouter une dimension pour le batch

    # Prédire l'angle d'orientation
    prediction = model.predict(processed_image)
    predicted_angle = np.argmax(prediction) * 2  # Reconversion en degrés

    # Redresser l'image
    if predicted_angle != 0:
        # Appliquer une rotation
        center = (processed_image.shape[1] // 2, processed_image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, -predicted_angle, 1.0)  # Rotation dans le sens des aiguilles d'une montre
        corrected_image = cv2.warpAffine(processed_image[0], matrix, (processed_image.shape[1], processed_image.shape[0]))
    else:
        corrected_image = processed_image[0]  # Pas de rotation nécessaire

    return corrected_image

# Assurer que le dossier de sortie existe
os.makedirs(output_directory, exist_ok=True)

# Traiter toutes les images dans le dossier d'entrée
for filename in os.listdir(input_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adapter les formats d'image si nécessaire
        image_path = os.path.join(input_directory, filename)
        print(f"Processing {filename}...")

        # Corriger l'orientation de l'image
        corrected_image = correct_image_orientation(image_path)

        # Enregistrer l'image corrigée dans le dossier de sortie
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, (corrected_image * 255).astype(np.uint8))  # Convertir l'image en uint8
        print(f"Saved corrected image to {output_path}")
