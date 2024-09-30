import numpy as np
import os
import cv2
from skimage import io, transform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Chemins
rotated_images_dir = "rotated_images"
training_images_dir = "../files/training"  # Dossier contenant toutes les images d'entraînement
model_path = 'model_orientation.h5'

# Fonction pour charger et prétraiter les images
def load_and_preprocess_image(file_path):
    image = io.imread(file_path)
    if len(image.shape) == 2:  # Image en niveaux de gris
        image = np.stack((image,) * 3, axis=-1)  # Convertir en RGB
    image = transform.resize(image, (128, 128))  # Redimensionner à 128x128
    return image

# Charger les images depuis le dossier "rotated_images"
def load_images_from_directory(directory):
    images = []
    labels = []  # Ajouter des labels si nécessaire
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Adapter le format si nécessaire
            image_path = os.path.join(directory, filename)
            image = load_and_preprocess_image(image_path)
            images.append(image)
            # Extraire les labels à partir du nom de fichier
            angle = int(filename.split('_')[-1].replace('rot', '').replace('.png', ''))
            labels.append(angle)
    return np.array(images), np.array(labels)

# Fonction pour évaluer la précision à 2 degrés près
def evaluate_model(model, test_images, test_labels):
    predictions = model.predict(test_images)
    correct_within_2_deg = 0
    total = len(test_labels)

    for i, pred in enumerate(predictions):
        predicted_angle = np.argmax(pred) * 2  # Reconvertir en degrés
        true_angle = np.argmax(test_labels[i]) * 2

        # Vérifier si la prédiction est dans la marge de 2 degrés
        if abs(predicted_angle - true_angle) <= 2:
            correct_within_2_deg += 1

    accuracy_within_2_deg = correct_within_2_deg / total
    print(f"Précision à +/- 2 degrés : {accuracy_within_2_deg * 100:.2f}%")

# Vérifie si le modèle existe déjà ou en crée un nouveau
if os.path.exists(model_path):
    print("Loading existing model from model_orientation.h5...")
    model = load_model(model_path)
else:
    # Définir le modèle
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))  # Entrée RGB
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(180, activation='softmax'))  # 180 classes pour précision à 2 degrés

# Compiler le modèle (réinitialiser l'optimiseur)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle sur toutes les images du dossier de formation
for image_file in os.listdir(training_images_dir):
    if image_file.endswith('.png'):  # Vérifie que c'est un fichier image
        training_image_path = os.path.join(training_images_dir, image_file)
        print(f"Training with {image_file} using images from {rotated_images_dir}/{image_file.split('.')[0]}...")

        # Charger les images pour l'image d'entraînement spécifique
        training_images_dir_for_file = os.path.join(rotated_images_dir, image_file.split('.')[0])
        images, labels = load_images_from_directory(training_images_dir_for_file)

        # Ajuster les labels pour regrouper les angles par tranche de 2 degrés
        labels = [label // 2 for label in labels]  # Diviser les angles par 2
        labels = to_categorical(labels, num_classes=180)

        # Diviser les données en ensembles d'entraînement et de test
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Entraîner le modèle
        print("Entraînement du modèle...")
        model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

        # Évaluer le modèle
        evaluate_model(model, test_images, test_labels)

# Sauvegarder le modèle
model.save(model_path)
print(f"Modèle sauvegardé sous '{model_path}'")
