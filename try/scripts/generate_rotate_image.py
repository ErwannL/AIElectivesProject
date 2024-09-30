import cv2
import os
import numpy as np

# Fonction pour faire pivoter l'image sans crop
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculer les dimensions de la nouvelle image après rotation
    if angle % 180 == 0:  # Si l'angle est 0 ou 180
        new_w, new_h = w, h
    else:  # Pour les angles 90 et 270
        new_w, new_h = h, w

    # Calculer la matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculer le cosinus et sinus de l'angle pour ajuster les dimensions
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Nouvelle largeur et hauteur en tenant compte de l'angle
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Ajuster la matrice de rotation pour le décalage
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    # Appliquer la rotation et récupérer l'image
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image

# Générer des images tournées avec des angles connus
def generate_rotated_dataset(input_dir, output_dir):
    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcourir toutes les images dans le répertoire d'entrée
    for image_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, image_file)

        # Lire l'image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Créer un sous-répertoire pour cette image dans rotated_images
        image_name = os.path.splitext(image_file)[0]  # Extraire le nom de l'image sans l'extension
        image_output_dir = os.path.join(output_dir, image_name)

        # Créer le répertoire pour cette image si il n'existe pas
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        # Générer des rotations pour chaque angle de 0 à 359 degrés
        for angle in range(0, 360, 1):  # Créer des rotations tous les 1°
            rotated_img = rotate_image(image, angle)
            output_file = f"{image_name}_rot_{angle}.png"

            # Sauvegarder l'image pivotée dans le répertoire correspondant
            cv2.imwrite(os.path.join(image_output_dir, output_file), rotated_img)

# Appeler la fonction pour générer les images tournées
generate_rotated_dataset('../files/training/', 'rotated_images')
