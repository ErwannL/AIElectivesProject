import cv2
import os
import sys

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

def main(image_path, rotation_angle):
    # Lire l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Vérifier si l'image a été chargée correctement
    if image is None:
        print(f"Erreur : Impossible de charger l'image à partir de '{image_path}'")
        return

    # Faire pivoter l'image avec l'angle spécifié
    rotated_img = rotate_image(image, rotation_angle)

    # Créer un dossier 'rotated' s'il n'existe pas déjà
    output_dir = 'rotated'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extraire le nom de l'image sans l'extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Créer le nom du fichier pour l'image tourné
    output_file = f"rotated_{image_name}_{rotation_angle}.png"

    # Sauvegarder l'image pivotée dans le répertoire 'rotated'
    cv2.imwrite(os.path.join(output_dir, output_file), rotated_img)
    print(f"L'image a été sauvegardée sous '{os.path.join(output_dir, output_file)}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python rotate_image.py <chemin_de_l_image> <angle_de_rotation>")
        sys.exit(1)

    image_path = sys.argv[1]
    rotation_angle = int(sys.argv[2])
    main(image_path, rotation_angle)
