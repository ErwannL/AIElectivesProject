
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import sys

def add_background(img, scale_factor=1.5):
    """
    Ajoute un fond (padding) blanc autour de l'image pour éviter qu'elle ne soit rognée.
    Le facteur d'échelle (scale_factor) permet d'ajuster la taille du fond.
    """
    height, width, _ = img.shape

    # Calcul des nouvelles dimensions avec une marge de sécurité
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Crée une image blanche de taille agrandie
    background = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

    # Positionne l'image au centre du nouveau fond
    start_y = (new_height - height) // 2
    start_x = (new_width - width) // 2
    background[start_y:start_y + height, start_x:start_x + width] = img

    return background

def correct_orientation(image_path):
    # Lis l'image avec OpenCV
    img = cv2.imread(image_path)

    # Ajoute un fond blanc autour de l'image pour éviter le rognage
    img = add_background(img)

    # Convertit l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applique un flou pour réduire le bruit
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Effectue la détection de contours
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Trouve les lignes dans l'image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)

        # Calcule l'angle moyen
        median_angle = np.median(angles)

        # Corrige l'orientation de l'image
        if median_angle != 0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

    # Crée le dossier 'corrected' s'il n'existe pas
    corrected_dir = "corrected"
    os.makedirs(corrected_dir, exist_ok=True)

    # Récupère le nom du fichier d'origine
    filename = os.path.basename(image_path)

    # Génère le chemin du fichier corrigé
    corrected_image_path = os.path.join(corrected_dir, f"corrected_{filename}")

    # Sauvegarde l'image corrigée
    cv2.imwrite(corrected_image_path, img)

    return corrected_image_path

def read_image(image_path):
    # Corrige l'orientation de l'image
    corrected_image_path = correct_orientation(image_path)

    # Ouvre l'image corrigée
    img = Image.open(corrected_image_path)

    # Utilise Tesseract pour faire de l'OCR
    text = pytesseract.image_to_string(img)

    return text

def main(image_path):
    text = read_image(image_path)
    print("Texte extrait de l'image :")
    print(text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python read_image.py <chemin_de_l_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)
