
import pytesseract
from PIL import Image
import cv2
import numpy as np
import sys

def correct_orientation(image_path):
    # Lis l'image avec OpenCV
    img = cv2.imread(image_path)
    
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
    
    # Enregistre l'image corrigée
    corrected_image_path = "corrected_image.png"
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
