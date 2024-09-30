
import pytesseract
from PIL import Image
import sys

def read_image(image_path):
    # Ouvre l'image
    img = Image.open(image_path)

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
