
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import sys

def add_background(img, scale_factor=1.5):
    """
    Adds a white background (padding) around the image to avoid cropping.
    The scale_factor allows you to adjust the size of the background.
    """
    height, width, _ = img.shape

    # Calculate new dimensions with a safety margin
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Create a white image of the enlarged size
    background = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

    # Center the image on the new background
    start_y = (new_height - height) // 2
    start_x = (new_width - width) // 2
    background[start_y:start_y + height, start_x:start_x + width] = img

    return background

def correct_orientation(image_path):
    # Read the image with OpenCV
    img = cv2.imread(image_path)

    # Add a white background to avoid cropping
    img = add_background(img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)

        # Calculate the median angle
        median_angle = np.median(angles)

        # Correct the image orientation
        if median_angle != 0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

    # Create 'corrected' folder if it doesn't exist
    corrected_dir = "corrected"
    os.makedirs(corrected_dir, exist_ok=True)

    # Get the original filename
    filename = os.path.basename(image_path)

    # Generate the corrected image path
    corrected_image_path = os.path.join(corrected_dir, f"corrected_{filename}")

    # Save the corrected image
    cv2.imwrite(corrected_image_path, img)

    return corrected_image_path

def read_image(image_path):
    # Correct the image orientation
    corrected_image_path = correct_orientation(image_path)

    # Open the corrected image
    img = Image.open(corrected_image_path)

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(img)

    return text

def main(image_path):
    text = read_image(image_path)
    print("Extracted text from the image:")
    print(text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)
