import cv2
import pytesseract
import sys

def preprocess_image_for_ocr(image_path):
    """
    Preprocess the image for better OCR results.
    Applies grayscale, thresholding, and resizing.
    """
    # Read the image with OpenCV
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Resize the image to increase OCR accuracy
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Save the preprocessed image
    preprocessed_image_path = "preprocessed_image.png"
    cv2.imwrite(preprocessed_image_path, gray)

    return preprocessed_image_path

def extract_rpps_area(image_path):
    """
    Extract the area of the image containing the RPPS code for more focused OCR.
    This is based on the assumption that RPPS/FRPP numbers are typically in the top-right.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Define the area where the RPPS/FRPP code typically resides (tune these values if needed)
    h, w = img.shape[:2]
    top_right_corner = img[0:int(h*0.2), int(w*0.7):w]  # Top 20% height, right 30% width

    # Save the extracted area for RPPS/FRPP detection
    rpps_image_path = "rpps_extracted.png"
    cv2.imwrite(rpps_image_path, top_right_corner)

    return rpps_image_path

def extract_text_from_image(image_path):
    """
    Main function to extract text from the image, focusing on FRPP and RPPS codes.
    """
    # Preprocess the entire image for better OCR
    preprocessed_image_path = preprocess_image_for_ocr(image_path)

    # Run Tesseract on the preprocessed full image to extract all text
    custom_config = r'--oem 3 --psm 6 -l fra'  # Adjust for French language and dense text
    full_text = pytesseract.image_to_string(preprocessed_image_path, config=custom_config)

    print("Extracted text from the full image:")
    print(full_text)

    # Extract the RPPS/FRPP area for more focused detection
    rpps_image_path = extract_rpps_area(image_path)

    # Preprocess the extracted RPPS/FRPP area
    preprocessed_rpps_image_path = preprocess_image_for_ocr(rpps_image_path)

    # Run Tesseract on the preprocessed RPPS/FRPP area
    rpps_text = pytesseract.image_to_string(preprocessed_rpps_image_path, config=custom_config)

    print("\nExtracted RPPS/FRPP text from the specific area:")
    print(rpps_text)

    return full_text, rpps_text

# Example usage
image_path = sys.argv[1]
full_text, rpps_text = extract_text_from_image(image_path)
