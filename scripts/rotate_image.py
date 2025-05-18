
import cv2
import os
import sys

# Function to rotate an image without cropping
def rotate_image(image, angle):  # sourcery skip: assign-if-exp
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute new image dimensions after rotation
    if angle % 180 == 0:  # If the angle is 0 or 180
        new_w, new_h = w, h
    else:  # For 90 and 270 degrees
        new_w, new_h = h, w

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the cosine and sine of the angle to adjust dimensions
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # New width and height considering the angle
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix for the offset
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    # Apply the rotation and get the rotated image
    return cv2.warpAffine(
        image,
        M,
        (
            new_w,
            new_h
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def main(image_path, rotation_angle):
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        sys.exit(1)

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load the image from '{image_path}'")
        return

    # Rotate the image with the specified angle
    rotated_img = rotate_image(image, rotation_angle)

    # Create a 'rotated' folder if it doesn't already exist
    output_dir = 'rotated'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create the output filename for the rotated image
    output_file = f"rotated_{image_name}_{rotation_angle}.png"

    # Save the rotated image in the 'rotated' directory
    cv2.imwrite(os.path.join(output_dir, output_file), rotated_img)
    print(f"Image saved as '{os.path.join(output_dir, output_file)}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rotate_image.py <image_path> <rotation_angle>")
        sys.exit(1)

    image_path = sys.argv[1]
    rotation_angle = int(sys.argv[2])
    main(image_path, rotation_angle)
