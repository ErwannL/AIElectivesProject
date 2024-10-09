
# AI_electives_project

This project uses Docker to manage Python scripts for reading and correcting image orientation, leveraging tools like Tesseract OCR and OpenCV. The project has two main components:

Text Reader: Reads an image, corrects its orientation, and extracts text using Tesseract OCR.

Image Rotator: Rotates images by a given angle without cropping them.

## Prerequisites

Docker
Docker Compose

## Project Structure

```bash
AIElectivesProject/
├── docker-compose.yml       # Docker compose file for the services
├── Dockerfile/              # Dockerfiles for each service
│   ├── Dockerfile_read      # Dockerfile for the text reader
│   └── Dockerfile_rotate    # Dockerfile for the image rotator
├── images/                  # Folder containing images to process
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── requirements.txt         # Python dependencies
├── scripts/                 # Python scripts for text reading and rotation
│   ├── read_image.py        # Script for reading and correcting images
│   └── rotate_image.py      # Script for rotating images
├── corrected/               # (Auto-generated) Folder for corrected images
└── rotated/                 # (Auto-generated) Folder for rotated images
```

## Docker Setup

Clone this repository:

```bash
git clone https://github.com/yourusername/AIElectivesProject.git
```

Navigate into the project directory:

```bash
cd AIElectivesProject
```

Build the Docker containers using Docker Compose:

```bash
docker-compose build
```

Ensure your images are placed in the images/ directory.

## Usage

### Image Rotator

This script rotates an image by a specified angle without cropping it.

To rotate an image, run the following command, replacing [angle] with the desired rotation angle and [image] with the desired image:

```bash
docker-compose run rotate [image] [angle]
```

For example, to rotate images/image_1.png by 28 degrees:

```bash
docker-compose run rotate images/image_1.png 28
```

### Text Reader

This script reads an image, corrects its orientation, and extracts text using OCR.

To use it, ensure you have an image, then run the following command with Docker:

```bash
docker-compose run text_reader [image]
```

For example, to read images/image_1.png:

```bash
docker-compose run text_reader images/image_1.png
```
