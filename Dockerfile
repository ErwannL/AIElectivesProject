
# Utilise une image de base Python
FROM python:3.9-slim

# Installe Tesseract, OpenCV et ses dépendances
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définit le répertoire de travail
WORKDIR /app

# Copie le fichier requirements.txt et installe les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le script Python dans le conteneur
COPY read_image.py .

# Définit la commande par défaut pour le conteneur
CMD ["python", "read_image.py"]