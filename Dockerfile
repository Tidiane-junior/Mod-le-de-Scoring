# Utiliser une image de base officielle de Python
FROM python:3.9-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le répertoire de travail
COPY requirements.txt .

# Installer les dépendances de l'application
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va s'exécuter
EXPOSE 8000

# copier les fichiers de l'application dans le répertoire de travail
COPY . .

# # Définir la commande par défaut pour exécuter l'application
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Construire l'image docker, exécute : 
# docker build -t fastapi_image:v0 .

# Exéction du conteneur
# docker run -p 8002:8000 fastapi_image:v0

# GCLOUD
# 1- gcloud init puis suivre les instructions
# 2- gcloud auth configure-docker
# Afficher les images
# 3- docker image ls
# Renommer l'image
# 4- docker tag fastapi_image:v0 gcr.io/PROJECT_ID/fastapi_image:v0
# Conncter docker à gcloud
# 5- gcloud auth configure-docker
# Pousser l'image sur gcloud
# 6- docker push gcr.io/p7-credit-scoring/fastapi_image:v0