# Utiliser une image de base officielle de Python
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le répertoire de travail
COPY . /requirements.txt /app

# Installer les dépendances de l'application
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va s'exécuter
EXPOSE 5000

# Définir la commande par défaut pour exécuter l'application
CMD ["python", "app.py"]