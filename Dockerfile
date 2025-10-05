#Utiliser une image de base Python
FROM python:3.9-slim

#Definir le repertoire de travail
WORKDIR /app

#Copier le fichier des dependances et les installer
COPY requirements.txt .
RUN pip install--no-cache-dir-r requirements.txt

#Copier le reste de l’application
COPY . .

#Exposer le port de l’application Flask
EXPOSE 5000

#Commande pour d marrer l’application
CMD ["python", "app.py"]