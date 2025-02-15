# curso_qgis
Este repositorio es para carga archivos del curso de Qgis
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar la imagen satelital
image_path = "ruta_a_tu_imagen_satelital.jpg"
image = cv2.imread(image_path)

# Preprocesar la imagen (ajustar tamaño, convertir a escala de grises, etc.)
def preprocesar_imagen(imagen):
    # Cambiar el tamaño de la imagen
    imagen_redimensionada = cv2.resize(imagen, (256, 256))
    # Convertir la imagen a escala de grises (o usar imágenes en color dependiendo de tu enfoque)
    imagen_gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)
    # Normalizar la imagen
    imagen_normalizada = imagen_gris / 255.0
    return imagen_normalizada

# Cargar un modelo preentrenado (si tienes uno)
modelo = load_model("ruta_a_tu_modelo_entrenado.h5")

# Aplicar preprocesamiento
imagen_procesada = preprocesar_imagen(image)

# Suponiendo que el modelo es para clasificación de áreas de la imagen con árboles de café
# Aquí se realiza una predicción (dependiendo de la arquitectura de tu modelo)
resultado = modelo.predict(np.expand_dims(imagen_procesada, axis=0))

# Si el modelo es de clasificación por áreas (por ejemplo, un mapa de calor), deberás generar predicciones en áreas
# de la imagen, como un mapa de calor o segmentación
def segmentar_arboles(imagen, resultado_prediccion):
    # Puedes usar técnicas de umbral o segmentación de la imagen
    _, umbral = cv2.threshold(resultado_prediccion, 0.5, 1.0, cv2.THRESH_BINARY)
    # Identificar contornos (posibles árboles)
    contornos, _ = cv2.findContours((umbral * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar contornos sobre la imagen original
    imagen_contornos = cv2.drawContours(imagen.copy(), contornos, -1, (0, 255, 0), 2)
    
    return imagen_contornos, len(contornos)

# Mostrar resultados
imagen_contornos, numero_de_arboles = segmentar_arboles(image, resultado)

# Mostrar la imagen con los contornos de los árboles
plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
plt.title(f"Árboles de Café Detectados: {numero_de_arboles}")
plt.show()

# Imprimir el número de árboles detectados
print(f"Número de árboles de café detectados: {numero_de_arboles}")

