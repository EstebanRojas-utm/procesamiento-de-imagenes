import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
def manual_rgb_a_gris(image):
    """
    Convierte una imagen RGB a escala de grises
    usando la fórmula de luminosidad.

    """
    # 1. Obtenemos las dimensiones
    height, width, channels = image.shape
    
    # 2. Creamos una matriz vacía de ceros para la imagen gris
    # Usamos float32 para no perder precisión en la multiplicación
    gray_image = np.zeros((height, width), dtype=np.float32)
    
    # NOTA: Las imágenes cargadas con OpenCV vienen en formato BGR (Blue, Green, Red)
    # Las cargadas con Matplotlib vienen en RGB.
    # Asumiremos formato BGR porque se usa cv2.imread()
    
    # Extraemos los canales individualmente (Slicing)
    # B = Canal 0, G = Canal 1, R = Canal 2
    b_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    r_channel = image[:, :, 2]
    
    # 3. Aplicamos la ecuación pixel a pixel (Vectorizado con NumPy para eficiencia)
    # Fórmula: Y = 0.299*R + 0.587*G + 0.114*B
    gray_image = (0.299 * r_channel) + (0.587 * g_channel) + (0.114 * b_channel)
    
    # 4. Convertimos de vuelta a enteros de 8 bits (0-255)
    gray_image = gray_image.astype(np.uint8)
    
    return gray_image

# --- Bloque principal ---
# 1. Definimos las carpetas (basado en tu estructura)
input_folder = 'imagenes'
output_folder = 'imagenes_EscalaGris'

# Nos aseguramos de que la carpeta de destino exista
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Carpeta creada: {output_folder}")

# 2. Bucle para procesar las 10 imágenes
# range(1, 11) genera números del 1 al 10
for i in range(1, 11):
    filename = f"imagen{i}.jpg"
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    # Leemos la imagen
    img_original = cv2.imread(input_path)
    
    if img_original is not None:
        print(f"Procesando: {filename}...")
        
        # A. Conversión MANUAL a escala de grises
        img_gris = manual_rgb_a_gris(img_original)
        
        # B. Guardamos el resultado en la carpeta de destino
        cv2.imwrite(output_path, img_gris)
        
    else:
        print(f"Advertencia: No se pudo cargar {filename} en {input_path}")

print("¡Proceso completado! Revisa la carpeta 'imagenes_EscalaGris'.")

#Prueba
'''# Cargamos la imagen (Solo aquí usamos librerías externas)
ruta_imagen = 'imagenes/prueba.jpg' 
img_original = cv2.imread(ruta_imagen)

if img_original is not None:
    # Llamamos a NUESTRA función manual
    img_gris = manual_rgb_a_gris(img_original)

    # Mostramos resultados
    plt.figure(figsize=(10, 5))
    
    # Original (Convertimos BGR a RGB solo para mostrarla bien en matplotlib)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("Original (Color)")
    
    # Escala de Grises (cmap='gray' es necesario para que matplotlib sepa que es B/N)
    plt.subplot(1, 2, 2)
    plt.imshow(img_gris, cmap='gray')
    plt.title("Manual Grayscale")
    
    plt.show()
else:
    print("Error: No se encontró la imagen. Revisa la ruta.")'''