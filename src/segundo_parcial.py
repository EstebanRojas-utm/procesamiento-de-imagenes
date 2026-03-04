import numpy as np
import cv2
import os
import glob
import funciones_Parcial2 as fun

input_folder = 'resultados_procesamiento_MuchoFuego/01_EscalaGris'
output_folder = 'resultados_MuchoFuego_Parcial2'

carpetas = {
    'clahe': '01_clahe',
    'filtro': '02_filtro_adap_loc',
    'histeq': '03_ecualizacion_histograma',
    'highboost': '04_highboost',
    'gradienteLaplaciano':'05_gradienteLaplaciano',
    'filtroMediana': '06_filtroMediana'
}

# Crear carpetas
if not os.path.exists(output_folder): os.makedirs(output_folder)
for k, nombre in carpetas.items():
    path = os.path.join(output_folder, nombre)
    if not os.path.exists(path): os.makedirs(path)

# Obtener imágenes
tipos = ('*.jpg', '*.png', '*.jpeg')
lista_imagenes = []
for ext in tipos:
    lista_imagenes.extend(glob.glob(os.path.join(input_folder, ext)))

if not lista_imagenes:
    print(f"Error: No se encontraron imágenes en '{input_folder}'")
    exit()

print(f"Procesando {len(lista_imagenes)} imágenes...")

for img_path in lista_imagenes:
    filename = os.path.basename(img_path)

    #Cargar imagen original
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        print(f"No se pudo leer: {filename}")
        continue

    # 1. CLAHE
    img_clahe = fun.claheOpt(img)
    cv2.imwrite(os.path.join(output_folder, carpetas['clahe'], filename), img_clahe)
    
    # 2. Filtro adaptativo local
    img_filtrada = fun.filtradoOpt(img)
    img_filtrada = np.clip(img_filtrada, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, carpetas['filtro'], filename), img_filtrada)

    # 3. Ecualizacion de histograma
    img_histeq = fun.ecualizacion_histograma(img)
    cv2.imwrite(os.path.join(output_folder, carpetas['histeq'], filename), img_histeq)

    # 4. Highboost
    img_hb = fun.highboost(img, k=1.8, ksize=5)
    cv2.imwrite(os.path.join(output_folder, carpetas['highboost'], filename), img_hb)

    # 5. Gradiente Laplaciano
    # La función retorna 4 valores, tomamos el último (g_final).
    # Como g_final está normalizado entre 0 y 1, lo multiplicamos por 255 para guardarlo.
    _, _, _, g_final = fun.gradiente_laplaciano(img, gamma=0.8)
    g_final_uint8 = (g_final * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, carpetas['gradienteLaplaciano'], filename), g_final_uint8)

    # 6. Filtro de Mediana Adaptativo
    # S_max debe ser impar. Le ponemos 7 por defecto como en el libro.
    img_mediana = fun.filtro_mediana(img, S_max=7)
    cv2.imwrite(os.path.join(output_folder, carpetas['filtroMediana'], filename), img_mediana)

print("¡Proceso finalizado exitosamente!")
