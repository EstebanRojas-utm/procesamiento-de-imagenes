import cv2
import os
import funciones as fun
import random

# --- Configuración ---
input_folder = 'imagenes'
output_base = 'resultados_procesamiento'

# Definimos subcarpetas para cada técnica
carpetas = {
    'gris': '01_EscalaGris',
    'flip': '02_Volteado_Horizontal',
    'rot':  '03_Rotacion_45_Grados',
    'tras': '04_Traslacion',
    'esc':  '05_Escalamiento'
}

# Crear estructura de carpetas
for k, nombre in carpetas.items():
    path = os.path.join(output_base, nombre)
    if not os.path.exists(path):
        os.makedirs(path)

# --- Bucle de Procesamiento ---
# Procesaremos las imágenes (asegúrate de tener imagen1.jpg a imagen10.jpg)
for i in range(1, 11):
    filename = f"imagen{i}.jpg"
    input_path = os.path.join(input_folder, filename)
    
    # 1. Cargar imagen original
    img_original = cv2.imread(input_path)
    
    if img_original is not None:
        print(f"[{i}/10] Procesando {filename}...")
        
        # --- A. Paso a Escala de Grises (Base para lo demás) ---
        img_gris = fun.manual_rgb_a_gris(img_original)
        cv2.imwrite(os.path.join(output_base, carpetas['gris'], filename), img_gris)
        
        # --- B. Volteado (Flipping) ---
        # Ejemplo: Horizontal
        img_flip = fun.volteado(img_gris, modo='h')
        cv2.imwrite(os.path.join(output_base, carpetas['flip'], filename), img_flip)
        
        # --- C. Rotación ---
        # Ejemplo: Rotar theta grados
        img_rot = fun.rotacion(img_gris, angulo_grados=random.randint(-360,360))
        cv2.imwrite(os.path.join(output_base, carpetas['rot'], filename), img_rot)
        
        # --- D. Traslación ---
        # Ejemplo: Mover m px a la derecha y n px abajo
        img_tras = fun.traslacion(img_gris, tx=random.randint(-200, 200), ty=random.randint(-200, 200))
        cv2.imwrite(os.path.join(output_base, carpetas['tras'], filename), img_tras)
        
        # --- E. Escalamiento ---
        
        # Ejemplo: agrandar o reducir la imagen
        factor_scale=random.uniform(0.5, 1.5)
        img_esc = fun.escalamiento(img_gris, s=factor_scale)
        cv2.imwrite(os.path.join(output_base, carpetas['esc'], filename), img_esc)


    else:
        print(f"Error: No se encontró {input_path}")

print("\n¡Proceso finalizado! Revisa la carpeta 'resultados_procesamiento'.")