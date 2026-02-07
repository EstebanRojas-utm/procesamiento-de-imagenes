import cv2
import os
import glob
import random
import funciones as fun

# --- Configuración ---
input_folder = 'imagenes'
output_base = 'resultados_procesamiento'

# Subcarpetas
carpetas = {
    'gris':   '01_EscalaGris',
    'flip':   '02_Volteado_Horizontal',
    'rot':    '03_Rotacion',
    'tras':   '04_Traslacion',
    'esc':    '05_Escalamiento',
    'erase':  '06_Random_Erase',
    'cutmix': '07_CutMix'
}

# Crear carpetas
if not os.path.exists(output_base): os.makedirs(output_base)
for k, nombre in carpetas.items():
    path = os.path.join(output_base, nombre)
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
    
    # 1. Cargar imagen original
    img_original = cv2.imread(img_path)
    if img_original is None: continue

    # 2. Escala de Grises (Fundamental para el resto)
    img_gris = fun.manual_rgb_a_gris(img_original)
    cv2.imwrite(os.path.join(output_base, carpetas['gris'], filename), img_gris)
    
    # 3. Augmentations
    
    # a. Volteado Horizontal
    img_flip = fun.volteado(img_gris, modo='h')
    cv2.imwrite(os.path.join(output_base, carpetas['flip'], filename), img_flip)
    
    # b. Rotación (ej. entre -45 y 45 grados)
    img_rot = fun.rotacion(img_gris, random.randint(-45, 45))
    cv2.imwrite(os.path.join(output_base, carpetas['rot'], filename), img_rot)
    
    # c. Traslación
    tx, ty = random.randint(-44, 44), random.randint(-44, 44)
    img_tras = fun.traslacion(img_gris, tx, ty)
    cv2.imwrite(os.path.join(output_base, carpetas['tras'], filename), img_tras)
    
    # d. Escalamiento (Zoom in/out)
    scale = random.uniform(0.1, 1.9)
    img_esc = fun.escalamiento(img_gris, scale)
    cv2.imwrite(os.path.join(output_base, carpetas['esc'], filename), img_esc)
    
    # e. Random Erase 
    img_erase = fun.random_erase(img_gris, p=1.0) 
    cv2.imwrite(os.path.join(output_base, carpetas['erase'], filename), img_erase)
    
    # f. CutMix
    if len(lista_imagenes) > 1:
        partner = random.choice(lista_imagenes)
        img_partner = cv2.imread(partner)
        if img_partner is not None:
            img_partner_gris = fun.manual_rgb_a_gris(img_partner)
            img_cutmix = fun.cutmix(img_gris, img_partner_gris)
            cv2.imwrite(os.path.join(output_base, carpetas['cutmix'], filename), img_cutmix)

print("¡Proceso finalizado exitosamente!")