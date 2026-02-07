import numpy as np
import math
import random

# --- 1. Transformación a Escala de Grises ---
def manual_rgb_a_gris(image):
    """
    Convierte RGB a Grises.
    Fórmula de luminosidad (ITU-R 601-2): Y = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) < 3: return image # Ya es gris
    
    # OpenCV carga en BGR, no en RGB
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    
    gray = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return gray.astype(np.uint8)

# --- Auxiliar: Interpolación Bilineal Vectorizada ---
def interpolacion_bilineal_vectorizada(img, x_map, y_map):
    """
    Calcula la intensidad de los píxeles basándose en sus 4 vecinos.
    x_map, y_map: Matrices con las coordenadas flotantes de origen.
    """
    h, w = img.shape
    
    # Coordenadas enteras de los 4 vecinos
    x0 = np.floor(x_map).astype(int)
    y0 = np.floor(y_map).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Aseguramos que los índices estén dentro de la imagen
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    # Obtenemos los valores de los píxeles vecinos
    Ia = img[y0, x0] # Top-left
    Ib = img[y1, x0] # Bottom-left
    Ic = img[y0, x1] # Top-right
    Id = img[y1, x1] # Bottom-right
    
    # Calculamos pesos (distancias)
    # wa es el peso para Ia, que depende de la distancia al opuesto (Id)
    wa = (x1 - x_map) * (y1 - y_map)
    wb = (x1 - x_map) * (y_map - y0)
    wc = (x_map - x0) * (y1 - y_map)
    wd = (x_map - x0) * (y_map - y0)
    
    # Interpolación final
    intensity = (wa * Ia + wb * Ib + wc * Ic + wd * Id)
    
    return intensity.astype(np.uint8)

# --- 2a. Volteado (Flipping) ---
def volteado(img, modo='h'):
    if modo == 'h':
        return img[:, ::-1] # Invierte columnas
    elif modo == 'v':
        return img[::-1, :] # Invierte filas
    return img

# --- 2b. Rotación con Interpolación ---
def rotacion(img, angulo_grados):
    h, w = img.shape
    theta = np.radians(angulo_grados)
    cx, cy = w // 2, h // 2 
    
    # Creamos una malla de coordenadas de la imagen DESTINO
    y_idxs, x_idxs = np.indices((h, w))
    
    # Centramos coordenadas
    x_shifted = x_idxs - cx
    y_shifted = y_idxs - cy
    
    # Mapeo Inverso: Para hallar qué pixel de origen va en (x,y) destino,
    # rotamos por -theta.
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    
    # Fórmula de rotación
    x_src = (x_shifted * cos_t) - (y_shifted * sin_t) + cx
    y_src = (x_shifted * sin_t) + (y_shifted * cos_t) + cy
    
    return interpolacion_bilineal_vectorizada(img, x_src, y_src)

# --- 2c. Traslación ---
def traslacion(img, tx, ty):
    h, w = img.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    new_img = np.zeros((h, w), dtype=np.uint8)
    
    # Cálculo manual de límites para slicing (mucho más rápido que loop)
    dst_y_start = max(0, ty)
    dst_y_end   = min(h, h + ty)
    dst_x_start = max(0, tx)
    dst_x_end   = min(w, w + tx)
    
    src_y_start = max(0, -ty)
    src_y_end   = min(h, h - ty)
    src_x_start = max(0, -tx)
    src_x_end   = min(w, w - tx)
    
    # Verificamos si hay superposición válida
    if (dst_y_end > dst_y_start) and (dst_x_end > dst_x_start):
        new_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            img[src_y_start:src_y_end, src_x_start:src_x_end]
            
    return new_img

# --- 2d. Escalamiento con Interpolación ---
def escalamiento(img, scale):
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Malla de coordenadas destino
    y_idxs, x_idxs = np.indices((new_h, new_w))
    
    # Mapeo Inverso: src = dst / scale
    x_src = x_idxs / scale
    y_src = y_idxs / scale
    
    # Nota: Clampeamos coordenadas para interpolar con la imagen original
    return interpolacion_bilineal_vectorizada(img, x_src, y_src)

# --- 2e. Borrado Aleatorio (Random Erase) ---
def random_erase(img, p=0.5, sl=0.02, sh=0.4, r1=0.3):
    """
    Implementación basada en Zhong et al.
    sl, sh: Proporción mínima y máxima del área a borrar.
    r1: Relación de aspecto.
    """
    if random.random() > p: return img # Probabilidad de no aplicar
    
    h, w = img.shape
    area = h * w
    
    for _ in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)
        
        h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if w_erase < w and h_erase < h:
            x1 = random.randint(0, w - w_erase)
            y1 = random.randint(0, h - h_erase)
            
            new_img = img.copy()
            # Rellenar con ruido aleatorio (0-255)
            noise = np.random.randint(0, 255, (h_erase, w_erase), dtype=np.uint8)
            new_img[y1:y1+h_erase, x1:x1+w_erase] = noise
            return new_img
            
    return img

# --- 2f. CutMix ---
def cutmix(img_dest, img_src, beta=1.0):
    """
    Implementación basada en Yun et al.
    Recorta un parche de img_src y lo pega en img_dest.
    """
    h, w = img_dest.shape
    # Redimensionamos la fuente al tamaño del destino usando nuestra función
    # (Para simplificar, usamos interpolación de vecino más cercano si difieren mucho,
    # o simplemente cortamos si es manual estricto. Aquí asumimos recorte o reajuste simple)
    if img_src.shape != img_dest.shape:
        # Ajuste simple de recorte/relleno para coincidir dimensiones
        h_src, w_src = img_src.shape
        temp = np.zeros((h, w), dtype=np.uint8)
        min_h, min_w = min(h, h_src), min(w, w_src)
        temp[:min_h, :min_w] = img_src[:min_h, :min_w]
        img_src = temp

    # Generar Lambda de distribución Beta
    lam = np.random.beta(beta, beta)
    
    # Coordenadas del Bounding Box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    new_img = img_dest.copy()
    new_img[bby1:bby2, bbx1:bbx2] = img_src[bby1:bby2, bbx1:bbx2]
    
    return new_img