import numpy as np
import math

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

# --- 2a. Volteado (Flipping) ---
def volteado(img, modo='h'):
    """
    Voltea la imagen.
    modo 'h': Horizontal (espejo)
    modo 'v': Vertical
    """
    h, w = img.shape
    new_img = np.zeros_like(img)

    if modo == 'h':
        # Matemáticamente: x' = width - 1 - x
        new_img = img[:, ::-1]
    elif modo == 'v':
        # Matemáticamente: y' = height - 1 - y
        new_img = img[::-1, :]
    
    return new_img

# --- Función Auxiliar: Interpolación Bilineal ---
def aplicar_interpolacion_bilineal(img, x_coords, y_coords):
    """
    Calcula la intensidad de los píxeles en coordenadas flotantes (x_coords, y_coords)
    usando sus 4 vecinos más cercanos.
    """
    h, w = img.shape
    
    # Coordenadas enteras de los vecinos (Top-Left)
    x0 = np.floor(x_coords).astype(int)
    y0 = np.floor(y_coords).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Máscara para asegurar que estamos dentro de la imagen
    mask = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    
    # Inicializamos salida en negro
    output = np.zeros_like(x_coords, dtype=np.float32)
    
    # Solo calculamos donde la máscara es válida para evitar errores de índice
    x0_v, x1_v = x0[mask], x1[mask]
    y0_v, y1_v = y0[mask], y1[mask]
    
    # Pesos (distancia a los enteros)
    wa = (x1_v - x_coords[mask]) * (y1_v - y_coords[mask])
    wb = (x1_v - x_coords[mask]) * (y_coords[mask] - y0_v)
    wc = (x_coords[mask] - x0_v) * (y1_v - y_coords[mask])
    wd = (x_coords[mask] - x0_v) * (y_coords[mask] - y0_v)
    
    # Fórmula de Interpolación Bilineal:
    # I = wa*Ia + wb*Ib + wc*Ic + wd*Id
    intensity = (wa * img[y0_v, x0_v] +
                 wb * img[y1_v, x0_v] +
                 wc * img[y0_v, x1_v] +
                 wd * img[y1_v, x1_v])
    
    output[mask] = intensity
    return output.astype(np.uint8)

# --- 2b. Rotación ---
def rotacion(img, angulo_grados):
    """ Rota la imagen usando interpolación bilineal (Mapeo Inverso). """
    h, w = img.shape
    theta = np.radians(angulo_grados)
    cx, cy = w // 2, h // 2  # Centro de rotación

    # 1. Crear grid de coordenadas de la imagen DESTINO
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # 2. Aplicar Transformación Inversa para buscar en la imagen ORIGEN
    # x_src = (x - cx)cos(-th) - (y - cy)sin(-th) + cx
    # y_src = (x - cx)sin(-th) + (y - cy)cos(-th) + cy
    # Nota: cos(-th) = cos(th), sin(-th) = -sin(th)
    
    x_shifted = x_grid - cx
    y_shifted = y_grid - cy
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_src = x_shifted * cos_t + y_shifted * sin_t + cx
    y_src = -x_shifted * sin_t + y_shifted * cos_t + cy

    # 3. Interpolación
    return aplicar_interpolacion_bilineal(img, x_src, y_src)

# --- 2c. Traslación ---
def traslacion(img, tx, ty):
    """ Traslada la imagen tx píxeles a la derecha y ty abajo. """
    h, w = img.shape
    new_img = np.zeros((h, w), dtype=np.uint8)
    
    # Calculamos los límites de recorte para no salirnos del array
    # Lógica: new_img[y, x] = img[y-ty, x-tx]
    
    # Definimos coordenadas de pegado
    dst_y1 = max(0, ty)
    dst_y2 = min(h, h + ty)
    dst_x1 = max(0, tx)
    dst_x2 = min(w, w + tx)
    
    # Definimos coordenadas de copiado (origen)
    src_y1 = max(0, -ty)
    src_y2 = min(h, h - ty)
    src_x1 = max(0, -tx)
    src_x2 = min(w, w - tx)
    
    # Copiamos el bloque (vectorizado)
    # Verificamos que las dimensiones coincidan antes de copiar
    if (dst_y2 > dst_y1) and (dst_x2 > dst_x1):
        new_img[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        
    return new_img

# --- 2d. Escalamiento ---
def escalamiento(img, s):
    """ Escala la imagen por el factor s usando interpolación bilineal. """
    h, w = img.shape
    new_h = int(h * s)
    new_w = int(w * s)
    
    # 1. Crear grid de coordenadas de la imagen DESTINO
    y_grid, x_grid = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')
    
    # 2. Mapeo Inverso: x_src = x_dst / sx
    x_src = x_grid / s
    y_src = y_grid / s

    # 3. Interpolación
    return aplicar_interpolacion_bilineal(img, x_src, y_src)