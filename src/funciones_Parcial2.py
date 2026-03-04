import numpy as np
from skimage import io, util
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def clahe(image):
    height, width = image.shape
    tam1 = height // 2
    tam2 = width // 2
    
    PARTE1 = image[0:tam1,0:tam2]
    PARTE2 = image[tam1:height,0:tam2]
    PARTE3 = image[0:tam1,tam2:width]
    PARTE4 = image[tam1:height,tam2:width]
    
    cliplimit = 80
    
    EP1 = ecualizacion_histograma(PARTE1)
    EP2 = ecualizacion_histograma(PARTE2)
    EP3 = ecualizacion_histograma(PARTE3)
    EP4 = ecualizacion_histograma(PARTE4)
    
    img_array = np.array(EP1)
    img_array2 = np.array(EP2)
    img_array3 = np.array(EP3)
    img_array4 = np.array(EP4)
    
    freq = np.bincount(EP1.flatten(), minlength=256)
    freq2 = np.bincount(EP2.flatten(), minlength=256)
    freq3 = np.bincount(EP3.flatten(), minlength=256)
    freq4 = np.bincount(EP4.flatten(), minlength=256)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    recorte1 = np.zeros(256)
    recorte2 = np.zeros(256)
    recorte3 = np.zeros(256)
    recorte4 = np.zeros(256)

    for i in range(256):
        if freq[i] > cliplimit:
            recorte1[i] = cliplimit
            sum1 = sum1 + (freq[i] - cliplimit)
        else:
            recorte1[i] = freq[i]    
        if freq2[i] > cliplimit:
            recorte2[i] = cliplimit
            sum2 = sum2 + (freq2[i] - cliplimit)
        else:
            recorte2[i] = freq2[i]
        if freq3[i] > cliplimit:
            recorte3[i] = cliplimit
            sum3 = sum3 + (freq3[i] - cliplimit)
        else:
            recorte3[i] = freq3[i]
        if freq4[i] > cliplimit:
            recorte4[i] = cliplimit
            sum4 = sum4 + (freq4[i] - cliplimit)
        else:
            recorte4[i] = freq4[i]

    for i in range(256):
        recorte1[i] = recorte1[i] + math.floor(sum1/256)
        recorte2[i] = recorte2[i] + math.floor(sum2/256)
        recorte3[i] = recorte3[i] + math.floor(sum3/256)
        recorte4[i] = recorte4[i] + math.floor(sum4/256)
        
    for i in range(int(sum1 - np.floor(sum1/256)*256)):
        recorte1[i] += 1
        
    for i in range(int(sum2 - np.floor(sum2/256)*256)):
        recorte2[i] += 1
        
    for i in range(int(sum3 - np.floor(sum3/256)*256)):
        recorte3[i] += 1
        
    for i in range(int(sum4 - np.floor(sum4/256)*256)):
        recorte4[i] += 1
        
    T1 = np.zeros(256)
    T2 = np.zeros(256)
    T3 = np.zeros(256)
    T4 = np.zeros(256)
    sum1 = sum2 = sum3 = sum4 = 0
    
    for i in range(256):
        sum1 = sum1 + recorte1[i]/(tam1*tam2)
        T1[i] = 255*sum1
        sum2 = sum2 + recorte2[i]/(tam1*tam2)
        T2[i] = 255*sum2 
        sum3 = sum3 + recorte3[i]/(tam1*tam2)
        T3[i] = 255*sum3 
        sum4 = sum4 + recorte4[i]/(tam1*tam2)
        T4[i] = 255*sum4 
    
    IMG = np.zeros_like(image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):

            g = image[i, j]

        # Coordenadas normalizadas globales (0 a 1)
            dy = i / (height - 1)
            dx = j / (width - 1)

        # Pesos bilineales
            w1 = (1 - dx) * (1 - dy)   # superior izquierda (T1)
            w2 = dx * (1 - dy)         # superior derecha (T3)
            w3 = (1 - dx) * dy         # inferior izquierda (T2)
            w4 = dx * dy               # inferior derecha (T4)

            valor = (
                w1 * T1[g] +
                w2 * T3[g] +
                w3 * T2[g] +
                w4 * T4[g]
            )

            IMG[i, j] = np.uint8(valor)
    
    return IMG

def filtrado(imagen):
    height, width = imagen.shape
    
    # random_noise devuelve flotantes entre 0.0 y 1.0
    J_flotante = util.random_noise(imagen, mode='s&p', amount=0.05)
    
    # Re-escalamos de vuelta al rango 0-255 y mantenemos formato float64 para los cálculos
    J = (J_flotante * 255).astype(np.float64)
    
    var_total = np.var(J)
    
    # Inicializamos ImgR
    ImgR = np.zeros((height, width), dtype=np.float64)

    # Si la imagen no tiene varianza (es plana/completamente negra como el ejemplo), 
    # devolvemos la imagen con ruido tal cual.
    if var_total == 0:
        return J 
        
    Img1A = np.zeros((height + 2, width + 2), dtype=np.float64)
    Img1A[1:height + 1, 1:width + 1] = J
    
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            pixVeci = Img1A[x - 1: x + 2, y - 1: y + 2]
            var_local = np.var(pixVeci)
            sumR = np.mean(pixVeci)
            
            # Prevención de división por cero y aplicación de la regla del filtro
            if var_local == 0 or var_local < var_total:
                ImgR[x-1, y-1] = sumR
            else: 
                ImgR[x-1, y-1] = J[x-1,y-1] - (var_total/var_local)*(J[x-1,y-1] - sumR)
                
    return ImgR

### -- HISTOGRAMA -- ###
def ecualizacion_histograma(image):

    # 1) Histograma nk
    nk = np.bincount(image.flatten(), minlength=256)

    # 2) Total de pixeles
    total = image.size

    # 3) Probabilidad pk
    pk = nk / total

    # 4) Suma acumulada (CDF)
    cdf = np.zeros(256)
    acumulada = 0
    for i in range(256):
        acumulada = acumulada + pk[i]
        cdf[i] = acumulada

    # 5) Transformación T
    T = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        T[i] = np.uint8(round(255 * cdf[i]))

    # 6) Aplicar transformación pixel por pixel
    H, W = image.shape
    salida = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            g = image[i, j]
            salida[i, j] = T[g]

    return salida

### -- HIGHBOOST -- ###

def highboost(image, k=1.5, ksize=5):

    # 1) Convertir a float
    f = image.astype(np.float32)

    # 2) Suavizado promedio (box)
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    f_bar = cv2.filter2D(f, -1, kernel)

    # 3) Máscara
    mask = f - f_bar

    # 4) Highboost
    g = f + k * mask

    # 5) Recorte a rango válido
    g = np.clip(g, 0, 255)
    g = g.astype(np.uint8)

    return g

#--------------------------------------------#
#---------- GRADIENTE-LAPLACIANO ------------#
#--------------------------------------------#

def gradiente_laplaciano(imagen_gris, gamma=0.8):
    """
    Aplica Laplaciano y gradiente suavizado.
    
    Parámetros:
    imagen_gris (numpy.ndarray): Imagen de entrada en 2D (escala de grises).
    gamma (float): Valor para la corrección gamma.
    
    Retorna:
    tupla: (imagen_original, laplaciano, gradiente_suavizado, imagen_final)
    """
    img = imagen_gris.astype(np.float64)

    # 2. Imagen con Laplaciano
    # Usamos máscara: [0 1 0; 1 -4 1; 0 1 0]
    kernel_lap = np.array([[0, 1, 0], 
                           [1, -4, 1], 
                           [0, 1, 0]], dtype=np.float64)
    
    # filter2D aplica la convolución 
    lap = cv2.filter2D(img, cv2.CV_64F, kernel_lap)
    c = -1
    R = img + (c * lap)

    # 3. Magnitud del Gradiente (Operador Sobel de 3x3)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag_grad = np.sqrt(gx**2 + gy**2)

    # 4. Suavizar la magnitud del gradiente (Filtro de media 5x5)
    mag_suave = cv2.blur(mag_grad, (5, 5))

    # 5. Multiplicar imagen realzada por magnitud suavizada (Máscara) 
    max_val = np.max(mag_suave)
    mag_norm = mag_suave / max_val if max_val > 0 else mag_suave
    Mask = R * mag_norm

    # 6. Sumar la máscara a la imagen original
    g = img + Mask

    # 7. Aplicar corrección Gamma
    min_g = np.min(g)
    max_g = np.max(g)
    g_norm = (g - min_g) / (max_g - min_g) if (max_g - min_g) > 0 else g
    
    g_final = g_norm ** gamma
    
    return img, lap, mag_norm, g_final

#--------------------------------------------#
#------ FILTRO DE MEDIANA ADAPTATIVO --------#
#--------------------------------------------#

def filtro_mediana(img, S_max):
    """
    Aplica el Filtro de Mediana Adaptativo a una imagen.
    
    Parámetros:
    img (numpy.ndarray): Imagen de entrada en 2D (escala de grises).
    S_max (int): Tamaño máximo permitido para la vecindad S_xy (debe ser impar).
    
    Retorna:
    numpy.ndarray: Imagen filtrada.
    """
    if S_max % 2 == 0:
        raise ValueError("El tamaño máximo de ventana S_max debe ser un número impar.")

    # Asegurar que la imagen sea de tipo entero para evitar problemas en comparaciones
    img = img.astype(np.float32)
    filas, columnas = img.shape
    
    # Crear la imagen de salida
    salida = np.copy(img)
    
    # El tamaño inicial de la ventana es 3x3
    S_inicial = 3
    
    # Calculamos el padding máximo necesario
    pad_max = S_max // 2
    
    # Añadimos padding (reflejando los bordes)
    img_pad = np.pad(img, pad_max, mode='reflect')

    # Iteramos sobre cada píxel de la imagen original
    for i in range(filas):
        for j in range(columnas):
            # Coordenadas en la imagen con padding
            i_pad = i + pad_max
            j_pad = j + pad_max
            
            # Tamaño de ventana actual
            S_xy = S_inicial
            
            while S_xy <= S_max:
                pad_actual = S_xy // 2
                
                # Extraer la vecindad actual (S_xy)
                ventana = img_pad[i_pad - pad_actual : i_pad + pad_actual + 1, 
                                  j_pad - pad_actual : j_pad + pad_actual + 1]
                
                # Nivel A
                z_min = np.min(ventana)
                z_max = np.max(ventana)
                z_med = np.median(ventana)
                z_xy = img_pad[i_pad, j_pad]
                
                # Si el valor mediano NO es un impulso (sal o pimienta)
                if z_min < z_med < z_max:
                    # Nivel B
                    # Si el píxel central NO es un impulso, lo conservamos
                    if z_min < z_xy < z_max:
                        salida[i, j] = z_xy
                    # Si el píxel central ES un impulso, lo cambiamos por la mediana
                    else:
                        salida[i, j] = z_med
                    break # Salimos del bucle while y pasamos al siguiente píxel
                    
                else:
                    # El valor mediano ES un impulso, aumentamos el tamaño de la ventana
                    S_xy += 2
                    
                    if S_xy > S_max:
                        # Si alcanzamos el tamaño máximo y la mediana sigue siendo impulso,
                        # entregamos la mediana de la ventana máxima de todas formas.
                        salida[i, j] = z_med
                        break

    return np.clip(salida, 0, 255).astype(np.uint8)

#--------------------------------------------#
#---------------- CLAHE OPT------------------#
#--------------------------------------------#
def claheOpt(image):
    height, width = image.shape
    tam1 = height // 2
    tam2 = width // 2
    
    PARTE1 = image[0:tam1, 0:tam2]
    PARTE2 = image[tam1:height, 0:tam2]
    PARTE3 = image[0:tam1, tam2:width]
    PARTE4 = image[tam1:height, tam2:width]
    
    cliplimit = 80
    
    EP1 = ecualizacion_histograma(PARTE1)
    EP2 = ecualizacion_histograma(PARTE2)
    EP3 = ecualizacion_histograma(PARTE3)
    EP4 = ecualizacion_histograma(PARTE4)
    
    freq = np.bincount(EP1.flatten(), minlength=256)
    freq2 = np.bincount(EP2.flatten(), minlength=256)
    freq3 = np.bincount(EP3.flatten(), minlength=256)
    freq4 = np.bincount(EP4.flatten(), minlength=256)

    # Vectorización del recorte
    recorte1 = np.where(freq > cliplimit, cliplimit, freq)
    sum1 = np.sum(freq - recorte1)
    
    recorte2 = np.where(freq2 > cliplimit, cliplimit, freq2)
    sum2 = np.sum(freq2 - recorte2)
    
    recorte3 = np.where(freq3 > cliplimit, cliplimit, freq3)
    sum3 = np.sum(freq3 - recorte3)
    
    recorte4 = np.where(freq4 > cliplimit, cliplimit, freq4)
    sum4 = np.sum(freq4 - recorte4)

    # Redistribución
    recorte1 += int(sum1 // 256)
    recorte2 += int(sum2 // 256)
    recorte3 += int(sum3 // 256)
    recorte4 += int(sum4 // 256)
        
    for i in range(int(sum1 % 256)): recorte1[i] += 1
    for i in range(int(sum2 % 256)): recorte2[i] += 1
    for i in range(int(sum3 % 256)): recorte3[i] += 1
    for i in range(int(sum4 % 256)): recorte4[i] += 1
        
    # Transformaciones acumuladas vectorizadas
    area = tam1 * tam2
    T1 = np.cumsum(recorte1) / area * 255
    T2 = np.cumsum(recorte2) / area * 255
    T3 = np.cumsum(recorte3) / area * 255
    T4 = np.cumsum(recorte4) / area * 255
    
    # --- VECTORIZACIÓN DE LA INTERPOLACIÓN BILINEAL ---
    # Creamos mallas de coordenadas para evitar el doble for
    dy = np.linspace(0, 1, height).reshape(height, 1)
    dx = np.linspace(0, 1, width).reshape(1, width)

    w1 = (1 - dx) * (1 - dy)
    w2 = dx * (1 - dy)
    w3 = (1 - dx) * dy
    w4 = dx * dy

    # Mapeo directo usando advanced indexing de NumPy
    val1 = T1[image]
    val2 = T2[image]
    val3 = T3[image]
    val4 = T4[image]

    IMG = w1 * val1 + w2 * val3 + w3 * val2 + w4 * val4
    
    return np.uint8(IMG)


#--------------------------------------------#
#--------FILTRO ADAPTATIVO LOCAL OPT---------#
#--------------------------------------------#

def filtradoOpt(imagen):
    # random_noise devuelve flotantes entre 0.0 y 1.0
    J_flotante = util.random_noise(imagen, mode='s&p', amount=0.05)
    J = (J_flotante * 255).astype(np.float64)
    
    var_total = np.var(J)
    if var_total == 0: return J 

    # --- VECTORIZACIÓN DEL CÁLCULO DE VARIANZA LOCAL ---
    # Usamos la propiedad matemática: Varianza = E[X^2] - (E[X])^2
    S = 3 # Tamaño de la ventana (3x3 como en tu for original)
    
    # Media local E[X]
    media_local = cv2.blur(J, (S, S))
    # Media de los cuadrados E[X^2]
    media_cuadrados = cv2.blur(J**2, (S, S))
    
    # Varianza local
    var_local = media_cuadrados - (media_local**2)
    # Evitar divisiones por cero o varianzas negativas por redondeo
    var_local = np.maximum(var_local, 1e-6)
    
    # Proporción de varianzas
    ratio = var_total / var_local
    # En el filtro adaptativo local, la relación no debe ser mayor a 1
    ratio = np.clip(ratio, 0, 1)
    
    # Aplicamos la fórmula a toda la matriz de golpe
    ImgR = J - ratio * (J - media_local)
                
    return ImgR    