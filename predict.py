import os

from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re

def detectarMatricula(frame, rectangle):
    x, y, w, h = rectangle
    recorte = frame[y:y+h, x:x+w]

    # Convertir a HSV para mejor detección de color
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)

    # Definir rangos para colores a filtrar
    # Estos rangos pueden necesitar ajustes
    rango_bajo_azul = np.array([100, 150, 0])
    rango_alto_azul = np.array([140, 255, 255])
    rango_bajo_amarillo = np.array([20, 100, 100])
    rango_alto_amarillo = np.array([30, 255, 255])
    rango_bajo_blanco = np.array([0, 0, 168])
    rango_alto_blanco = np.array([172, 111, 255])

    # Crear máscaras para filtrar colores
    mascara_azul = cv2.inRange(hsv, rango_bajo_azul, rango_alto_azul)
    mascara_amarillo = cv2.inRange(hsv, rango_bajo_amarillo, rango_alto_amarillo)

    # Combinar máscaras para filtrar azul, amarillo y blanco
    mascara_combinada = cv2.bitwise_or(mascara_azul, mascara_amarillo)

    # Aplicar máscara invertida para eliminar los colores especificados

    mascara_combinada = 255 - mascara_combinada

    recorte_filtrado = cv2.bitwise_and(recorte, recorte, mask= mascara_combinada)

    # Convertir a escala de grises y aplicar umbral
    gris = cv2.cvtColor(recorte_filtrado, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detección de texto con Tesseract
    texto = pytesseract.image_to_string(umbral, config='--psm 7')
    texto_filtrado = re.sub('[^A-Z0-9]', '', texto.upper())
    if len(texto_filtrado) > 7:
        texto_filtrado = texto_filtrado[-7:]


    return texto_filtrado.strip()


VIDEOS_DIR = os.path.join('.')

video_path = os.path.join(VIDEOS_DIR, 'video.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

licensePlateDetector_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
licensePlateDetector = YOLO(licensePlateDetector_path)  # load a custom model

threshold = 0.5

while ret:

    results = licensePlateDetector(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        rectangle = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            plate = detectarMatricula(frame, rectangle)   

            # Calcular el tamaño del texto
            (text_width, text_height), _ = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)

            # Calcular la esquina inferior izquierda del texto
            text_offset_x = int(x1)
            text_offset_y = int(y1) - 40

            # Coordenadas del rectángulo blanco (esquina superior izquierda y esquina inferior derecha)
            box_coords = ((text_offset_x, text_offset_y + 10), (text_offset_x + text_width, text_offset_y - text_height - 10))

            # Dibujar el rectángulo blanco
            cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

            # Dibujar el texto en negro
            cv2.putText(frame, plate, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()