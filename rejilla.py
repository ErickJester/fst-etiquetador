"""Un fotograma del video con una rejilla numerada, para leer a ojo las cuatro
cifras que pide --geometria (videos de UN solo cilindro).

    py rejilla.py video1.avi video2.avi ... [--salida DIR] [--cada 50] [--segundo 150]

Por cada video deja <nombre>_rejilla.jpg. Se lee en la imagen:
    x0, x1   las paredes izquierda y derecha del cilindro
    agua     la superficie del agua (donde el agua toca el vidrio)
    fondo    el suelo del tubo (donde termina la columna de agua)
y se escriben en un .csv con columnas video,x0,x1,agua,fondo, o directo con
    py etiquetar.py video.avi --tubos=1 --geometria=x0,x1,agua,fondo

Despues, MIRA el _control.jpg que deja etiquetar.py: ahi se ve si las lineas
cayeron donde debian. Las cifras son a ojo y pueden estar unos pixeles corridas.
"""

import argparse
import os

import cv2


def rejilla(ruta, salida, cada, segundo):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(segundo * fps))
    ok, im = cap.read()
    cap.release()
    if not ok:
        raise SystemExit('no se pudo leer un fotograma de %s' % ruta)
    H, W = im.shape[:2]
    for x in range(0, W, cada):
        fuerte = x % (cada * 2) == 0
        cv2.line(im, (x, 0), (x, H), (0, 255, 255) if fuerte else (0, 160, 160), 1)
        if fuerte:
            cv2.putText(im, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
    for y in range(0, H, cada):
        fuerte = y % (cada * 2) == 0
        cv2.line(im, (0, y), (W, y), (255, 160, 0) if fuerte else (160, 100, 0), 1)
        if fuerte:
            cv2.putText(im, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 200, 0), 1)
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    destino = os.path.join(salida, nombre + '_rejilla.jpg')
    cv2.imwrite(destino, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('%s  (%dx%d) -> %s' % (nombre, W, H, destino))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('videos', nargs='+')
    p.add_argument('--salida', default='.')
    p.add_argument('--cada', type=int, default=50, help='separacion de la rejilla en px')
    p.add_argument('--segundo', type=float, default=150, help='que fotograma mostrar')
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    for v in a.videos:
        rejilla(v, a.salida, a.cada, a.segundo)


if __name__ == '__main__':
    main()
