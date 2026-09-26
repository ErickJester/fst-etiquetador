"""Comprueba que los videos de videos_sin_etiquetar son los que dicen los CSV.

    py comprobar_videos.py
    py comprobar_videos.py --videos videos_sin_etiquetar D:/otros_videos

No modifica nada. Por cada X_rasgos.csv de etiquetas/ busca el video X.* y
revisa tres cosas:

    1. que exista un video con ese nombre
    2. que dure lo que dice el CSV (el ultimo bloque debe caer al final)
    3. que los recuadros de los tubos quepan dentro de la imagen

Si el nombre coincide pero la duracion o el tamano no, casi seguro es OTRO
video con el mismo nombre (pasa con los IMG_XXXX de distintos celulares) o una
copia recortada. Recortar clips de ese video meteria ratas equivocadas con
etiquetas ajenas, sin dar ningun error.

Al final lista tambien los videos que aun no tienen CSV: los que faltan por
etiquetar.
"""
import os
import glob
import argparse

import cv2
import pandas as pd

EXT = ('.mov', '.mp4', '.avi', '.mkv')
BLOQUE = 5.0
OK, MAL, AVISO = '[ OK ]', '[FALLA]', '[AVISO]'


def videos_en(carpetas):
    out = {}
    for d in carpetas:
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            n, e = os.path.splitext(f)
            if e.lower() in EXT:
                out.setdefault(n, os.path.join(d, f))
    return out


def medir(ruta):
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # un archivo truncado declara mas fotogramas de los que tiene: se prueba
    # a leer cerca del final
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - int(2 * fps)))
    final_ok, _ = cap.read()
    cap.release()
    return fps, n / fps if fps else 0, W, H, final_ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--videos', nargs='+', action='extend', default=[])
    p.add_argument('--etiquetas', default='etiquetas')
    a = p.parse_args()
    if not a.videos:
        a.videos = ['videos_sin_etiquetar']

    vids = videos_en(a.videos)
    csvs = sorted(os.path.basename(f)[:-11]
                  for f in glob.glob(os.path.join(a.etiquetas, '*_rasgos.csv')))
    fallas, sin_video = [], []

    print('=' * 78)
    print('  Videos contra CSV de etiquetas')
    print('=' * 78)
    for n in csvs:
        mano = os.path.exists(os.path.join(a.etiquetas, n + '_mano.csv'))
        tag = '' if mano else '  (sin _mano.csv)'
        if n not in vids:
            print('%s %-16s no esta el video%s' % (AVISO, n, tag))
            sin_video.append(n)
            continue
        r = pd.read_csv(os.path.join(a.etiquetas, n + '_rasgos.csv'), encoding='utf-8-sig')
        m = medir(vids[n])
        if m is None:
            print('%s %-16s el video no se deja abrir' % (MAL, n))
            fallas.append(n)
            continue
        fps, dur, W, H, final_ok = m
        fin_csv = float(r['t1'].max())
        problemas = []
        # el ultimo bloque incompleto se descarta, asi que al final del video
        # sobran entre 0 y ~2 bloques; mas que eso es otro video o esta cortado
        sobra = dur - fin_csv
        # el ultimo bloque se conserva si tiene al menos el 60% de sus fotogramas
        if sobra < -0.5 * BLOQUE:
            problemas.append('el CSV llega a %.0f s y el video dura %.0f s' % (fin_csv, dur))
        elif sobra > 2 * BLOQUE + 1:
            problemas.append('el video dura %.0f s y el CSV acaba en %.0f s' % (dur, fin_csv))
        if not final_ok:
            problemas.append('no se puede leer el final: archivo incompleto')
        if {'bx0', 'by0', 'bx1', 'by1'} <= set(r.columns):
            cx = (r['bx0'] + r['bx1']) / 2
            cy = (r['by0'] + r['by1']) / 2
            fuera = ((cx < 0) | (cx > W) | (cy < 0) | (cy > H)).mean()
            if fuera > 0.05:
                problemas.append('%.0f%% de los tubos caen fuera de la imagen %dx%d'
                                 % (100 * fuera, W, H))
        info = '%dx%d  %.0f s  CSV %.0f s  %d bloques%s' % (
            W, H, dur, fin_csv, r['bloque'].nunique(), tag)
        if problemas:
            print('%s %-16s %s' % (MAL, n, info))
            for pr in problemas:
                print('        -> %s' % pr)
            fallas.append(n)
        else:
            print('%s %-16s %s' % (OK, n, info))

    pendientes = [n for n in vids if n not in csvs]
    print('=' * 78)
    if fallas:
        print('  NO coinciden (no los uses para recortar clips): %s' % ', '.join(fallas))
    if sin_video:
        print('  Tienen CSV pero no encuentro su video: %s' % ', '.join(sin_video))
        print('  Si estan en otra carpeta: py comprobar_videos.py --videos videos_sin_etiquetar OTRA')
    if pendientes:
        print('  Videos sin etiquetar todavia: %s' % ', '.join(pendientes))
    if not fallas and not sin_video:
        print('  Todo coincide.')
    print('=' * 78)


if __name__ == '__main__':
    main()
