"""Recorta un clip pequeño por cada bloque etiquetado, para entrenar la CNN.

    py cnn/recortar.py --videos videos_sin_etiquetar

Corre en la PC donde estan los videos. No necesita GPU: solo lee el video y
guarda recortes. Tarda alrededor de un minuto por video.

Por cada X_mano.csv de etiquetas/ busca su X_rasgos.csv (que trae el recuadro
del tubo, bx0..by1) y el video X.* en las carpetas de --videos. De cada bloque
de 5 s saca --frames fotogramas repartidos, recorta el tubo de esa rata, lo
endereza si el video se proceso con --rotar y lo reduce a gris de 128x96.

Deja en clips/ un X.npz por video. Esa carpeta es lo que se sube a Google
Drive para el cuaderno fst_cnn.ipynb.

Tambien deja clips/rasgos.csv: los 15 rasgos del etiquetador de cada clip,
que la CNN version 5 usa junto con el video. Si ya tienes los clips y solo
falta ese archivo (no relee ningun video, tarda segundos):

    py cnn/recortar.py --solo-rasgos

Opciones:
    --solo-rasgos           solo escribe clips/rasgos.csv a partir de los .npz
    --videos DIR [DIR ...]  carpetas donde buscar los videos (se puede repetir)
    --etiquetas DIR         carpeta con los CSV (por defecto etiquetas)
    --salida DIR            carpeta de clips (por defecto clips)
    --frames N              fotogramas por bloque (8)
"""
import os
import sys
import glob
import time
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import RASGOS

ALTO, ANCHO = 128, 96
EXT = ('.mov', '.mp4', '.avi', '.mkv', '.MOV', '.MP4', '.AVI', '.MKV')
GIRO = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def buscar_video(nombre, carpetas):
    for d in carpetas:
        for e in EXT:
            r = os.path.join(d, nombre + e)
            if os.path.exists(r):
                return r
    return None


def tabla(nombre, carpeta):
    r = pd.read_csv(os.path.join(carpeta, nombre + '_rasgos.csv'), encoding='utf-8-sig')
    m = pd.read_csv(os.path.join(carpeta, nombre + '_mano.csv'), encoding='utf-8-sig')
    cols = ['bloque', 'especimen', 't0', 't1', 'bx0', 'by0', 'bx1', 'by1']
    if 'rotar' in r.columns:
        cols.append('rotar')
    faltan = [c for c in cols if c not in r.columns]
    if faltan:
        raise ValueError('%s_rasgos.csv no trae %s; vuelve a pasarlo por '
                         'etiquetar.py' % (nombre, faltan))
    mc = ['bloque', 'especimen', 'clase'] + (['confianza'] if 'confianza' in m.columns else [])
    d = r[cols].merge(m[mc], on=['bloque', 'especimen'], how='inner')
    d['clase'] = d['clase'].fillna('').astype(str).str.strip().str.lower()
    d = d[d['clase'].isin(('inmovilidad', 'nado', 'escalamiento', 'activa'))]
    if 'confianza' not in d.columns:
        d['confianza'] = ''
    if 'rotar' not in d.columns:
        d['rotar'] = 0
    return d.reset_index(drop=True)


def recortar(ruta, d, n_frames):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fotograma -> [(fila, posicion)]; se lee el video una sola vez de corrido,
    # que es mucho mas rapido que saltar con set() en archivos .MOV
    pide = {}
    for i, (t0, t1) in enumerate(zip(d['t0'], d['t1'])):
        for k in range(n_frames):
            t = t0 + (k + 0.5) * (t1 - t0) / n_frames
            pide.setdefault(int(round(t * fps)), []).append((i, k))
    X = np.zeros((len(d), n_frames, ALTO, ANCHO), np.uint8)
    visto = np.zeros((len(d), n_frames), bool)
    ultimo = max(pide)
    f = 0
    while f <= ultimo:
        if f in pide:
            ok, fr = cap.read()
            if not ok:
                break
            gris = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            H, W = gris.shape
            for i, k in pide[f]:
                x0, y0 = max(0, int(d.at[i, 'bx0'])), max(0, int(d.at[i, 'by0']))
                x1, y1 = min(W, int(d.at[i, 'bx1'])), min(H, int(d.at[i, 'by1']))
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                c = gris[y0:y1, x0:x1]
                rot = int(d.at[i, 'rotar'])
                if rot in GIRO:
                    c = cv2.rotate(c, GIRO[rot])
                X[i, k] = cv2.resize(c, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)
                visto[i, k] = True
        else:
            if not cap.grab():
                break
        f += 1
    cap.release()
    return X, visto


def escribir_rasgos(salida, etiquetas):
    """clips/rasgos.csv: los rasgos del etiquetador de cada clip guardado,
    cruzados por (video, especimen, bloque). Es lo unico que hay que subir a
    Drive para pasar de la version 4 a la 5."""
    trozos = []
    for f in sorted(glob.glob(os.path.join(salida, '*.npz'))):
        z = np.load(f)
        n = str(z['video'])
        r = pd.read_csv(os.path.join(etiquetas, n + '_rasgos.csv'), encoding='utf-8-sig')
        c = pd.DataFrame(dict(especimen=z['especimen'], bloque=z['bloque']))
        c = c.merge(r[['especimen', 'bloque'] + RASGOS], on=['especimen', 'bloque'], how='left')
        c.insert(0, 'video', n)
        trozos.append(c)
    if not trozos:
        print('no hay clips en %s' % salida)
        return
    d = pd.concat(trozos, ignore_index=True)
    dest = os.path.join(salida, 'rasgos.csv')
    d.to_csv(dest, index=False, encoding='utf-8')
    print('rasgos de %d clips de %d videos -> %s' % (len(d), d['video'].nunique(), dest))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--videos', nargs='+', action='extend', default=[])
    p.add_argument('--etiquetas', default='etiquetas')
    p.add_argument('--salida', default='clips')
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--solo-rasgos', action='store_true')
    a = p.parse_args()
    if a.solo_rasgos:
        escribir_rasgos(a.salida, a.etiquetas)
        return
    if not a.videos:
        a.videos = ['videos_sin_etiquetar']
    os.makedirs(a.salida, exist_ok=True)

    nombres = sorted(os.path.basename(f)[:-9]
                     for f in glob.glob(os.path.join(a.etiquetas, '*_mano.csv')))
    hechos, faltan = 0, []
    for n in nombres:
        if not os.path.exists(os.path.join(a.etiquetas, n + '_rasgos.csv')):
            continue
        dest = os.path.join(a.salida, n + '.npz')
        if os.path.exists(dest):
            print('  %-16s ya estaba, se salta' % n)
            hechos += 1
            continue
        ruta = buscar_video(n, a.videos)
        if not ruta:
            faltan.append(n)
            print('  %-16s NO encuentro el video' % n)
            continue
        t = time.time()
        d = tabla(n, a.etiquetas)
        X, visto = recortar(ruta, d, a.frames)
        completos = visto.all(1)
        if not completos.all():
            print('  %-16s %d bloques incompletos, se descartan' % (n, (~completos).sum()))
        d, X = d[completos].reset_index(drop=True), X[completos]
        np.savez_compressed(dest, X=X, clase=d['clase'].to_numpy(str),
                            confianza=d['confianza'].fillna('').astype(str).to_numpy(str),
                            especimen=d['especimen'].to_numpy(int),
                            bloque=d['bloque'].to_numpy(int), video=n)
        # una tira de muestra para comprobar a ojo que el recorte cae sobre la rata
        muestra = np.vstack([np.hstack(list(X[j])) for j in
                             np.linspace(0, len(X) - 1, min(6, len(X))).astype(int)])
        cv2.imwrite(os.path.join(a.salida, n + '_muestra.jpg'), muestra)
        print('  %-16s %4d clips  (%.0f s)' % (n, len(X), time.time() - t))
        hechos += 1

    print('\nlistos: %d de %d videos' % (hechos, len(nombres)))
    escribir_rasgos(a.salida, a.etiquetas)
    if faltan:
        print('sin video: %s' % ', '.join(faltan))
        print('Pasa la carpeta donde esten con --videos, por ejemplo:')
        print('  py cnn/recortar.py --videos videos_sin_etiquetar D:/videos_fst')
    print('\nAHORA: mira los %s/*_muestra.jpg. Cada fila es un bloque; la rata\n'
          'debe verse dentro del recorte. Despues sube la carpeta %s a Google Drive.'
          % (a.salida, a.salida))


if __name__ == '__main__':
    main()
