"""Recorta los clips POR SEGUNDO para la CNN, a partir de tus hojas ciegas.

    py cnn/recortar_seg.py

Corre en la PC donde estan los videos, sin GPU. Es la version por segundo de
recortar.py: aquella saca un clip por bloque de 5 s etiquetado en _mano.csv;
esta saca uno por cada SEGUNDO que tu etiquetaste en revisor_segundos.html.

Que hojas lee. En etiquetas/ busca X_seg.csv, X_seg1.csv, X_seg3.csv... (una
por especimen, la convencion de revisor_segundos). De cada una toma solo los
segundos con origen humano_* (humano_ciego, humano_confirma, humano_corrige)
y conducta escrita. Los 'sin_revisar' y las casillas vacias no se usan: son
justo lo que hizo circulares las notas contra _mano.csv. Si dos hojas
etiquetan el mismo segundo de la misma rata, gana la ultima en orden
alfabetico y se avisa.

Que guarda. Para cada rata etiquetada, la PELICULA ENTERA de su tubo a 4
fotogramas por segundo (gris 128x96), no un clip por segundo: los clips de
segundos vecinos se pisan en un 75 % y guardarlos por separado ocuparia 16
veces mas. fst_cnn_seg.py arma el clip de cada segundo al vuelo: los 16
fotogramas de una ventana de 4 s centrada en el (la misma ventana que usa
segundos.py, de k-1.5 a k+2.5 s).

El recuadro del tubo se interpola en el tiempo entre los centros de bloque
de _rasgos.csv (bx0..by1), asi sigue a la camara cuando se desplaza. Los
recuadros absurdos (mas de 1.5 veces el tamano tipico, pasa en IMG_0846 y en
el bloque 1 de acu-yintang1) se ignoran y se rellenan con los vecinos.

Tambien guarda los 15 rasgos del etiquetador medidos en esa misma ventana
de 4 s (lib.segundos sobre X_cuadros.csv), que la rama de medidas usa.

Deja clips_seg/X.npz y X_muestra.jpg. Sube la carpeta clips_seg a
Mi unidad/fst/ en Google Drive.

Opciones:
    --videos DIR [DIR ...]  carpetas donde buscar los videos (se puede repetir)
    --etiquetas DIR         carpeta con los CSV (por defecto etiquetas)
    --salida DIR            carpeta de clips (por defecto clips_seg)
    --rehacer               vuelve a recortar aunque ya exista el .npz
"""
import os
import re
import sys
import glob
import time
import argparse

import cv2
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
import lib
from recortar import ALTO, ANCHO, GIRO, buscar_video

FPS_CLIP = 4                 # fotogramas por segundo que se guardan
VENTANA = 4.0                # segundos que ve la CNN alrededor de cada segundo
T = int(FPS_CLIP * VENTANA)  # fotogramas por clip: 16, como la version 4/5
CONDUCTAS_OK = ('inmovilidad', 'nado', 'escalamiento', 'activa')
HOJA = re.compile(r'^(.+)_seg(\d*)\.csv$')


def inicio(segundo):
    """Primer fotograma del clip del segundo 'segundo' (1..N) en la pelicula
    a FPS_CLIP. El fotograma j esta en t = (j + 0.5) / FPS_CLIP; la ventana
    del segundo k (de k a k+1 s) va de k + 0.5 - VENTANA/2 a k + 0.5 + VENTANA/2."""
    k = np.asarray(segundo) - 1
    return np.round((k + 0.5 - VENTANA / 2) * FPS_CLIP).astype(int)


def ventana(segundo, n_fotogramas):
    """Inicio de la ventana, corrida hacia dentro en los bordes del video:
    el primer y el ultimo segundo se miran con 4 s completos aunque no
    queden centrados."""
    return np.clip(inicio(segundo), 0, max(0, n_fotogramas - T))


def hojas(carpeta):
    """{video: [rutas de sus hojas por segundo]}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(carpeta, '*_seg*.csv'))):
        m = HOJA.match(os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def humanas(rutas):
    """Los segundos que etiqueto una persona, juntando todas las hojas del video."""
    trozos = []
    for f in rutas:
        h = pd.read_csv(f, encoding='utf-8-sig')
        if not {'especimen', 'segundo', 'conducta', 'origen'} <= set(h.columns):
            continue
        h['conducta'] = h['conducta'].fillna('').astype(str).str.strip().str.lower()
        h['origen'] = h['origen'].fillna('').astype(str)
        h = h[h['origen'].str.startswith('humano') & h['conducta'].isin(CONDUCTAS_OK)]
        h = h[['especimen', 'segundo', 'conducta', 'origen']].copy()
        h['hoja'] = os.path.basename(f)
        trozos.append(h)
    if not trozos:
        return None
    d = pd.concat(trozos, ignore_index=True)
    rep = d.duplicated(['especimen', 'segundo'], keep='last')
    if rep.any():
        print('    OJO: %d segundos aparecen en dos hojas; gana la ultima (%s)'
              % (rep.sum(), d.loc[~rep, 'hoja'].iloc[-1]))
    d = d[~rep].astype({'especimen': int, 'segundo': int})
    return d.sort_values(['especimen', 'segundo']).reset_index(drop=True)


def cajas(rasgos, especimen):
    """Funcion t -> recuadro (x0, y0, x1, y1) del tubo, interpolado entre los
    centros de bloque. Sin recuadros absurdos."""
    r = rasgos[rasgos['especimen'] == especimen].sort_values('bloque')
    b = r[['bx0', 'by0', 'bx1', 'by1']].to_numpy(float)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    mw, mh = np.median(w), np.median(h)
    ok = (w < 1.5 * mw) & (w > mw / 1.5) & (h < 1.5 * mh) & (h > mh / 1.5)
    if not ok.any():
        raise ValueError('ningun recuadro valido para el especimen %d' % especimen)
    tc = (r['bloque'].to_numpy(float) - 0.5) * lib.BLOCK_S
    tc, b = tc[ok], b[ok]

    def f(t):
        return np.stack([np.interp(t, tc, b[:, j]) for j in range(4)], -1)
    return f, int((~ok).sum())


def peliculas(ruta, rasgos, especimenes, log=print):
    """Lee el video UNA vez y devuelve {especimen: (n, ALTO, ANCHO) uint8} a
    FPS_CLIP fotogramas por segundo, n = FPS_CLIP * segundos enteros."""
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_seg = int(np.floor(total / fps + 1e-6))
    n = n_seg * FPS_CLIP
    t = (np.arange(n) + 0.5) / FPS_CLIP
    cuadro = np.round(t * fps).astype(int)
    rot = int(rasgos['rotar'].iloc[0]) if 'rotar' in rasgos.columns else 0
    caja = {}
    for e in especimenes:
        f, malas = cajas(rasgos, e)
        caja[e] = np.round(f(t)).astype(int)
        if malas:
            log('    especimen %d: %d recuadros absurdos ignorados' % (e, malas))
    out = {e: np.zeros((n, ALTO, ANCHO), np.uint8) for e in especimenes}
    pide = {}
    for j, c in enumerate(cuadro):
        pide.setdefault(int(c), []).append(j)
    leidos, f = 0, 0
    ultimo = max(pide)
    while f <= ultimo:
        if f in pide:
            ok, fr = cap.read()
            if not ok:
                break
            gris = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            H, W = gris.shape
            for j in pide[f]:
                for e in especimenes:
                    x0, y0, x1, y1 = caja[e][j]
                    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    c = gris[y0:y1, x0:x1]
                    if rot in GIRO:
                        c = cv2.rotate(c, GIRO[rot])
                    out[e][j] = cv2.resize(c, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)
                leidos = j + 1
        elif not cap.grab():
            break
        f += 1
    cap.release()
    if leidos < n:
        # el contador de fotogramas de algunos .MOV miente por unos pocos
        n = leidos - leidos % FPS_CLIP
        out = {e: v[:n] for e, v in out.items()}
    return out, n // FPS_CLIP


def rasgos_seg(carpeta, nombre):
    """Los 15 rasgos en la ventana de cada segundo, o None si no hay _cuadros.csv."""
    ruta = os.path.join(carpeta, nombre + '_cuadros.csv')
    if not os.path.exists(ruta):
        return None, None
    cu = pd.read_csv(ruta)
    paso = int(cu['paso'].iloc[0]) if 'paso' in cu.columns else None
    return lib.segundos(cu, VENTANA), paso


def muestra(F, d, destino):
    """Tira para revisar a ojo: 6 segundos repartidos, 8 de sus 16 fotogramas."""
    filas = []
    for i in np.linspace(0, len(d) - 1, min(6, len(d))).astype(int):
        e, s = int(d.at[i, 'especimen']), int(d.at[i, 'segundo'])
        j0 = int(ventana(s, len(F[e])))
        filas.append(np.hstack(list(F[e][j0:j0 + T:2])))
    cv2.imwrite(destino, np.vstack(filas))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--videos', nargs='+', action='extend', default=[])
    p.add_argument('--etiquetas', default='etiquetas')
    p.add_argument('--salida', default='clips_seg')
    p.add_argument('--rehacer', action='store_true')
    a = p.parse_args()
    if not a.videos:
        a.videos = ['videos_sin_etiquetar']
    os.makedirs(a.salida, exist_ok=True)

    total, faltan = 0, []
    for nombre, rutas in hojas(a.etiquetas).items():
        d = humanas(rutas)
        if d is None or d.empty:
            continue
        print('  %-16s %4d segundos tuyos, especimen %s (%s)'
              % (nombre, len(d), ', '.join(map(str, sorted(d['especimen'].unique()))),
                 ', '.join(os.path.basename(r) for r in rutas)))
        dest = os.path.join(a.salida, nombre + '.npz')
        if os.path.exists(dest) and not a.rehacer:
            print('    ya estaba, se salta (usa --rehacer si cambiaste la hoja)')
            total += len(d)
            continue
        rr = os.path.join(a.etiquetas, nombre + '_rasgos.csv')
        if not os.path.exists(rr):
            print('    falta %s, se salta' % rr)
            continue
        ruta = buscar_video(nombre, a.videos)
        if not ruta:
            faltan.append(nombre)
            print('    NO encuentro el video')
            continue
        t0 = time.time()
        R, paso = rasgos_seg(a.etiquetas, nombre)
        if R is None:
            print('    OJO: no hay %s_cuadros.csv; la rama de medidas ira vacia.\n'
                  '    Pasa el video por etiquetar.py para tenerlo.' % nombre)
            R = pd.DataFrame(columns=['especimen', 'segundo'] + lib.RASGOS)
        elif paso != 12:
            print('    OJO: _cuadros.csv sacado con --paso %s, no 12' % paso)
        F, n_seg = peliculas(ruta, pd.read_csv(rr, encoding='utf-8-sig'),
                             sorted(d['especimen'].unique()))
        d = d[d['segundo'] <= n_seg].reset_index(drop=True)
        d = d.merge(R[['especimen', 'segundo'] + lib.RASGOS], on=['especimen', 'segundo'], how='left')
        np.savez_compressed(
            dest, video=nombre, fps_clip=FPS_CLIP, ventana=VENTANA,
            especimen=d['especimen'].to_numpy(int), segundo=d['segundo'].to_numpy(int),
            conducta=d['conducta'].to_numpy(str), origen=d['origen'].to_numpy(str),
            hoja=d['hoja'].to_numpy(str), rasgos=d[lib.RASGOS].to_numpy(np.float32),
            **{'F%d' % e: v for e, v in F.items()})
        muestra(F, d, os.path.join(a.salida, nombre + '_muestra.jpg'))
        print('    %d segundos -> %s  (%.0f s)' % (len(d), dest, time.time() - t0))
        total += len(d)

    print('\nlisto: %d segundos etiquetados en %s' % (total, a.salida))
    if faltan:
        print('sin video: %s. Pasa la carpeta con --videos.' % ', '.join(faltan))
    print('\nAHORA: mira %s/*_muestra.jpg (la rata debe verse en cada fila) y sube\n'
          'la carpeta %s a Mi unidad/fst/ en Google Drive.' % (a.salida, a.salida))


if __name__ == '__main__':
    main()
