"""CNN POR SEGUNDO: la misma red de la version 5, entrenada con tus hojas ciegas.

Se usa desde fst_cnn_seg.ipynb en Colab (necesita fst_cnn.py al lado) y
tambien corre en local, lento, para probar que todo cuadra:

    py cnn/fst_cnn_seg.py clips_seg --validar --epocas 1
    py cnn/fst_cnn_seg.py clips_seg --final --destino fst_cnn_seg.pt

Que cambia frente a fst_cnn.py (por bloques):

    - Un ejemplo es un SEGUNDO de una rata, no un bloque de 5 s. La red ve
      16 fotogramas de una ventana de 4 s centrada en ese segundo, igual que
      segundos.py: un segundo solo es muy poco para distinguir flotar de
      nadar despacio.
    - Las etiquetas salen SOLO de lo que tu marcaste en revisor_segundos
      (origen humano_*). Nada de _mano.csv: el 97 % de esas filas eran la
      propuesta de la maquina que nadie miro, y medir contra ellas es medir
      al modelo contra si mismo.
    - La red es la de fst_cnn.py, entrada 'fusion': CNN 3D R(2+1)D-18 +
      15 rasgos del etiquetador (medidos en la misma ventana de 4 s) + 7
      reglas de pataleo y nariz. No se reescribe nada de eso aqui.

La validacion es POR VIDEO: cada video se predice con una red que nunca lo
vio. Los segundos vecinos son casi el mismo clip (se pisan un 75 %); si se
mezclaran entre entrenamiento y prueba la nota saldria inflada.

Imprime tres niveles:
    1. segundo contra segundo (lo que pide el laboratorio)
    2. por bloque de 5 s, mayoria contra mayoria (para comparar con Porsolt
       y con las notas viejas por bloque)
    3. segundos de cada conducta por rata, tuyos contra los de la CNN
"""
import os
import glob
import time
import argparse

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import fst_cnn
from fst_cnn import CONDUCTAS

fst_cnn.ENTRADA = 'fusion'
# Formato x.x.x. Sube el numero con cada cambio; se imprime al cargar para
# confirmar que Colab lee la copia nueva de Drive.
#   0.1.0  primera version: red 'fusion' por segundo, validacion por video
# 1.0.0 queda reservada para cuando este completamente probado y listo para produccion.
VERSION_SEG = '0.1.0'
print('fst_cnn_seg version %s cargada (red de fst_cnn version %s)' % (VERSION_SEG, fst_cnn.VERSION))


class Ventanas:
    """Se comporta como un arreglo (N, T, H, W) de clips, pero sin copiarlos:
    cada clip es un trozo de la pelicula de su rata. 16 veces menos memoria
    que guardar los clips por separado."""

    def __init__(self, F, clave, j0, T):
        self.F, self.clave, self.j0, self.T = F, np.asarray(clave), np.asarray(j0), T
        a = next(iter(F.values()))
        self.shape = (len(self.j0), T, a.shape[1], a.shape[2])

    def __len__(self):
        return len(self.j0)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return self.F[self.clave[i]][self.j0[i]:self.j0[i] + self.T]
        if isinstance(i, slice):
            return np.stack([self[k] for k in range(*i.indices(len(self)))])
        return Ventanas(self.F, self.clave[i], self.j0[i], self.T)


def ventana(segundo, n, fps_clip, dur):
    """Primer fotograma del clip de cada segundo (1..N). Misma cuenta que
    recortar_seg.ventana; se repite aqui porque en Colab solo estan este
    archivo y fst_cnn.py."""
    T = int(round(fps_clip * dur))
    j = np.round((np.asarray(segundo) - 1 + 0.5 - dur / 2) * fps_clip).astype(int)
    return np.clip(j, 0, max(0, n - T)), T


def armar(F, video, especimen, segundo, fps_clip, dur):
    """Ventanas para una lista de (video, especimen, segundo). F tiene las
    peliculas con clave 'video|especimen'."""
    clave = np.array(['%s|%d' % (v, e) for v, e in zip(video, especimen)])
    j0 = np.zeros(len(clave), int)
    T = None
    for k in np.unique(clave):
        s = clave == k
        j0[s], T = ventana(segundo[s], len(F[k]), fps_clip, dur)
    return Ventanas(F, clave, j0, T)


def cargar(carpeta):
    """Todos los X.npz de clips_seg -> clips, etiquetas y rasgos."""
    F, cols = {}, dict(video=[], especimen=[], segundo=[], conducta=[], origen=[], rasgos=[])
    params = set()
    for f in sorted(glob.glob(os.path.join(carpeta, '*.npz'))):
        z = np.load(f)
        v = str(z['video'])
        params.add((int(z['fps_clip']), float(z['ventana'])))
        for e in np.unique(z['especimen']):
            F['%s|%d' % (v, e)] = z['F%d' % e]
        n = len(z['segundo'])
        cols['video'].append(np.full(n, v))
        for k in ('especimen', 'segundo', 'conducta', 'origen', 'rasgos'):
            cols[k].append(z[k] if k in z.files else z['clase'])   # clips viejos
        print('  %-16s %4d segundos, especimen %s'
              % (v, n, ', '.join(map(str, np.unique(z['especimen'])))))
    if not F:
        raise SystemExit('no hay .npz en %s; corre antes py cnn/recortar_seg.py' % carpeta)
    if len(params) > 1:
        raise SystemExit('los .npz se recortaron con ventanas distintas: %s' % params)
    fps_clip, dur = params.pop()
    d = {k: np.concatenate(v) for k, v in cols.items()}
    X = armar(F, d['video'], d['especimen'], d['segundo'], fps_clip, dur)
    print('%d segundos, %d ratas, %d videos; clips de %d fotogramas (%.0f s)'
          % (len(X), len(F), len(set(d['video'])), X.shape[1], dur))
    vc = dict(zip(*np.unique(d['conducta'], return_counts=True)))
    print('  ' + '   '.join('%s %d' % (c, vc.get(c, 0)) for c in CONDUCTAS + ['activa']))
    return X, d, (fps_clip, dur)


def medidas(X, d):
    """Rasgos de la ventana (ya en el .npz) + reglas medidas en el clip."""
    t = time.time()
    G = fst_cnn.reglas(X)
    R = d['rasgos'].astype(np.float32)
    print('medidas: %d rasgos + %d reglas por clip (%.0f s); %d segundos sin rasgos'
          % (R.shape[1], G.shape[1], time.time() - t, int(np.isnan(R).all(1).sum())))
    return np.hstack([R, G])


def validar(carpeta, partes=5, epocas=None, cada=1):
    """Validacion cruzada POR VIDEO. 'cada' = 2 entrena con uno de cada dos
    segundos (los vecinos se pisan un 75 %, se pierde poco y tarda la mitad);
    la prueba siempre usa todos."""
    X, d, _ = cargar(carpeta)
    M = medidas(X, d)
    conducta, video = d['conducta'], d['video']
    tres = np.isin(conducta, CONDUCTAS)
    y = np.array([CONDUCTAS.index(c) if c in CONDUCTAS else -1 for c in conducta])
    nv = len(set(video))
    if nv < 2:
        raise SystemExit('Hace falta etiquetar al menos 2 videos distintos para validar por video.')
    if nv < 4:
        print('\nOJO: solo %d videos. Cada red entrena con %d; la nota sirve para\n'
              'ver que el circuito funciona, no para la tesis.' % (nv, nv - 1))
    P = np.full((len(X), len(CONDUCTAS)), np.nan)
    gk = GroupKFold(n_splits=min(partes, nv))
    print('\nvalidando por video, en %d partes (%s)' % (gk.n_splits, fst_cnn.dispositivo()))
    for k, (tr, te) in enumerate(gk.split(np.zeros(len(X)), y, video)):
        tr = tr[tres[tr] & (d['segundo'][tr] % cada == 0)]
        print('  parte %d: prueba con %s, entrena con %d segundos'
              % (k + 1, ', '.join(sorted(set(video[te]))), len(tr)))
        m = fst_cnn.entrenar(X[tr], y[tr], epocas=epocas, M=M[tr])
        P[te] = fst_cnn.predecir(m, X[te], M=M[te])
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    informe(P, d)
    return P, d


def informe(P, d):
    conducta = d['conducta']
    tres = np.isin(conducta, CONDUCTAS)
    y = np.array([CONDUCTAS.index(c) if c in CONDUCTAS else -1 for c in conducta])
    pred = P.argmax(1)
    yt, pt = y[tres], pred[tres]

    print('\n=== 1. SEGUNDO CONTRA SEGUNDO (%d segundos) ===' % len(yt))
    print('  adivinar la conducta mas comun : %.3f' % (np.bincount(yt).max() / len(yt)))
    print('  exactitud                   : %.3f' % (yt == pt).mean())
    print('  kappa                       : %+.3f' % cohen_kappa_score(yt, pt))
    cm = confusion_matrix(yt, pt, labels=range(len(CONDUCTAS)))
    print('\n  confusion en segundos (fila = tu etiqueta, columna = la CNN):')
    print('  %-14s' % '' + ''.join('%12.10s' % c for c in CONDUCTAS))
    for j, c in enumerate(CONDUCTAS):
        print('  %-14s' % c + ''.join('%12d' % v for v in cm[j]))
    inm = CONDUCTAS.index('inmovilidad')
    ri, pi = conducta == 'inmovilidad', P[:, inm] >= 0.5
    print('\n  nivel Porsolt, inmovil contra activa (%d segundos, incluye los "activa"):' % len(conducta))
    print('  exactitud %.3f   kappa %+.3f' % ((ri == pi).mean(), cohen_kappa_score(ri, pi)))
    fst_cnn.informe_f1(yt, pt, ri, pi)

    print('\n=== 2. POR BLOQUE DE 5 s, mayoria contra mayoria ===')
    bloque = (d['segundo'] - 1) // 5 + 1
    tuya, maq = {}, {}
    for v, e, b, c, p, ok in zip(d['video'], d['especimen'], bloque, conducta, pred, tres):
        if ok:
            tuya.setdefault((v, e, b), []).append(c)
            maq.setdefault((v, e, b), []).append(CONDUCTAS[p])
    may = lambda xs: max(set(xs), key=xs.count)
    ks = [k for k in tuya if len(tuya[k]) == 5]      # solo bloques completos
    bt, bm = [may(tuya[k]) for k in ks], [may(maq[k]) for k in ks]
    if ks:
        print('  %d bloques completos: exactitud %.3f   kappa %+.3f'
              % (len(ks), np.mean([a == b for a, b in zip(bt, bm)]), cohen_kappa_score(bt, bm)))
        mez = np.mean([len(set(tuya[k])) > 1 for k in ks])
        print('  en el %.0f%% de tus bloques hay mas de una conducta' % (100 * mez))

    print('\n=== 3. SEGUNDOS DE CADA CONDUCTA POR RATA (tu / CNN) ===')
    print('  %-22s' % 'rata' + ''.join('%17s' % c for c in CONDUCTAS))
    for v, e in sorted(set(zip(d['video'], d['especimen']))):
        s = (d['video'] == v) & (d['especimen'] == e) & tres
        ys, ps = y[s], pred[s]
        fila = ''.join('%17s' % ('%d / %d' % ((ys == j).sum(), (ps == j).sum()))
                       for j in range(len(CONDUCTAS)))
        print('  %-22s' % ('%s #%d' % (v, e)) + fila)

    print('\n  si descartas los segundos donde la CNN duda:')
    print('  umbral     quedan  % total  exactitud')
    conf = P[tres].max(1)
    for u in (0.0, 0.6, 0.7, 0.8, 0.9):
        s = conf >= u
        if s.any():
            print('  %.2f     %6d    %4.0f%%      %.3f' % (u, s.sum(), 100 * s.mean(), (yt[s] == pt[s]).mean()))


def final(carpeta, destino, epocas=None, cada=1):
    """Entrena con TODOS los segundos y guarda la red."""
    X, d, (fps_clip, dur) = cargar(carpeta)
    M = medidas(X, d)
    tres = np.isin(d['conducta'], CONDUCTAS) & (d['segundo'] % cada == 0)
    y = np.array([CONDUCTAS.index(c) for c in d['conducta'][tres]])
    print('\nentrenando la red final con %d segundos (%s)' % (tres.sum(), fst_cnn.dispositivo()))
    m = fst_cnn.entrenar(X[tres], y, epocas=epocas, M=M[tres])
    torch.save(dict(pesos=m.state_dict(), conductas=CONDUCTAS, frames=X.shape[1],
                    medidas=fst_cnn.RASGOS + fst_cnn.REGLAS, alto=X.shape[2], ancho=X.shape[3],
                    videos=sorted(set(d['video'])), n_clips=int(tres.sum()),
                    version=fst_cnn.VERSION, entrada='fusion', unidad='segundo',
                    version_seg=VERSION_SEG, fps_clip=fps_clip, ventana=dur), destino)
    print('red -> %s' % destino)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clips')
    p.add_argument('--validar', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--destino', default='fst_cnn_seg.pt')
    p.add_argument('--epocas', type=int, default=None, help='por defecto 15')
    p.add_argument('--partes', type=int, default=5)
    p.add_argument('--cada', type=int, default=1,
                   help='entrenar con uno de cada N segundos (2 = la mitad de tiempo)')
    a = p.parse_args()
    if not (a.validar or a.final):
        p.error('pide --validar, --final o los dos')
    if a.validar:
        validar(a.clips, a.partes, a.epocas, a.cada)
    if a.final:
        final(a.clips, a.destino, a.epocas, a.cada)


if __name__ == '__main__':
    main()
