"""CNN para las conductas del FST, entrenada con los clips de recortar.py.

Se usa desde el cuaderno fst_cnn.ipynb en Google Colab. Tambien corre en
local, lento y sin GPU, para probar que todo cuadra:

    py cnn/fst_cnn.py clips --validar --epocas 1
    py cnn/fst_cnn.py clips --final --destino fst_cnn.pt

La idea: una ResNet18 preentrenada en ImageNet ve una sola imagen de 3 canales.
En vez de rojo, verde y azul se le meten 3 imagenes hechas con los 8
fotogramas del bloque (ENTRADA = 'movimiento', la de por defecto):

    1. un fotograma del medio con la luz normalizada: la postura
    2. desviacion de cada pixel a lo largo de los 5 s: cuanto se desplazo
    3. cambio medio entre fotogramas seguidos: el pataleo

El fondo quieto sale casi a cero en los canales 2 y 3, asi que la red no puede
apoyarse en como se ve cada video. La version anterior (ENTRADA = 'trios', 3
fotogramas crudos) aprendia los fondos: kappa +0.43 validando por video.
"""
import os
import glob
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix

CLASES = ['escalamiento', 'inmovilidad', 'nado']
MEDIA, DESV = 0.45, 0.225
ENTRADA = 'movimiento'   # o 'trios'
MOV = 0.08                # escala tipica del movimiento de la rata (0-1)


def cargar(carpeta, usar_dudosas=False):
    """Todos los X.npz de la carpeta -> X (N, T, H, W) uint8 y sus etiquetas."""
    X, clase, video = [], [], []
    for f in sorted(glob.glob(os.path.join(carpeta, '*.npz'))):
        z = np.load(f)
        ok = np.ones(len(z['clase']), bool)
        if not usar_dudosas:
            ok = ~np.char.startswith(np.char.lower(z['confianza'].astype(str)), 'baj')
        X.append(z['X'][ok])
        clase.append(z['clase'][ok])
        video.append(np.full(ok.sum(), str(z['video'])))
        print('  %-16s %4d clips' % (str(z['video']), ok.sum()))
    X, clase, video = np.concatenate(X), np.concatenate(clase), np.concatenate(video)
    print('%d clips, %d videos, %d fotogramas por clip, %dx%d'
          % (len(X), len(set(video)), X.shape[1], X.shape[3], X.shape[2]))
    return X, clase, video


def trios(T):
    """Trios fijos para predecir: separaciones de ~1/4 y ~1/3 del bloque."""
    out = []
    for g in sorted({max(1, T // 4), max(1, (T - 1) // 3)}):
        for s in range(0, T - 2 * g):
            out.append((s, s + g, s + 2 * g))
    return out


class Clips(Dataset):
    def __init__(self, X, y=None, entrenar=False, trio=None):
        self.X, self.y, self.entrenar, self.trio = X, y, entrenar, trio

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        c = self.X[i]
        y = -1 if self.y is None else int(self.y[i])
        if ENTRADA == 'movimiento':
            return self._movimiento(c), y
        T = c.shape[0]
        if self.entrenar:
            g = np.random.randint(1, (T - 1) // 2 + 1)
            s = np.random.randint(0, T - 2 * g)
            idx = (s, s + g, s + 2 * g)
        else:
            idx = self.trio
        x = self._aumentar(torch.from_numpy(c[list(idx)].astype(np.float32) / 255.0))
        return (x - MEDIA) / DESV, y

    def _aumentar(self, x):
        if self.entrenar:
            # los videos cambian de luz y de encuadre: que la red no se apoye
            # en el brillo ni en la posicion exacta
            x = x * np.random.uniform(0.7, 1.3) + np.random.uniform(-0.1, 0.1)
            if np.random.rand() < 0.5:
                x = x.flip(-1)
            dy, dx = np.random.randint(-6, 7, 2)
            x = torch.roll(x, (int(dy), int(dx)), (1, 2))
            x = x.clamp(0, 1)
        return x

    def _movimiento(self, c):
        f = self._aumentar(torch.from_numpy(c.astype(np.float32) / 255.0))
        medio = f[f.shape[0] // 2]
        postura = (medio - medio.mean()) / (medio.std() + 0.05)
        desplaz = f.std(0) / MOV - 1
        pataleo = (f[1:] - f[:-1]).abs().mean(0) / MOV - 1
        return torch.stack([postura, desplaz.clamp(-1, 6), pataleo.clamp(-1, 6)])


def red(preentrenada=True):
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if preentrenada else None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASES))
    return m


def dispositivo():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def entrenar(X, y, epocas=40, lote=64, lr=3e-4, log=print):
    dev = dispositivo()
    m = red().to(dev)
    # nado es 4-5 veces escalamiento. Compensarlo del todo (peso 1/frecuencia)
    # hizo que viera escalamiento en 418 bloques de nado; con la raiz se
    # compensa a medias
    frec = np.bincount(y, minlength=len(CLASES)).astype(float)
    peso = np.sqrt(frec.sum() / (len(CLASES) * np.maximum(frec, 1)))
    peso = torch.tensor(peso / peso.mean(), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=peso.to(dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(Clips(X, y, entrenar=True), batch_size=lote, shuffle=True,
                    num_workers=2 if dev.type == 'cuda' else 0, drop_last=len(X) > lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epocas * len(dl))
    amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for ep in range(epocas):
        m.train()
        t, tot, n = time.time(), 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            with torch.autocast(dev.type, enabled=amp):
                perd = crit(m(xb), yb)
            scaler.scale(perd).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += perd.item() * len(xb)
            n += len(xb)
        log('    epoca %2d/%d  perdida %.3f  (%.0f s)' % (ep + 1, epocas, tot / n, time.time() - t))
    return m


@torch.no_grad()
def predecir(m, X, lote=256):
    dev = dispositivo()
    m.eval()
    P = np.zeros((len(X), len(CLASES)))
    ts = trios(X.shape[1]) if ENTRADA == 'trios' else [None]
    for tr in ts:
        dl = DataLoader(Clips(X, trio=tr), batch_size=lote)
        out = []
        for xb, _ in dl:
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                out.append(F.softmax(m(xb.to(dev)).float(), 1).cpu().numpy())
        P += np.concatenate(out)
    return P / len(ts)


def validar(carpeta, partes=5, epocas=40, usar_dudosas=False):
    """Validacion cruzada POR VIDEO: cada video se predice con una red que
    nunca lo vio. Imprime lo mismo que entrenar.py para poder compararlos."""
    X, clase, video = cargar(carpeta, usar_dudosas)
    tres = np.isin(clase, CLASES)
    y = np.array([CLASES.index(c) if c in CLASES else -1 for c in clase])
    P = np.full((len(X), len(CLASES)), np.nan)
    gk = GroupKFold(n_splits=min(partes, len(set(video))))
    print('\nvalidando por video, en %d partes (%s)' % (gk.n_splits, dispositivo()))
    for k, (tr, te) in enumerate(gk.split(X, y, video)):
        tr = tr[tres[tr]]
        print('  parte %d: prueba con %s' % (k + 1, ', '.join(sorted(set(video[te])))))
        m = entrenar(X[tr], y[tr], epocas=epocas)
        P[te] = predecir(m, X[te])
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred = P.argmax(1)
    yt, pt, Pt = y[tres], pred[tres], P[tres]
    print('\n  adivinar la clase mas comun : %.3f' % (np.bincount(yt).max() / len(yt)))
    print('  exactitud                   : %.3f' % (yt == pt).mean())
    print('  kappa                       : %+.3f' % cohen_kappa_score(yt, pt))
    for j, c in enumerate(CLASES):
        print('  acierto en %-17s: %.3f' % (c, (pt[yt == j] == j).mean()))
    cm = confusion_matrix(yt, pt, labels=range(len(CLASES)))
    print('\n  confusion (fila = tu etiqueta, columna = la de la CNN):')
    print('  %-14s' % '' + ''.join('%12.10s' % c for c in CLASES))
    for j, c in enumerate(CLASES):
        print('  %-14s' % c + ''.join('%12d' % v for v in cm[j]))

    inm = CLASES.index('inmovilidad')
    ri, pi = clase == 'inmovilidad', P[:, inm] >= 0.5
    print('\n  nivel Porsolt, inmovil contra activa (%d clips, incluye los "activa"):' % len(X))
    print('  exactitud                   : %.3f' % (ri == pi).mean())
    print('  kappa                       : %+.3f' % cohen_kappa_score(ri, pi))

    print('\n  si descartas los clips donde la CNN duda:')
    print('  umbral     quedan  % total  exactitud')
    conf = Pt.max(1)
    for u in (0.0, 0.6, 0.7, 0.8, 0.9):
        s = conf >= u
        print('  %.2f     %6d    %4.0f%%      %.3f' % (u, s.sum(), 100 * s.mean(), (yt[s] == pt[s]).mean()))
    return P, clase, video


def final(carpeta, destino, epocas=40, usar_dudosas=False):
    """Entrena con TODOS los clips y guarda la red."""
    X, clase, video = cargar(carpeta, usar_dudosas)
    tres = np.isin(clase, CLASES)
    y = np.array([CLASES.index(c) for c in clase[tres]])
    print('\nentrenando la red final con %d clips (%s)' % (tres.sum(), dispositivo()))
    m = entrenar(X[tres], y, epocas=epocas)
    torch.save(dict(pesos=m.state_dict(), clases=CLASES, frames=X.shape[1],
                    alto=X.shape[2], ancho=X.shape[3], videos=sorted(set(video)),
                    n_clips=int(tres.sum())), destino)
    print('red -> %s' % destino)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clips')
    p.add_argument('--validar', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--destino', default='fst_cnn.pt')
    p.add_argument('--epocas', type=int, default=40)
    p.add_argument('--partes', type=int, default=5)
    p.add_argument('--usar-dudosas', action='store_true')
    a = p.parse_args()
    if not (a.validar or a.final):
        p.error('pide --validar, --final o los dos')
    if a.validar:
        validar(a.clips, a.partes, a.epocas, a.usar_dudosas)
    if a.final:
        final(a.clips, a.destino, a.epocas, a.usar_dudosas)


if __name__ == '__main__':
    main()
