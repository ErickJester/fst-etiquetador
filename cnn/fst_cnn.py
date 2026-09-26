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

ENTRADA = 'video3d' (version 4, la de por defecto) cambia de red: una
R(2+1)D-18 preentrenada con Kinetics-400, videos de acciones humanas. Es una
CNN 3D: recibe los 16 fotogramas del bloque como video y aprende el movimiento
ella misma, en vez de recibirlo resumido en 3 imagenes. Cada clip se
normaliza por su propio brillo para que no dependa de la luz de cada video.

ENTRADA = 'fusion' (version 5, la de por defecto) es la CNN 3D de la version 4
mas una rama de medidas: los 15 rasgos del etiquetador (movimiento, forma,
altura de la cabeza) y 7 medidas de las reglas del observador, sacadas del
propio clip:

    - pataleo trasero: energia de cambio en la mitad baja de la rata bajo el
      agua (media, maximo y que tan abrupto es respecto a su mediana). Fuerte
      y abrupto apunta a nado o escalamiento; suave, a inmovilidad.
    - pataleo delantero y la razon trasero/delantero.
    - nariz: cambio en la franja justo encima de la linea de agua (maximo y
      que tan abrupto). Un salto brusco apunta a escalamiento.

Es el diseno del capitulo 5 de la tesis (clasificador + heuristica de
movimiento y forma) con una red 3D en lugar de la 2D. Necesita
clips/rasgos.csv (py cnn/recortar.py --solo-rasgos).
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

# Sube este numero con cada cambio. Se imprime al cargar el modulo y al
# validar, para confirmar que Colab esta usando la copia nueva de Drive.
#   1  3 fotogramas crudos, 12 epocas
#   2  40 epocas, peso de clases con raiz
#   3  mapas de movimiento (kappa +0.574 con 8 fotogramas, +0.580 con 16)
#   4  CNN 3D R(2+1)D-18 preentrenada en Kinetics (kappa +0.625)
#   5  fusion: CNN 3D + rasgos del etiquetador + reglas de pataleo y nariz
VERSION = 5

CLASES = ['escalamiento', 'inmovilidad', 'nado']
MEDIA, DESV = 0.45, 0.225
ENTRADA = 'fusion'       # 'fusion', 'video3d', 'movimiento' o 'trios'
# lo que cambia por tipo de red: la 3D pesa mucho mas por clip
AJUSTES = {
    'fusion':     dict(epocas=15, lote=16, lr=1e-4, lote_pred=32),
    'video3d':    dict(epocas=15, lote=16, lr=1e-4, lote_pred=32),
    'movimiento': dict(epocas=40, lote=64, lr=3e-4, lote_pred=256),
    'trios':      dict(epocas=40, lote=64, lr=3e-4, lote_pred=256),
}
K_MEDIA, K_DESV = 0.43, 0.226     # normalizacion de Kinetics, en gris
MOV = 0.08                # escala tipica del movimiento de la rata (0-1)
F1_META = 0.85            # RNF-02 de la tesis: F1 >= 85 % por conducta

RASGOS = ['nq', 'rng', 'path', 'spanx', 'spany', 'me', 'me_max',
          'vert', 'elong', 'hrise', 'hmean', 'above', 'area', 'iou', 'dice']
REGLAS = ['tras_media', 'tras_max', 'tras_abrupto', 'del_media',
          'tras_sobre_del', 'nariz_max', 'nariz_abrupto']
# La caja del tubo va de 0.45 columnas de agua por encima de la superficie a
# 0.05 por debajo del fondo: la linea de agua cae al 30 % del alto del recorte.
AGUA = 0.30


print('fst_cnn version %d cargada' % VERSION)


def cargar(carpeta, usar_dudosas=False):
    """Todos los X.npz de la carpeta -> X (N, T, H, W) uint8 y sus etiquetas."""
    X, clase, video, esp, blo = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(carpeta, '*.npz'))):
        z = np.load(f)
        ok = np.ones(len(z['clase']), bool)
        if not usar_dudosas:
            ok = ~np.char.startswith(np.char.lower(z['confianza'].astype(str)), 'baj')
        X.append(z['X'][ok])
        clase.append(z['clase'][ok])
        video.append(np.full(ok.sum(), str(z['video'])))
        esp.append(z['especimen'][ok])
        blo.append(z['bloque'][ok])
        print('  %-16s %4d clips' % (str(z['video']), ok.sum()))
    X, clase, video = np.concatenate(X), np.concatenate(clase), np.concatenate(video)
    print('%d clips, %d videos, %d fotogramas por clip, %dx%d'
          % (len(X), len(set(video)), X.shape[1], X.shape[3], X.shape[2]))
    return X, clase, video, dict(especimen=np.concatenate(esp), bloque=np.concatenate(blo))


def reglas(X, lote=128):
    """Las reglas del observador, medidas en cada clip (N, T, H, W) uint8.

    No hay segmentacion fina: bajo el agua la rata es lo claro sobre agua
    oscura, asi que basta un umbral sobre el clip normalizado. La mitad baja
    de esa silueta son los cuartos traseros; la alta, el tronco y las patas
    delanteras. La nariz se mira como cambio en la franja justo encima del
    agua, donde no hay nada que se mueva salvo la cabeza y el oleaje."""
    out = np.zeros((len(X), len(REGLAS)), np.float32)
    for i0 in range(0, len(X), lote):
        f = X[i0:i0 + lote].astype(np.float32) / 255.0
        n, T, H, W = f.shape
        lo = np.percentile(f, 1, axis=(1, 2, 3), keepdims=True)
        hi = np.percentile(f, 99, axis=(1, 2, 3), keepdims=True)
        f = (f - lo) / np.maximum(hi - lo, 0.05)
        ya = int(AGUA * H)
        d = np.abs(np.diff(f, axis=1))                      # (n, T-1, H, W)

        sub = f[:, :, ya + 2:, :] > 0.5                     # rata bajo el agua
        ds = d[:, :, ya + 2:, :]
        mm = sub[:, 1:] | sub[:, :-1]
        filas = sub.sum(axis=(1, 3)).astype(np.float32)     # (n, h)
        acum = np.cumsum(filas, 1)
        medio = (acum < acum[:, -1:] / 2).sum(1)            # fila mediana
        r = np.arange(sub.shape[2])[None, :]
        baja = (r >= medio[:, None])[:, None, :, None]
        alta = ~baja

        def energia(zona):
            m = mm & zona
            return (ds * m).sum(axis=(2, 3)) / (m.sum(axis=(2, 3)) + 20.0)

        et, ed = energia(baja), energia(alta)               # (n, T-1)
        franja = d[:, :, max(0, ya - int(0.12 * H)):ya + 2, :].mean(axis=(2, 3))
        out[i0:i0 + n] = np.stack([
            et.mean(1), et.max(1), et.max(1) / (np.median(et, 1) + 0.01),
            ed.mean(1), et.mean(1) / (ed.mean(1) + 0.01),
            franja.max(1), franja.max(1) / (np.median(franja, 1) + 0.005)], 1)
    return out


def medidas(carpeta, X, video, meta):
    """Rasgos del etiquetador (clips/rasgos.csv) + reglas, en el orden de X."""
    import pandas as pd
    ruta = os.path.join(carpeta, 'rasgos.csv')
    if not os.path.exists(ruta):
        raise SystemExit('Falta %s. En la PC de los videos corre\n'
                         '    py cnn/recortar.py --solo-rasgos\n'
                         'y sube clips/rasgos.csv a Drive.' % ruta)
    r = pd.read_csv(ruta)
    k = pd.DataFrame(dict(video=video, especimen=meta['especimen'], bloque=meta['bloque']))
    R = k.merge(r, on=['video', 'especimen', 'bloque'], how='left')[RASGOS].to_numpy(np.float32)
    t = time.time()
    G = reglas(X)
    print('medidas: %d rasgos + %d reglas por clip (%.0f s); %d clips sin rasgos'
          % (len(RASGOS), len(REGLAS), time.time() - t, int(np.isnan(R).all(1).sum())))
    return np.hstack([R, G])


def trios(T):
    """Trios fijos para predecir: separaciones de ~1/4 y ~1/3 del bloque."""
    out = []
    for g in sorted({max(1, T // 4), max(1, (T - 1) // 3)}):
        for s in range(0, T - 2 * g):
            out.append((s, s + g, s + 2 * g))
    return out


class Clips(Dataset):
    def __init__(self, X, y=None, entrenar=False, trio=None, M=None):
        self.X, self.y, self.entrenar, self.trio, self.M = X, y, entrenar, trio, M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        c = self.X[i]
        y = -1 if self.y is None else int(self.y[i])
        if ENTRADA == 'movimiento':
            return self._movimiento(c), y
        if ENTRADA == 'video3d':
            return self._video(c), y
        if ENTRADA == 'fusion':
            return (self._video(c), torch.from_numpy(self.M[i])), y
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

    def _video(self, c):
        f = self._aumentar(torch.from_numpy(c.astype(np.float32) / 255.0))
        f = (f - f.mean()) / (f.std() + 0.05) * K_DESV     # brillo propio del clip
        return f.unsqueeze(0).expand(3, -1, -1, -1).contiguous()   # (3, T, H, W)


class Fusion(nn.Module):
    """CNN 3D para el video + red pequena para las medidas, unidas al final."""

    def __init__(self, n_med, preentrenada=True):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        self.cnn = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if preentrenada else None)
        self.cnn.fc = nn.Identity()
        self.med = nn.Sequential(nn.Linear(n_med, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 64), nn.ReLU())
        self.cab = nn.Sequential(nn.Dropout(0.3), nn.Linear(512 + 64, len(CLASES)))
        # centrado de las medidas; se fija con los datos de entrenamiento
        self.register_buffer('mediana', torch.zeros(n_med))
        self.register_buffer('mu', torch.zeros(n_med))
        self.register_buffer('sd', torch.ones(n_med))

    def ajustar(self, M):
        med = np.nan_to_num(np.nanmedian(M, 0))
        Z = np.where(np.isnan(M), med, M)
        self.mediana.copy_(torch.tensor(med, dtype=torch.float32))
        self.mu.copy_(torch.tensor(Z.mean(0), dtype=torch.float32))
        self.sd.copy_(torch.tensor(Z.std(0) + 1e-6, dtype=torch.float32))

    def forward(self, x, m):
        m = torch.where(torch.isnan(m), self.mediana, m)
        m = ((m - self.mu) / self.sd).clamp(-5, 5)
        return self.cab(torch.cat([self.cnn(x), self.med(m)], 1))


def red(preentrenada=True, n_med=None):
    if ENTRADA == 'fusion':
        return Fusion(n_med, preentrenada)
    if ENTRADA == 'video3d':
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if preentrenada else None)
        m.fc = nn.Linear(m.fc.in_features, len(CLASES))
        return m
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if preentrenada else None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASES))
    return m


def dispositivo():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def entrenar(X, y, epocas=None, lote=None, lr=None, M=None, log=print):
    aj = AJUSTES[ENTRADA]
    epocas, lote, lr = epocas or aj['epocas'], lote or aj['lote'], lr or aj['lr']
    dev = dispositivo()
    m = red(n_med=None if M is None else M.shape[1])
    if ENTRADA == 'fusion':
        m.ajustar(M)
    m = m.to(dev)
    # nado es 4-5 veces escalamiento. Compensarlo del todo (peso 1/frecuencia)
    # hizo que viera escalamiento en 418 bloques de nado; con la raiz se
    # compensa a medias
    frec = np.bincount(y, minlength=len(CLASES)).astype(float)
    peso = np.sqrt(frec.sum() / (len(CLASES) * np.maximum(frec, 1)))
    peso = torch.tensor(peso / peso.mean(), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=peso.to(dev), label_smoothing=0.05)
    if ENTRADA == 'fusion':
        # la CNN ya viene preentrenada; las ramas nuevas aprenden desde cero y
        # necesitan un paso mayor
        grupos = [dict(params=m.cnn.parameters(), lr=lr),
                  dict(params=list(m.med.parameters()) + list(m.cab.parameters()), lr=lr * 10)]
        maximos = [lr, lr * 10]
    else:
        grupos, maximos = m.parameters(), lr
    opt = torch.optim.AdamW(grupos, lr=lr, weight_decay=1e-4)
    dl = DataLoader(Clips(X, y, entrenar=True, M=M), batch_size=lote, shuffle=True,
                    num_workers=2 if dev.type == 'cuda' else 0, drop_last=len(X) > lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, maximos, total_steps=epocas * len(dl))
    amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for ep in range(epocas):
        m.train()
        t, tot, n = time.time(), 0.0, 0
        for xb, yb in dl:
            yb = yb.to(dev)
            opt.zero_grad()
            with torch.autocast(dev.type, enabled=amp):
                perd = crit(adelante(m, xb, dev), yb)
            scaler.scale(perd).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += perd.item() * len(yb)
            n += len(yb)
        log('    epoca %2d/%d  perdida %.3f  (%.0f s)' % (ep + 1, epocas, tot / n, time.time() - t))
    return m


def adelante(m, xb, dev):
    if isinstance(xb, (list, tuple)):
        return m(xb[0].to(dev), xb[1].to(dev))
    return m(xb.to(dev))


@torch.no_grad()
def predecir(m, X, lote=None, M=None):
    lote = lote or AJUSTES[ENTRADA]['lote_pred']
    dev = dispositivo()
    m.eval()
    P = np.zeros((len(X), len(CLASES)))
    ts = trios(X.shape[1]) if ENTRADA == 'trios' else [None]
    for tr in ts:
        dl = DataLoader(Clips(X, trio=tr, M=M), batch_size=lote)
        out = []
        for xb, _ in dl:
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                out.append(F.softmax(adelante(m, xb, dev).float(), 1).cpu().numpy())
        P += np.concatenate(out)
    return P / len(ts)


def validar(carpeta, partes=5, epocas=None, usar_dudosas=False):
    """Validacion cruzada POR VIDEO: cada video se predice con una red que
    nunca lo vio. Imprime lo mismo que entrenar.py para poder compararlos."""
    X, clase, video, meta = cargar(carpeta, usar_dudosas)
    M = medidas(carpeta, X, video, meta) if ENTRADA == 'fusion' else None
    tres = np.isin(clase, CLASES)
    y = np.array([CLASES.index(c) if c in CLASES else -1 for c in clase])
    P = np.full((len(X), len(CLASES)), np.nan)
    gk = GroupKFold(n_splits=min(partes, len(set(video))))
    print('\nfst_cnn version %d, entrada %s' % (VERSION, ENTRADA))
    print('validando por video, en %d partes (%s)' % (gk.n_splits, dispositivo()))
    for k, (tr, te) in enumerate(gk.split(X, y, video)):
        tr = tr[tres[tr]]
        print('  parte %d: prueba con %s' % (k + 1, ', '.join(sorted(set(video[te])))))
        m = entrenar(X[tr], y[tr], epocas=epocas, M=None if M is None else M[tr])
        P[te] = predecir(m, X[te], M=None if M is None else M[te])
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

    informe_f1(yt, pt, ri, pi)

    print('\n  si descartas los clips donde la CNN duda:')
    print('  umbral     quedan  % total  exactitud')
    conf = Pt.max(1)
    for u in (0.0, 0.6, 0.7, 0.8, 0.9):
        s = conf >= u
        print('  %.2f     %6d    %4.0f%%      %.3f' % (u, s.sum(), 100 * s.mean(), (yt[s] == pt[s]).mean()))
    return P, clase, video


def informe_f1(yt, pt, ri, pi):
    """F1 por conducta frente a la meta de la tesis (RNF-02) y, si no se
    alcanza, al nivel de conducta activa que permite RN-13."""
    from sklearn.metrics import precision_recall_fscore_support as prfs
    pr, rc, f1, _ = prfs(yt, pt, labels=range(len(CLASES)), zero_division=0)
    ok = lambda v: 'SI' if v >= F1_META else 'no'
    print('\n  F1 por conducta (meta de la tesis: >= %.2f)' % F1_META)
    print('  %-14s %9s %8s %6s  cumple' % ('', 'precision', 'recall', 'F1'))
    for j, c in enumerate(CLASES):
        print('  %-14s %9.3f %8.3f %6.3f  %s' % (c, pr[j], rc[j], f1[j], ok(f1[j])))
    print('  %-14s %25.3f' % ('promedio', f1.mean()))
    p2, r2, f2, _ = prfs(ri, pi, labels=[True, False], zero_division=0)
    print('\n  nivel RN-13 (nado + escalamiento = conducta activa):')
    for nombre, a, b, c in (('inmovilidad', p2[0], r2[0], f2[0]), ('activa', p2[1], r2[1], f2[1])):
        print('  %-14s %9.3f %8.3f %6.3f  %s' % (nombre, a, b, c, ok(c)))
    if (f1 >= F1_META).all():
        print('\n  => cumple RNF-02 con las 3 conductas')
    elif (f2 >= F1_META).all():
        print('\n  => no cumple con 3 conductas; SI cumple al nivel de conducta activa (RN-13)')
    else:
        print('\n  => no cumple la meta ni al nivel de conducta activa')


def final(carpeta, destino, epocas=None, usar_dudosas=False):
    """Entrena con TODOS los clips y guarda la red."""
    X, clase, video, meta = cargar(carpeta, usar_dudosas)
    M = medidas(carpeta, X, video, meta) if ENTRADA == 'fusion' else None
    tres = np.isin(clase, CLASES)
    y = np.array([CLASES.index(c) for c in clase[tres]])
    print('\nfst_cnn version %d, entrada %s' % (VERSION, ENTRADA))
    print('entrenando la red final con %d clips (%s)' % (tres.sum(), dispositivo()))
    m = entrenar(X[tres], y, epocas=epocas, M=None if M is None else M[tres])
    torch.save(dict(pesos=m.state_dict(), clases=CLASES, frames=X.shape[1],
                    medidas=(RASGOS + REGLAS) if M is not None else None,
                    alto=X.shape[2], ancho=X.shape[3], videos=sorted(set(video)),
                    n_clips=int(tres.sum()), version=VERSION, entrada=ENTRADA), destino)
    print('red -> %s' % destino)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clips')
    p.add_argument('--validar', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--destino', default='fst_cnn.pt')
    p.add_argument('--epocas', type=int, default=None,
                   help='por defecto 15 para fusion y video3d, 40 para las otras')
    p.add_argument('--partes', type=int, default=5)
    p.add_argument('--usar-dudosas', action='store_true')
    p.add_argument('--entrada', default=None, choices=sorted(AJUSTES))
    a = p.parse_args()
    if not (a.validar or a.final):
        p.error('pide --validar, --final o los dos')
    global ENTRADA
    ENTRADA = a.entrada or ENTRADA
    if a.validar:
        validar(a.clips, a.partes, a.epocas, a.usar_dudosas)
    if a.final:
        final(a.clips, a.destino, a.epocas, a.usar_dudosas)


if __name__ == '__main__':
    main()
