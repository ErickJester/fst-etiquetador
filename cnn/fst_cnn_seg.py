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
import re
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
#   0.2.0  figuras para la documentacion (datos, muestras, curva de
#          entrenamiento en vivo, matriz de confusion, precision/recall/F1
#          por conducta, lineas de tiempo, errores); la red no cambia
#   0.3.0  opciones para probar mejoras SIN cambiar lo que hace por defecto:
#          guardar las predicciones de cada corrida, suavizado en el tiempo,
#          modelo de transiciones, apilar el contexto de los segundos vecinos,
#          menos epocas / mas regularizacion / congelar capas, varias redes
#          promediadas y volteo al predecir, entrenar sin los segundos de
#          transicion, y una tabla que compara todas las corridas
# 1.0.0 queda reservada para cuando este completamente probado y listo para produccion.
VERSION_SEG = '0.3.0'
print('fst_cnn_seg version %s cargada (red de fst_cnn version %s)' % (VERSION_SEG, fst_cnn.VERSION))


# Perdida por epoca de cada entrenamiento ('parte 1', ..., 'final'). El
# cuaderno la grafica; AL_EPOCA, si no es None, se llama despues de cada epoca.
HISTORIA = {}
AL_EPOCA = None
_actual = None


def _nueva(nombre):
    global _actual
    _actual = nombre
    HISTORIA[nombre] = []


def _log(msg):
    print(msg)
    m = re.search(r'perdida\s+([0-9.]+)', msg)
    if m and _actual is not None:
        HISTORIA[_actual].append(float(m.group(1)))
        if AL_EPOCA is not None:
            AL_EPOCA(HISTORIA)


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


class Volteada:
    """Los mismos clips, volteados de izquierda a derecha (para predecir los
    dos lados y promediar). No copia nada."""

    def __init__(self, X):
        self.X, self.shape = X, X.shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return np.ascontiguousarray(np.asarray(self.X[i])[..., ::-1])


def _semilla(n):
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def transiciones(d):
    """True en los segundos de una rata donde tu etiqueta es distinta a la del
    segundo anterior o a la del siguiente (los de la frontera entre conductas)."""
    conducta = {(str(v), int(e), int(s)): c for v, e, s, c
                in zip(d['video'], d['especimen'], d['segundo'], d['conducta'])}
    out = np.zeros(len(d['segundo']), bool)
    for i, (v, e, s, c) in enumerate(zip(d['video'], d['especimen'], d['segundo'], d['conducta'])):
        for t in (int(s) - 1, int(s) + 1):
            o = conducta.get((str(v), int(e), t))
            if o is not None and o != c:
                out[i] = True
    return out


def entrenar_ajustable(X, y, epocas=None, M=None, log=print, decay=None,
                       suavizado_etiq=None, congelar=0):
    """Igual que fst_cnn.entrenar, pero con tres cosas que se pueden mover:
    decay (weight_decay de AdamW, 1e-4 por defecto), suavizado_etiq
    (label_smoothing, 0.05 por defecto) y congelar (cuantas etapas de la CNN
    preentrenada no se entrenan, de las primeras: stem, layer1, layer2, layer3)."""
    from torch import nn
    from torch.utils.data import DataLoader
    aj = fst_cnn.AJUSTES[fst_cnn.ENTRADA]
    epocas, lote, lr = epocas or aj['epocas'], aj['lote'], aj['lr']
    decay = 1e-4 if decay is None else decay
    suavizado_etiq = 0.05 if suavizado_etiq is None else suavizado_etiq
    dev = fst_cnn.dispositivo()
    m = fst_cnn.red(n_med=M.shape[1])
    m.ajustar(M)
    m = m.to(dev)
    congeladas = ['stem', 'layer1', 'layer2', 'layer3'][:congelar]
    for nombre in congeladas:
        for p in getattr(m.cnn, nombre).parameters():
            p.requires_grad = False
    frec = np.bincount(y, minlength=len(CONDUCTAS)).astype(float)
    peso = np.sqrt(frec.sum() / (len(CONDUCTAS) * np.maximum(frec, 1)))
    peso = torch.tensor(peso / peso.mean(), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=peso.to(dev), label_smoothing=suavizado_etiq)
    grupos = [dict(params=[p for p in m.cnn.parameters() if p.requires_grad], lr=lr),
              dict(params=list(m.med.parameters()) + list(m.cab.parameters()), lr=lr * 10)]
    grupos = [g for g in grupos if g['params']]
    maximos = [g['lr'] for g in grupos]
    opt = torch.optim.AdamW(grupos, lr=lr, weight_decay=decay)
    dl = DataLoader(fst_cnn.Clips(X, y, entrenar=True, M=M), batch_size=lote, shuffle=True,
                    num_workers=2 if dev.type == 'cuda' else 0, drop_last=len(X) > lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, maximos, total_steps=epocas * len(dl))
    amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for ep in range(epocas):
        m.train()
        for nombre in congeladas:                     # sin actualizar sus estadisticas
            getattr(m.cnn, nombre).eval()
        t, tot, n = time.time(), 0.0, 0
        for xb, yb in dl:
            yb = yb.to(dev)
            opt.zero_grad()
            with torch.autocast(dev.type, enabled=amp):
                perd = crit(fst_cnn.adelante(m, xb, dev), yb)
            scaler.scale(perd).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += perd.item() * len(yb)
            n += len(yb)
        log('    epoca %2d/%d  perdida %.3f  (%.0f s)' % (ep + 1, epocas, tot / n, time.time() - t))
    return m


def validar(carpeta, partes=5, epocas=None, cada=1, decay=None, suavizado_etiq=None,
            congelar=0, semillas=1, tta=False, quitar_transiciones=False,
            nombre=None, carpeta_corridas=None):
    """Validacion cruzada POR VIDEO. 'cada' = 2 entrena con uno de cada dos
    segundos (los vecinos se pisan un 75 %, se pierde poco y tarda la mitad);
    la prueba siempre usa todos.

    Con los argumentos por defecto hace exactamente lo de siempre. Las opciones
    para probar mejoras (cada una por separado o juntas):
      epocas, decay, suavizado_etiq, congelar   menos sobreajuste (ver entrenar_ajustable)
      semillas=3      entrena 3 redes por parte y promedia sus probabilidades
      tta=True        predice tambien el clip volteado y promedia
      quitar_transiciones=True   no entrena con los segundos de frontera entre
                      conductas (la prueba sigue usando todos)
      nombre, carpeta_corridas   guarda las predicciones en carpeta_corridas/nombre.npz
                      para compararlas despues con resumen_corridas()"""
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
    por_defecto = decay is None and suavizado_etiq is None and not congelar
    frontera = transiciones(d) if quitar_transiciones else np.zeros(len(X), bool)
    print('\nvalidando por video, en %d partes (%s)' % (gk.n_splits, fst_cnn.dispositivo()))
    if not (por_defecto and semillas == 1 and not tta and not quitar_transiciones):
        print('  opciones: epocas=%s decay=%s suavizado_etiq=%s congelar=%d semillas=%d tta=%s '
              'quitar_transiciones=%s' % (epocas, decay, suavizado_etiq, congelar, semillas, tta,
                                          quitar_transiciones))
    for k, (tr, te) in enumerate(gk.split(np.zeros(len(X)), y, video)):
        tr = tr[tres[tr] & (d['segundo'][tr] % cada == 0) & ~frontera[tr]]
        print('  parte %d: prueba con %s, entrena con %d segundos'
              % (k + 1, ', '.join(sorted(set(video[te]))), len(tr)))
        suma, pasadas = 0.0, 0
        for s in range(semillas):
            if semillas > 1:
                _semilla(1000 * (k + 1) + s)
            _nueva('parte %d' % (k + 1) if semillas == 1 else 'parte %d red %d' % (k + 1, s + 1))
            if por_defecto:
                m = fst_cnn.entrenar(X[tr], y[tr], epocas=epocas, M=M[tr], log=_log)
            else:
                m = entrenar_ajustable(X[tr], y[tr], epocas=epocas, M=M[tr], log=_log, decay=decay,
                                       suavizado_etiq=suavizado_etiq, congelar=congelar)
            suma = suma + fst_cnn.predecir(m, X[te], M=M[te])
            pasadas += 1
            if tta:
                suma = suma + fst_cnn.predecir(m, Volteada(X[te]), M=M[te])
                pasadas += 1
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        P[te] = suma / pasadas
    informe(P, d)
    if nombre and carpeta_corridas:
        guardar_corrida(P, d, nombre, carpeta_corridas)
    print('\nsiguiente: fst_cnn_seg.figuras_resultados(P, d, X)')
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


def final(carpeta, destino, epocas=None, cada=1, decay=None, suavizado_etiq=None, congelar=0,
          quitar_transiciones=False):
    """Entrena con TODOS los segundos y guarda la red. Las opciones son las de
    validar: usa aqui las mismas que dieron mejor resultado."""
    X, d, (fps_clip, dur) = cargar(carpeta)
    M = medidas(X, d)
    tres = np.isin(d['conducta'], CONDUCTAS) & (d['segundo'] % cada == 0)
    if quitar_transiciones:
        tres &= ~transiciones(d)
    y = np.array([CONDUCTAS.index(c) for c in d['conducta'][tres]])
    print('\nentrenando la red final con %d segundos (%s)' % (tres.sum(), fst_cnn.dispositivo()))
    _nueva('final')
    if decay is None and suavizado_etiq is None and not congelar:
        m = fst_cnn.entrenar(X[tres], y, epocas=epocas, M=M[tres], log=_log)
    else:
        m = entrenar_ajustable(X[tres], y, epocas=epocas, M=M[tres], log=_log, decay=decay,
                               suavizado_etiq=suavizado_etiq, congelar=congelar)
    torch.save(dict(pesos=m.state_dict(), conductas=CONDUCTAS, frames=X.shape[1],
                    medidas=fst_cnn.RASGOS + fst_cnn.REGLAS, alto=X.shape[2], ancho=X.shape[3],
                    videos=sorted(set(d['video'])), n_clips=int(tres.sum()),
                    version=fst_cnn.VERSION, entrada='fusion', unidad='segundo',
                    version_seg=VERSION_SEG, fps_clip=fps_clip, ventana=dur), destino)
    print('red -> %s' % destino)
    return HISTORIA


# ---------------------------------------------------------------------------
# PROBAR MEJORAS SIN VOLVER A ENTRENAR
# Trabajan sobre P (las probabilidades de la validacion por video). Ninguna
# toca la red: sirven para ver cuanto se gana con mirar los segundos vecinos.
# ---------------------------------------------------------------------------
def guardar_corrida(P, d, nombre, carpeta):
    """Guarda las predicciones de una validacion para no perderlas al cerrar Colab."""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre + '.npz')
    np.savez_compressed(ruta, P=P, video=d['video'], especimen=d['especimen'],
                        segundo=d['segundo'], conducta=d['conducta'])
    print('corrida guardada:', ruta)


def cargar_corrida(ruta):
    z = np.load(ruta, allow_pickle=True)
    return z['P'], {k: z[k] for k in ('video', 'especimen', 'segundo', 'conducta')}


def _hojas(d):
    """Indices de cada rata, en orden de segundo."""
    h = {}
    for i, (v, e) in enumerate(zip(d['video'], d['especimen'])):
        h.setdefault((str(v), int(e)), []).append(i)
    return {k: np.array(sorted(ix, key=lambda i: d['segundo'][i])) for k, ix in h.items()}


def suavizar(P, d, k=2):
    """Promedia las probabilidades de cada segundo con las de los k segundos de
    cada lado (de la misma rata). Quita los cambios sueltos de una conducta a
    otra y de vuelta."""
    out = np.zeros_like(P)
    for ix in _hojas(d).values():
        s = np.asarray(d['segundo'])[ix].astype(float)
        cerca = (np.abs(s[:, None] - s[None, :]) <= k).astype(float)
        out[ix] = (cerca @ P[ix]) / cerca.sum(1, keepdims=True)
    return out


def modelo_transiciones(P, d, permanencia=0.9):
    """Probabilidad de cada conducta en cada segundo usando que una conducta
    suele seguir: pasada hacia adelante y hacia atras de un modelo de
    transiciones (cada segundo sigue igual con probabilidad 'permanencia').
    Devuelve probabilidades, no solo la conducta."""
    n = P.shape[1]
    A = np.full((n, n), (1 - permanencia) / (n - 1))
    np.fill_diagonal(A, permanencia)
    prior = P.mean(0)
    E = P / np.maximum(prior, 1e-6)                 # P como verosimilitud
    out = np.zeros_like(P)
    for ix in _hojas(d).values():
        s = np.asarray(d['segundo'])[ix]
        m = len(ix)
        paso = [np.linalg.matrix_power(A, int(max(1, min(s[t] - s[t - 1], 20)))) for t in range(1, m)]
        f = np.zeros((m, n))
        f[0] = E[ix[0]] * prior
        f[0] /= f[0].sum()
        for t in range(1, m):
            f[t] = E[ix[t]] * (f[t - 1] @ paso[t - 1])
            f[t] /= f[t].sum()
        b = np.ones((m, n))
        for t in range(m - 2, -1, -1):
            b[t] = paso[t] @ (E[ix[t + 1]] * b[t + 1])
            b[t] /= b[t].sum()
        g = f * b
        out[ix] = g / g.sum(1, keepdims=True)
    return out


def apilar_contexto(P, d, k=5):
    """Una regresion logistica pequena que decide cada segundo mirando las
    probabilidades de los k segundos de cada lado. Se entrena y se prueba POR
    VIDEO (con las P que la validacion ya dio para videos que la red no vio),
    asi que la nota es honesta salvo un sesgo pequeno."""
    from sklearn.linear_model import LogisticRegression
    n = P.shape[1]
    F = np.zeros((len(P), n * (2 * k + 1)))
    for ix in _hojas(d).values():
        pos = {int(d['segundo'][i]): i for i in ix}
        for i in ix:
            s = int(d['segundo'][i])
            for j, o in enumerate(range(-k, k + 1)):
                v = pos.get(s + o)
                F[i, j * n:(j + 1) * n] = P[v] if v is not None else P[i]
    tres = np.isin(d['conducta'], CONDUCTAS)
    y = np.array([CONDUCTAS.index(c) if c in CONDUCTAS else -1 for c in d['conducta']])
    video = np.asarray(d['video'])
    out = np.full_like(P, np.nan)
    gk = GroupKFold(n_splits=min(5, len(set(video))))
    for tr, te in gk.split(F, y, video):
        tr = tr[tres[tr]]
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(F[tr], y[tr])
        out[te] = lr.predict_proba(F[te])
    return out


def _numeros(P, d, sin_frontera=False):
    from sklearn.metrics import precision_recall_fscore_support as prfs
    conducta = np.asarray(d['conducta'])
    tres = np.isin(conducta, CONDUCTAS)
    if sin_frontera:
        tres = tres & ~transiciones(d)
    y = np.array([CONDUCTAS.index(c) for c in conducta[tres]])
    p = P.argmax(1)[tres]
    f1 = prfs(y, p, labels=range(len(CONDUCTAS)), zero_division=0)[2]
    ri = conducta == 'inmovilidad'
    pi = P[:, CONDUCTAS.index('inmovilidad')] >= 0.5
    f2 = prfs(ri, pi, labels=[True, False], zero_division=0)[2]
    return dict(exactitud=(y == p).mean(), kappa=cohen_kappa_score(y, p),
                F1_prom=f1.mean(), F1_escal=f1[0], F1_inmov=f1[1], F1_nado=f1[2],
                F1_activa=f2[1], segundos=int(tres.sum()))


def comparar_posproceso(P, d):
    """Prueba de una vez el suavizado, el modelo de transiciones y el apilado de
    contexto sobre P. Devuelve la tabla y un diccionario con las P de cada
    variante (para usar la que mejor salga en figuras_resultados)."""
    import pandas as pd
    variantes = {'base': P}
    for k in (1, 2, 3, 5):
        variantes['suavizado +-%d s' % k] = suavizar(P, d, k)
    for q in (0.8, 0.9, 0.95, 0.98):
        variantes['transiciones %.2f' % q] = modelo_transiciones(P, d, q)
    for k in (3, 5, 8):
        variantes['contexto +-%d s' % k] = apilar_contexto(P, d, k)
    filas = {nombre: _numeros(Q, d) for nombre, Q in variantes.items()}
    tabla = pd.DataFrame(filas).T.drop(columns='segundos')
    print(tabla.round(3).to_string())
    mejor = tabla['exactitud'].idxmax()
    print('\nmejor exactitud: %s (%.3f); la base es %.3f' % (mejor, tabla.loc[mejor, 'exactitud'],
                                                            tabla.loc['base', 'exactitud']))
    return tabla.round(3), variantes


def exactitud_sin_transiciones(P, d):
    """Cuanto del error esta justo en la frontera entre dos conductas, donde ni
    tu etiqueta es segura."""
    todo, sin = _numeros(P, d), _numeros(P, d, sin_frontera=True)
    print('exactitud con todos los segundos : %.3f  (%d segundos)' % (todo['exactitud'], todo['segundos']))
    print('exactitud sin los de frontera    : %.3f  (%d segundos)' % (sin['exactitud'], sin['segundos']))
    print('segundos de frontera             : %.0f %%' % (100 * (1 - sin['segundos'] / todo['segundos'])))


def resumen_corridas(carpeta):
    """Tabla con todas las corridas guardadas: exactitud, kappa y F1."""
    import pandas as pd
    filas = {}
    for f in sorted(glob.glob(os.path.join(carpeta, '*.npz'))):
        P, d = cargar_corrida(f)
        filas[os.path.splitext(os.path.basename(f))[0]] = _numeros(P, d)
    if not filas:
        print('no hay corridas en', carpeta)
        return None
    tabla = pd.DataFrame(filas).T.drop(columns='segundos').round(3)
    print(tabla.sort_values('exactitud', ascending=False).to_string())
    return tabla


# ---------------------------------------------------------------------------
# FIGURAS PARA LA DOCUMENTACION
# Cada funcion dibuja en el cuaderno y, si se da 'carpeta_fig', guarda un PNG
# a 200 dpi ahi (en Colab: CARPETA + '/figuras'). Los rotulos dicen
# "espécimen" y no llevan numeros de version.
# ---------------------------------------------------------------------------
COLORES = {'escalamiento': '#D55E00', 'inmovilidad': '#0072B2',
           'nado': '#009E73', 'activa': '#999999'}


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 110})
    return plt


def _guardar(fig, carpeta_fig, nombre):
    if carpeta_fig:
        os.makedirs(carpeta_fig, exist_ok=True)
        ruta = os.path.join(carpeta_fig, nombre + '.png')
        fig.savefig(ruta, dpi=200, bbox_inches='tight', facecolor='white')
        print('figura guardada:', ruta)


def _etiqueta(d, i):
    return '%s, espécimen %d, s %d' % (d['video'][i], d['especimen'][i], d['segundo'][i])


def figura_datos(d, carpeta_fig=None):
    """Segundos etiquetados por conducta: total y por espécimen (el desbalance).
    Devuelve la tabla."""
    import pandas as pd
    plt = _plt()
    orden = CONDUCTAS + ['activa']
    tab = pd.DataFrame({'video': d['video'], 'especimen': d['especimen'],
                        'conducta': d['conducta']})
    tab['hoja'] = tab['video'] + ' #' + tab['especimen'].astype(str)
    cuenta = tab.groupby(['hoja', 'conducta']).size().unstack(fill_value=0)
    cuenta = cuenta.reindex(columns=[c for c in orden if c in cuenta.columns], fill_value=0)
    total = cuenta.sum()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1.6]})
    b = ax[0].bar(total.index, total.values, color=[COLORES[c] for c in total.index])
    for r, v in zip(b, total.values):
        ax[0].text(r.get_x() + r.get_width() / 2, v, '%d\n(%.1f %%)' % (v, 100 * v / total.sum()),
                   ha='center', va='bottom', fontsize=9)
    ax[0].set_ylabel('segundos etiquetados')
    ax[0].set_ylim(0, total.max() * 1.22)
    ax[0].set_title('Total por conducta')
    izq = np.zeros(len(cuenta))
    for c in cuenta.columns:
        ax[1].barh(cuenta.index, cuenta[c], left=izq, color=COLORES[c], label=c)
        izq += cuenta[c].values
    ax[1].set_xlabel('segundos etiquetados')
    ax[1].set_title('Por espécimen')
    ax[1].legend(frameon=False, fontsize=8, loc='lower right')
    ax[1].invert_yaxis()
    fig.suptitle('Segundos etiquetados por un humano: %d, en %d videos y %d espécimenes'
                 % (total.sum(), tab['video'].nunique(), len(cuenta)), y=1.02)
    fig.tight_layout()
    _guardar(fig, carpeta_fig, 'datos_por_conducta')
    plt.show()
    t = cuenta.copy()
    t.loc['TOTAL'] = total
    t['total'] = t.sum(axis=1)
    return t


def figura_muestras(X, d, ejemplos=2, fotogramas=4, semilla=0, carpeta_fig=None):
    """Fotogramas reales de la ventana de 4 s de varios segundos de cada
    conducta: lo que ve la red."""
    plt = _plt()
    rng = np.random.RandomState(semilla)
    filas = []
    for c in CONDUCTAS:
        idx = np.where(d['conducta'] == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        elegidos, vistos = [], set()
        for i in idx:                                # de videos distintos si se puede
            if d['video'][i] not in vistos:
                elegidos.append(i)
                vistos.add(d['video'][i])
            if len(elegidos) == ejemplos:
                break
        for i in idx:
            if len(elegidos) == ejemplos:
                break
            if i not in elegidos:
                elegidos.append(i)
        filas += [(c, i) for i in elegidos]
    fig, ax = plt.subplots(len(filas), fotogramas, figsize=(2.3 * fotogramas, 1.9 * len(filas)),
                           squeeze=False)
    for r, (c, i) in enumerate(filas):
        clip = np.asarray(X[int(i)])
        for k, j in enumerate(np.linspace(0, len(clip) - 1, fotogramas).round().astype(int)):
            a = ax[r, k]
            a.imshow(clip[j], cmap='gray', aspect='auto')
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color(COLORES[c])
                s.set_linewidth(2.5)
            if r == 0:
                a.set_title('t = %.1f s' % (j / (len(clip) - 1) * 4), fontsize=9)
        ax[r, 0].set_ylabel(c, color=COLORES[c], fontsize=10, fontweight='bold')
        ax[r, -1].text(1.03, 0.5, _etiqueta(d, i), transform=ax[r, -1].transAxes,
                       fontsize=7, va='center')
    fig.suptitle('Muestras de entrada de la red: ventana de 4 s alrededor del segundo etiquetado',
                 y=1.0)
    fig.tight_layout()
    _guardar(fig, carpeta_fig, 'muestras_por_conducta')
    plt.show()


def _dibujar_curvas(historia, titulo='Pérdida de entrenamiento'):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    llenas = [v for v in historia.values() if v]
    for nombre, v in historia.items():
        if v:
            ax.plot(range(1, len(v) + 1), v, marker='o', ms=3, lw=1.2, alpha=0.8, label=nombre)
    if len(llenas) > 1:
        n = min(len(v) for v in llenas)
        ax.plot(range(1, n + 1), np.mean([v[:n] for v in llenas], axis=0),
                color='black', lw=2.4, label='media')
    ax.set_xlabel('época')
    ax.set_ylabel('pérdida')
    ax.set_title(titulo)
    if llenas:
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def activar_curva_viva():
    """La curva de pérdida se redibuja sola después de cada época."""
    global AL_EPOCA
    from IPython.display import display, update_display
    import matplotlib.pyplot as plt
    estado = {'primera': True}

    def dibujar(historia):
        fig = _dibujar_curvas(historia, 'Pérdida de entrenamiento (en vivo)')
        if estado['primera']:
            display(fig, display_id='curva_viva')
            estado['primera'] = False
        else:
            update_display(fig, display_id='curva_viva')
        plt.close(fig)
    AL_EPOCA = dibujar


def figura_entrenamiento(historia=None, carpeta_fig=None):
    plt = _plt()
    fig = _dibujar_curvas(historia or HISTORIA)
    _guardar(fig, carpeta_fig, 'curva_perdida')
    plt.show()


def metricas(P, d):
    """Precisión, recall y F1 por conducta. Las filas 'activa' no entran:
    no son una de las tres conductas."""
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    tres = np.isin(d['conducta'], CONDUCTAS)
    y = np.array([CONDUCTAS.index(c) for c in d['conducta'][tres]])
    p = P.argmax(1)[tres]
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=range(len(CONDUCTAS)),
                                                    zero_division=0)
    t = pd.DataFrame({'precisión': pr, 'recall': rc, 'F1': f1, 'segundos': n}, index=CONDUCTAS)
    t.loc['exactitud global'] = [np.nan, np.nan, (y == p).mean(), len(y)]
    return t.round(3)


def figuras_resultados(P, d, X=None, carpeta_fig=None, lineas=4, errores=6):
    """Todo lo que se pega en la documentación después de validar. Devuelve
    la tabla de precisión, recall y F1 por conducta."""
    plt = _plt()
    from sklearn.metrics import confusion_matrix
    tres = np.isin(d['conducta'], CONDUCTAS)
    y = np.array([CONDUCTAS.index(c) for c in d['conducta'][tres]])
    pred = P.argmax(1)
    pt = pred[tres]
    t = metricas(P, d)

    # 1. matriz de confusión + precisión, recall y F1
    cm = confusion_matrix(y, pt, labels=range(len(CONDUCTAS)))
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1, 1.15]})
    ax[0].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, '%d\n%.0f %%' % (cm[i, j], 100 * cmn[i, j]), ha='center',
                       va='center', color='white' if cmn[i, j] > 0.55 else 'black')
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(CONDUCTAS, rotation=20)
    ax[0].set_yticks(range(3))
    ax[0].set_yticklabels(CONDUCTAS)
    ax[0].set_xlabel('conducta que dice la red')
    ax[0].set_ylabel('conducta que marcó el humano')
    ax[0].set_title('Matriz de confusión (segundos)')
    for s in ax[0].spines.values():
        s.set_visible(False)
    ancho = 0.26
    for k, (col, color) in enumerate([('precisión', '#4C72B0'), ('recall', '#DD8452'),
                                      ('F1', '#55A868')]):
        ax[1].bar(np.arange(3) + (k - 1) * ancho, t.loc[CONDUCTAS, col], ancho, label=col,
                  color=color)
        for x_, v in enumerate(t.loc[CONDUCTAS, col]):
            ax[1].text(x_ + (k - 1) * ancho, v + 0.01, '%.2f' % v, ha='center', fontsize=8)
    ax[1].axhline(0.85, color='black', ls='--', lw=1)
    ax[1].text(2.45, 0.86, 'meta 0.85', ha='right', fontsize=8)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(CONDUCTAS)
    ax[1].set_ylim(0, 1.08)
    ax[1].legend(frameon=False, ncol=3, loc='upper left', fontsize=8)
    ax[1].set_title('Precisión, recall y F1 por conducta (validación por video)')
    fig.tight_layout()
    _guardar(fig, carpeta_fig, 'confusion_y_f1')
    plt.show()

    # 2. segundos de cada conducta por espécimen: humano contra red
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for j, c in enumerate(CONDUCTAS):
        h, r = [], []
        for v, e in sorted(set(zip(d['video'], d['especimen']))):
            s = (d['video'] == v) & (d['especimen'] == e) & tres
            h.append((d['conducta'][s] == c).sum())
            r.append((pred[s] == j).sum())
        lim = max(max(h), max(r)) * 1.08 + 1
        ax[j].plot([0, lim], [0, lim], color='gray', lw=1, ls='--')
        ax[j].scatter(h, r, color=COLORES[c], s=45, zorder=3)
        ax[j].set_xlim(0, lim)
        ax[j].set_ylim(0, lim)
        ax[j].set_xlabel('segundos del humano')
        ax[j].set_ylabel('segundos de la red')
        ax[j].set_title(c)
        ax[j].set_aspect('equal')
    fig.suptitle('Segundos de cada conducta por espécimen (un punto = un espécimen)', y=1.03)
    fig.tight_layout()
    _guardar(fig, carpeta_fig, 'segundos_por_especimen')
    plt.show()

    # 3. qué pasa al descartar los segundos donde la red duda
    conf = P[tres].max(1)
    us = np.linspace(0, 0.95, 20)
    cob = [(conf >= u).mean() for u in us]
    exa = [(y[conf >= u] == pt[conf >= u]).mean() if (conf >= u).any() else np.nan for u in us]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(us, exa, color='#0072B2', label='exactitud de lo que se queda')
    ax.plot(us, cob, color='#999999', label='fracción de segundos que se queda')
    ax.set_xlabel('umbral de seguridad de la red')
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Qué pasa al descartar los segundos dudosos')
    fig.tight_layout()
    _guardar(fig, carpeta_fig, 'umbral_seguridad')
    plt.show()

    # 4. líneas de tiempo: humano arriba, red abajo
    hojas = [h for h in sorted(set(zip(d['video'], d['especimen'])))
             if ((d['video'] == h[0]) & (d['especimen'] == h[1]) & tres).sum() >= 60][:lineas]
    if hojas:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        cmap = ListedColormap(['#EEEEEE'] + [COLORES[c] for c in CONDUCTAS])
        fig, ax = plt.subplots(len(hojas), 1, figsize=(11, 1.5 * len(hojas) + 0.7), squeeze=False)
        for a, (v, e) in zip(ax[:, 0], hojas):
            s = (d['video'] == v) & (d['especimen'] == e)
            seg = d['segundo'][s]
            n = int(seg.max())
            tira = np.zeros((2, n))
            for sg, c, p in zip(seg, d['conducta'][s], pred[s]):
                if c in CONDUCTAS:
                    tira[0, sg - 1] = CONDUCTAS.index(c) + 1
                    tira[1, sg - 1] = p + 1
            a.imshow(tira, aspect='auto', cmap=cmap, vmin=0, vmax=3, interpolation='nearest',
                     extent=(0, n, 2, 0))
            a.set_yticks([0.5, 1.5])
            a.set_yticklabels(['humano', 'red'])
            a.set_title('%s, espécimen %d' % (v, e), fontsize=9, loc='left')
            for sp in a.spines.values():
                sp.set_visible(False)
        ax[-1, 0].set_xlabel('segundo del video')
        fig.legend(handles=[Patch(color=COLORES[c], label=c) for c in CONDUCTAS]
                   + [Patch(color='#EEEEEE', label='sin etiqueta humana')],
                   loc='lower center', ncol=4, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        _guardar(fig, carpeta_fig, 'lineas_de_tiempo')
        plt.show()

    # 5. segundos donde la red no coincide, con su fotograma
    if X is not None and errores:
        yy = np.array([CONDUCTAS.index(c) if c in CONDUCTAS else -1 for c in d['conducta']])
        mal = np.where(tres & (pred != yy))[0]
        if len(mal):
            mal = np.random.RandomState(0).choice(mal, min(errores, len(mal)), replace=False)
            col = 3
            fil = int(np.ceil(len(mal) / col))
            fig, ax = plt.subplots(fil, col, figsize=(3.4 * col, 2.9 * fil), squeeze=False)
            for a in ax.ravel():
                a.axis('off')
            for a, i in zip(ax.ravel(), mal):
                clip = np.asarray(X[int(i)])
                a.imshow(clip[len(clip) // 2], cmap='gray', aspect='auto')
                a.set_title('humano: %s\nred: %s (%.2f)\n%s'
                            % (d['conducta'][i], CONDUCTAS[pred[i]], P[i].max(), _etiqueta(d, i)),
                            fontsize=8)
            fig.suptitle('Ejemplos de segundos donde la red no coincide con el humano', y=1.0)
            fig.tight_layout()
            _guardar(fig, carpeta_fig, 'errores_ejemplo')
            plt.show()

    if carpeta_fig:
        os.makedirs(carpeta_fig, exist_ok=True)
        t.to_csv(os.path.join(carpeta_fig, 'metricas_por_conducta.csv'), encoding='utf-8-sig')
    return t


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
