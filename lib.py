"""Nucleo compartido del etiquetador automatico de FST.

Diferencias clave contra el pipeline de un solo video:

  1. La geometria NO esta escrita a mano. Se deduce de cada video:
     paredes de los tubos desde el mapa de ocupacion, linea de agua y fondo
     desde los bordes horizontales del aparato vacio.

  2. Todas las medidas de distancia se dividen por el LARGO DEL CUERPO del
     animal en ese video. Asi "se movio 3 veces su largo" significa lo mismo
     con la camara cerca o lejos. Sin esto los rasgos no son comparables
     entre videos y el clasificador aprende el encuadre, no la conducta.

  3. El tiempo se normaliza por segundo, no por fotograma, porque los videos
     no vienen todos a la misma tasa.
"""
import cv2
import numpy as np

DIFF_THR   = 45    # diferencia minima contra el fondo
BRIGHT_THR = 80    # brillo absoluto minimo (rata clara sobre agua oscura)
ERODE_TAIL = 21    # borra la cola antes de medir la postura
BLOCK_S    = 5.0   # duracion del bloque de puntuacion

# Separacion temporal fija, en segundos, para las medidas que comparan dos
# instantes (energia de movimiento, IoU, Dice). Se fija en TIEMPO y no en
# fotogramas para que --paso no altere los valores: con paso 12 hay 0.40 s
# entre mediciones y con paso 2 solo 0.067 s, asi que comparar contra "el
# anterior" daria numeros tres veces menores y el modelo lo leeria como
# quietud. El valor 0.40 es el que se uso al medir el rendimiento.
DT_REF     = 0.40
DWELL_S    = 0.20  # permanencia minima en un cuadrante para contarlo

K5  = np.ones((5, 5), np.uint8)
K9  = np.ones((9, 9), np.uint8)
K15 = np.ones((15, 15), np.uint8)
ER  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_TAIL, ERODE_TAIL))

RASGOS = ['nq', 'rng', 'path', 'spanx', 'spany', 'me', 'me_max',
          'vert', 'elong', 'hrise', 'hmean', 'above', 'area', 'iou', 'dice']


# --------------------------- utilidades ---------------------------

def mayor_componente(mask):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, None, None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8), st[k], cen[k]


def medfilt1(x, k=5):
    """Mediana movil. El centroide crudo salta varios px por fotograma solo
    por ruido de segmentacion; sin suavizar, 'path' mide ese ruido."""
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    p = k // 2
    xp = np.pad(x, p, mode='edge')
    return np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)


def info_video(ruta):
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        raise IOError('no se pudo abrir: %s' % ruta)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not (1 < fps < 200):
        raise ValueError('fps ilegible (%r); pasa --fps a mano' % fps)
    return fps, n, w, h


# ----------------------- 1. registro de camara -----------------------

def registrar(ruta, paso=2, ref_frac=0.5, log=print):
    """Alinea cada fotograma contra uno de referencia (similitud: traslacion,
    rotacion y escala). Devuelve {indice: matriz 2x3}, el tamano y los fps.

    Sin esto, una camara de mano hace que cualquier medida en coordenadas
    fijas siga a la camara en vez del animal."""
    fps, n, W, H = info_video(ruta)
    # Un archivo truncado o mal copiado declara mas fotogramas de los que
    # tiene. Se prueban varias posiciones antes de rendirse, empezando por la
    # pedida y retrocediendo hacia el principio.
    cap = cv2.VideoCapture(ruta)
    fr = None
    for frac in [ref_frac, 0.35, 0.2, 0.1, 0.02]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ok, f = cap.read()
        if ok:
            fr, ref_frac = f, frac
            break
    if fr is None:
        raise IOError('el video no se deja leer en ninguna posicion; '
                      'lo mas probable es que este incompleto o dañado')
    if ref_frac != 0.5:
        log('    AVISO: referencia tomada al %.0f%% del video; el archivo '
            'puede estar incompleto' % (100 * ref_frac))
    SC = 0.5

    def prep(f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(2000, fastThreshold=7)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    kr, dr = orb.detectAndCompute(prep(fr), None)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    lut, fallos, cur = {}, 0, -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cur += 1
        if cur % paso:
            continue
        kq, dq = orb.detectAndCompute(prep(f), None)
        M = None
        if dq is not None and len(kq) > 12:
            pares = bf.knnMatch(dq, dr, k=2)
            buenos = [a for a, b in pares if a.distance < 0.75 * b.distance]
            if len(buenos) >= 12:
                src = np.float32([kq[m.queryIdx].pt for m in buenos]) / SC
                dst = np.float32([kr[m.trainIdx].pt for m in buenos]) / SC
                M, _ = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                    maxIters=3000)
        if M is None:
            fallos += 1
            M = np.float32([[1, 0, 0], [0, 1, 0]])
        lut[cur] = M
        if cur % 3000 == 0:
            log('    registro t=%.0fs' % (cur / fps))
    cap.release()
    log('    %d fotogramas alineados, %d sin puntos de apoyo' % (len(lut), fallos))
    return lut, (W, H), fps


def _alineados(ruta, lut, WH, cada):
    W, H = WH
    cap = cv2.VideoCapture(ruta)
    cur = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cur += 1
        if cur not in lut or cur % cada:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        yield cur, cv2.warpAffine(g, lut[cur], (W, H),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
    cap.release()


# ----------------- 2. fondo y mapa de ocupacion -----------------

def fondo_y_ocupacion(ruta, lut, WH, fps, log=print):
    """El fondo es el percentil 10 temporal: como la rata es clara, el
    percentil bajo la borra y deja el aparato vacio. La mediana NO sirve,
    un animal quieto se queda incrustado dentro del fondo."""
    cada = max(1, int(round(fps)))
    pila = [w for _, w in _alineados(ruta, lut, WH, cada)]
    if len(pila) < 10:
        raise RuntimeError('video demasiado corto para modelar el fondo')
    fondo = np.percentile(np.stack(pila), 10, axis=0).astype(np.uint8)
    log('    fondo con %d fotogramas' % len(pila))

    bg = fondo.astype(np.int16)
    acc, n = np.zeros(fondo.shape, np.float32), 0
    for _, w in _alineados(ruta, lut, WH, max(1, cada // 3)):
        wi = w.astype(np.int16)
        acc += ((wi - bg) > DIFF_THR) & (wi > BRIGHT_THR)
        n += 1
    log('    ocupacion con %d fotogramas' % n)
    return fondo, acc / max(n, 1)


# ----------------- 3. geometria automatica -----------------

def detectar_geometria(fondo, occ, n_tubos=None, log=print):
    """Devuelve dict con tubos, y_agua, y_fondo, roi_arriba, roi_abajo.

    El orden importa y no es el obvio:

    1. Las FILAS se sacan primero, del fotograma completo del aparato vacio.
       Los dos bordes horizontales mas fuertes son los rebordes del recipiente.
    2. Las COLUMNAS se sacan despues, midiendo la ocupacion SOLO dentro de la
       franja de agua. Fuera de ella mandan el estante, la puerta y la gente
       que pasa: en un video de prueba el borde del encuadre tenia el triple de
       movimiento que las propias ratas y se tragaba todos los tubos en uno.
    3. Se tiran las rachas pegadas al borde del encuadre. Un aparato nunca esta
       al filo de la pantalla; lo que aparece ahi es el laboratorio.
    4. Las filas se recalculan con las bandas estrechas ya conocidas, que dan
       el suelo mas fino que el fotograma entero.

    Un cilindro vacio no produce racha, asi que se numeran los tubos CON
    animal, de izquierda a derecha."""
    H, W = occ.shape

    def bordes_horizontales(cols=None):
        b = fondo if cols is None else np.concatenate(
            [fondo[:, a:b_] for a, b_ in cols], axis=1)
        b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 3)
        return np.convolve(np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=5)).mean(1),
                           np.ones(5) / 5, mode='same')

    def rebordes(gy):
        def picos(y0, y1, sep=25, n=1):
            y0, y1 = max(0, y0), min(H, y1)
            orden = np.argsort(gy[y0:y1])[::-1] + y0
            sel = []
            for p in orden:
                if all(abs(p - q) > sep for q in sel):
                    sel.append(int(p))
                if len(sel) == n:
                    break
            return sel
        f = sorted(picos(int(0.2 * H), int(0.95 * H), sep=int(0.15 * H), n=2))
        if len(f) < 2:
            raise RuntimeError('no se encontraron los bordes del aparato')
        ar, ab = f
        alto = ab - ar
        ya = picos(ar + int(0.10 * alto), ar + int(0.55 * alto), n=1)[0]
        yf = picos(ar + int(0.70 * alto), ab - int(0.10 * alto), n=1)[0]
        if yf - ya < 0.25 * alto:
            ya, yf = ar + int(0.22 * alto), ar + int(0.84 * alto)
            log('    AVISO: bordes internos poco claros, se usaron proporciones tipicas')
        return ya, yf

    # --- 1. filas aproximadas, del fotograma entero ---
    ya0, yf0 = rebordes(bordes_horizontales())

    # --- 2. columnas, midiendo solo dentro de la franja de agua ---
    cp = np.convolve(occ[ya0:yf0, :].sum(0), np.ones(15) / 15, mode='same')
    on = cp > 0.12 * cp.max()
    crudas, i = [], 0
    while i < W:
        if on[i]:
            j = i
            while j + 1 < W and on[j + 1]:
                j += 1
            if j - i > 0.03 * W:
                crudas.append((i, j))
            i = j + 1
        else:
            i += 1

    # --- 3. fuera lo que toca el borde del encuadre ---
    margen = max(3, int(0.01 * W))
    rachas = [(a, b) for a, b in crudas if a > margen and b < W - 1 - margen]
    if len(rachas) < len(crudas):
        log('    descartadas %d zonas pegadas al borde del encuadre '
            '(estanteria, puerta, gente)' % (len(crudas) - len(rachas)))
    if n_tubos and len(rachas) > n_tubos:
        # se queda con las de mas actividad, no con las mas anchas: una racha
        # ancha puede ser una sombra que se arrastra.
        rachas = sorted(rachas, key=lambda r: cp[r[0]:r[1]].sum(),
                        reverse=True)[:n_tubos]
        rachas.sort()
    if not rachas:
        raise RuntimeError('no se distinguio ningun tubo; revisa el video')
    if n_tubos and len(rachas) != n_tubos:
        log('    AVISO: se pidieron %d tubos y se hallaron %d. Puede que algun '
            'cilindro este vacio o que el animal no se distinga del fondo.'
            % (n_tubos, len(rachas)))

    # Ensanchar cada racha hasta la mitad del hueco con el vecino: ahi esta la
    # pared. Los dos tubos de los extremos no tienen vecino por fuera, asi que
    # se les da el ancho tipico de los de en medio; si no, quedan recortados y
    # se pierde al animal cuando se pega al borde exterior.
    cortes = [(rachas[k][0] + rachas[k - 1][1]) // 2 for k in range(1, len(rachas))]
    tubos = {}
    for k, (a, b) in enumerate(rachas, 1):
        izq = cortes[k - 2] if k > 1 else None
        der = cortes[k - 1] if k < len(rachas) else None
        tubos[k] = [izq, der]
    anchos = [d - i for i, d in tubos.values() if i is not None and d is not None]
    tipico = int(np.median(anchos)) if anchos else int(np.median([b - a for a, b in rachas]) * 1.2)
    for k, (i, d) in tubos.items():
        if i is None:
            i = max(0, d - tipico) if d is not None else rachas[k - 1][0]
        if d is None:
            d = min(W - 1, i + tipico)
        tubos[k] = (int(i), int(d))

    # --- 4. filas otra vez, ya con las bandas estrechas de cada tubo ---
    # Afinan el suelo: el fotograma entero lo dejaba 45 px desviado y estas
    # bandas lo dejan en 16. Se usan las rachas SIN ensanchar, porque las
    # paredes meten bordes propios que tapan el del suelo.
    y_agua, y_fondo = rebordes(bordes_horizontales(rachas))

    col = y_fondo - y_agua
    g = dict(tubos=tubos, y_agua=int(y_agua), y_fondo=int(y_fondo),
             roi_arriba=int(max(0, y_agua - 0.45 * col)),
             roi_abajo=int(min(H - 1, y_fondo + 0.05 * col)))
    log('    tubos: ' + '  '.join('%d:%d-%d' % (k, a, b)
                                  for k, (a, b) in tubos.items()))
    log('    agua y=%d   fondo y=%d   zona medida %d-%d'
        % (g['y_agua'], g['y_fondo'], g['roi_arriba'], g['roi_abajo']))
    return g


def dibujar_geometria(ruta, lut, WH, g, destino):
    """Imagen de control. Mirala 5 segundos por video: si las lineas no caen
    sobre el agua y el suelo, la deteccion fallo y no sirve seguir."""
    W, H = WH
    i = sorted(lut)[len(lut) // 2]
    cap = cv2.VideoCapture(ruta)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return
    im = cv2.warpAffine(fr, lut[i], (W, H), flags=cv2.INTER_LINEAR)
    for s, (x0, x1) in g['tubos'].items():
        cv2.rectangle(im, (x0, g['roi_arriba']), (x1, g['roi_abajo']),
                      (0, 255, 255), 2)
        cv2.line(im, (x0, g['y_agua']), (x1, g['y_agua']), (255, 40, 40), 3)
        cv2.line(im, (x0, g['y_fondo']), (x1, g['y_fondo']), (40, 220, 40), 2)
        cv2.putText(im, 'tubo %d' % s, (x0 + 5, max(20, g['roi_arriba'] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    cv2.putText(im, 'azul = linea de agua   verde = fondo   amarillo = zona medida',
                (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2)
    cv2.imwrite(destino, im, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ----------------- 4. extraccion por fotograma -----------------

def extraer(ruta, lut, WH, fondo, g, fps, log=print):
    """Una fila por fotograma y tubo. Todo en px todavia; la normalizacion
    por largo de cuerpo se hace al agregar en bloques."""
    W, H = WH
    bg = fondo.astype(np.int16)
    ya, ra, rb = g['y_agua'], g['roi_arriba'], g['roi_abajo']
    # Por cada tubo se guarda un historial corto de (instante, recorte,
    # mascara). Las medidas dinamicas no se toman contra el fotograma anterior
    # sino contra el mas cercano a DT_REF segundos atras, de modo que --paso
    # cambia el coste de calculo pero no los valores.
    from collections import deque
    filas, hist = [], {}
    cap = cv2.VideoCapture(ruta)
    cur = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cur += 1
        if cur not in lut:
            continue
        gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        w = cv2.warpAffine(gr, lut[cur], (W, H), flags=cv2.INTER_LINEAR,
                           borderValue=0)
        wi = w.astype(np.int16)
        dif = wi - bg
        t = cur / fps

        for s, (x0, x1) in g['tubos'].items():
            roi = wi[ra:rb, x0:x1]
            m = ((dif[ra:rb, x0:x1] > DIFF_THR) & (roi > BRIGHT_THR)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K5)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K9)
            blob, st, cen = mayor_componente(m)

            r = dict(t=t, s=s, cx=np.nan, cy=np.nan, area=np.nan, above=np.nan,
                     largo=np.nan, vert=np.nan, elong=np.nan, cabeza=np.nan,
                     me=np.nan, iou=np.nan, dice=np.nan)
            if blob is not None:
                r['cx'] = cen[0] + x0
                r['cy'] = cen[1] + ra
                r['area'] = float(st[cv2.CC_STAT_AREA])
                r['above'] = float(blob[:max(0, ya - ra), :].sum())
                nucleo, _, _ = mayor_componente(cv2.erode(blob, ER))
                if nucleo is not None:
                    bm = cv2.dilate(nucleo, ER) & blob
                    M = cv2.moments(bm, binaryImage=True)
                    if M['m00'] > 200:
                        u20, u02 = M['mu20'] / M['m00'], M['mu02'] / M['m00']
                        u11 = M['mu11'] / M['m00']
                        th = 0.5 * np.arctan2(2 * u11, u20 - u02)
                        d = np.sqrt(max((u20 - u02) ** 2 + 4 * u11 ** 2, 0))
                        l1 = (u20 + u02 + d) / 2
                        l2 = max((u20 + u02 - d) / 2, 1e-6)
                        ys, _ = np.nonzero(bm)
                        r['vert']   = float(abs(np.sin(th)))
                        r['elong']  = float(np.sqrt(l1 / l2))
                        r['cabeza'] = float(ys.min() + ra)
                        # largo del cuerpo = 4 sigma del eje mayor
                        r['largo']  = float(4 * np.sqrt(max(l1, 1e-6)))
            h = hist.setdefault(s, deque())
            # el fotograma mas reciente que quede al menos DT_REF atras; si el
            # historial aun no llega tan lejos, el mas antiguo disponible
            ref = None
            for reg in reversed(h):
                if t - reg[0] >= DT_REF:
                    ref = reg
                    break
            if ref is None and h:
                ref = h[0]
            if ref is not None:
                _, proi, pm = ref
                if blob is not None and pm is not None:
                    sel = cv2.dilate(((blob | pm) > 0).astype(np.uint8), K15) > 0
                    if sel.sum() > 50:
                        # energia de movimiento SOLO dentro del animal: el
                        # oleaje del agua satura la medida si se toma la ROI
                        r['me'] = float(np.abs(roi - proi)[sel].mean())
                    inter = float((blob & pm).sum())
                    union = float((blob | pm).sum())
                    a1, a2 = float(blob.sum()), float(pm.sum())
                    r['iou']  = inter / union if union else np.nan
                    r['dice'] = 2 * inter / (a1 + a2) if (a1 + a2) else np.nan
            h.append((t, roi.copy(), blob))
            while len(h) > 2 and t - h[0][0] > 2 * DT_REF:
                h.popleft()
            filas.append(r)
        if cur % 3000 == 0:
            log('    extraccion t=%.0fs' % t)
    cap.release()
    return filas


def cajas_crudas(bl, lut, g, fps):
    """Anade bx0, by0, bx1, by1: el recuadro de cada tubo en coordenadas del
    video TAL COMO SE VE, en el instante medio de cada bloque.

    Toda la geometria vive en el marco de referencia estabilizado. Eso sirve
    para medir, pero no para dibujar sobre el video original: si la camara se
    movio, el recuadro cae fuera de la rata. En IMG_0840 la camara se desplaza
    unos 200 px entre el segundo 55 y el 184. Se deshace el registro con la
    inversa de la matriz del fotograma mas cercano a la mitad del bloque."""
    idx = np.array(sorted(lut))
    tiempos = idx / fps
    col = g['y_fondo'] - g['y_agua']
    arriba = max(0.0, g['y_agua'] - 0.45 * col)
    abajo = g['y_fondo'] + 0.05 * col
    cajas = []
    for b, s in zip(bl['bloque'], bl['especimen']):
        medio = (int(b) - 0.5) * BLOCK_S
        i = int(idx[np.argmin(np.abs(tiempos - medio))])
        inv = cv2.invertAffineTransform(np.asarray(lut[i], np.float64))
        x0, x1 = g['tubos'][int(s)]
        esq = np.array([[x0, arriba], [x1, arriba], [x0, abajo], [x1, abajo]], np.float64)
        q = esq @ inv[:, :2].T + inv[:, 2]
        cajas.append((int(q[:, 0].min()), int(q[:, 1].min()),
                      int(q[:, 0].max()), int(q[:, 1].max())))
    bl = bl.copy()
    for j, c in enumerate(('bx0', 'by0', 'bx1', 'by1')):
        bl[c] = [k[j] for k in cajas]
    return bl


# ----------------- 5. rasgos por bloque, normalizados -----------------

def bloques(filas, g, fps, escala=None, log=print):
    """Agrega a bloques de 5 s y normaliza:
       distancias  -> en largos de cuerpo
       velocidades -> por segundo
       alturas     -> en alturas de columna de agua

    'escala' es el largo tipico del cuerpo en px. Se calcula como la mediana
    de TODO el video, no de cada bloque, para que el ruido de un bloque no
    altere su propia normalizacion."""
    import pandas as pd
    df = pd.DataFrame(filas)
    if escala is None:
        escala = float(np.nanmedian(df['largo']))
    if not (escala > 5):
        raise RuntimeError('no se pudo medir el largo del cuerpo')
    ya, yf = g['y_agua'], g['y_fondo']
    col = float(yf - ya)
    y_medio = (ya + yf) / 2

    df['bloque'] = (df['t'] // BLOCK_S).astype(int) + 1
    out = []
    for (s, b), d in df.groupby(['s', 'bloque']):
        d = d.sort_values('t')
        cx, cy = d['cx'].to_numpy(), d['cy'].to_numpy()
        ok = ~np.isnan(cx)
        n_ok = int(ok.sum())
        r = dict(bloque=int(b), especimen=int(s),
                 t0=round((b - 1) * BLOCK_S, 2), t1=round(b * BLOCK_S, 2),
                 n=len(d), n_visto=n_ok, escala_px=round(escala, 1),
                 # Geometria detectada. Se guarda para que revisar.py pueda
                 # recortar y marcar el tubo sin reprocesar el video entero.
                 gx0=int(g['tubos'][s][0]), gx1=int(g['tubos'][s][1]),
                 g_agua=int(ya), g_fondo=int(yf))
        if n_ok >= 5:
            xs, ys = medfilt1(cx[ok], 5), medfilt1(cy[ok], 5)
            dur = max(float(d['t'].iloc[-1] - d['t'].iloc[0]), 1e-3)
            r['rng']   = float(np.hypot(xs.max() - xs.min(),
                                        ys.max() - ys.min())) / escala
            r['spanx'] = float(xs.max() - xs.min()) / escala
            r['spany'] = float(ys.max() - ys.min()) / escala
            r['path']  = float(np.hypot(np.diff(xs), np.diff(ys)).sum()) / escala / dur
            x0, x1 = g['tubos'][s]
            qx = (xs >= (x0 + x1) / 2).astype(int)
            qy = (np.clip(ys, ya, yf) >= y_medio).astype(int)
            q = qy * 2 + qx
            # Permanencia minima en TIEMPO, no en fotogramas: con paso 2 hay
            # 75 mediciones por bloque y con paso 12 solo 12, asi que exigir
            # "3 fotogramas" significaria 0.2 s en un caso y 1.2 s en el otro.
            minimo = max(2, int(round(DWELL_S * len(xs) / max(dur, 1e-3))))
            r['nq'] = int(sum(1 for k in range(4) if (q == k).sum() >= minimo))
            cab = d['cabeza'].to_numpy()[ok]
            if np.isfinite(cab).any():
                r['hrise'] = float(np.nanmax(ya - cab)) / col
                r['hmean'] = float(np.nanmean(ya - cab)) / col
            else:
                r['hrise'] = r['hmean'] = np.nan
            r['above'] = float(np.nanmean(d['above'].to_numpy()[ok])) / escala ** 2
            r['area']  = float(np.nanmean(d['area'].to_numpy()[ok])) / escala ** 2
            r['vert']  = float(np.nanmean(d['vert']))
            r['elong'] = float(np.nanmean(d['elong']))
            me = d['me'].to_numpy()
            hay = np.isfinite(me).any()
            r['me']     = float(np.nanmean(me)) if hay else np.nan
            r['me_max'] = float(np.nanmax(me)) if hay else np.nan
            r['iou']    = float(np.nanmean(d['iou']))
            r['dice']   = float(np.nanmean(d['dice']))
        else:
            for k in RASGOS:
                r[k] = np.nan
        out.append(r)
    res = pd.DataFrame(out).sort_values(['especimen', 'bloque']).reset_index(drop=True)
    # El ultimo bloque casi nunca cabe completo (un video de 301 s deja 1 s
    # suelto). Un bloque a medias no es comparable con los demas: fuera.
    completo = res['n'] >= 0.6 * res['n'].median()
    if (~completo).any():
        log('    descartados %d bloques incompletos (bloque %s)'
            % (int((~completo).sum()),
               ', '.join(str(b) for b in sorted(set(res.loc[~completo, 'bloque'])))))
        res = res[completo].reset_index(drop=True)
    log('    %d bloques  (largo de cuerpo = %.0f px)' % (len(res), escala))
    return res
