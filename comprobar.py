"""Comprueba que esta maquina puede correr el etiquetador. No modifica nada.

    py comprobar.py

Ejecutalo UNA vez despues de clonar el repositorio en una computadora nueva.
En un minuto te dice si falta algo, en vez de que lo descubras a los diez
minutos de estar procesando un video.
"""
import os
import sys

OK, MAL, AVISO = '[ OK ]', '[FALLA]', '[AVISO]'
AQUI = os.path.dirname(os.path.abspath(__file__))
fallas = []
avisos = []


def linea(estado, texto, detalle=''):
    print('%s %-42s %s' % (estado, texto, detalle))


print('=' * 72)
print('  Comprobacion del etiquetador FST')
print('=' * 72)

# --- 1. Python ---
v = sys.version_info
if v[:2] >= (3, 9):
    linea(OK, 'Python', '%d.%d.%d' % v[:3])
else:
    linea(MAL, 'Python', '%d.%d.%d (hace falta 3.9 o mas)' % v[:3])
    fallas.append('Instala Python 3.9 o superior desde python.org')

# --- 2. Librerias ---
NECESARIAS = [
    ('cv2', 'opencv-python', (4, 8)),
    ('numpy', 'numpy', (1, 24)),
    ('pandas', 'pandas', (2, 0)),
    ('sklearn', 'scikit-learn', (1, 9)),
    ('joblib', 'joblib', (1, 3)),
]
faltan = []
sk_ver = None
for mod, paquete, minima in NECESARIAS:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', '?')
        if mod == 'sklearn':
            sk_ver = ver
        try:
            partes = tuple(int(x) for x in ver.split('.')[:2])
            vieja = partes < minima
        except ValueError:
            vieja = False
        if vieja:
            linea(AVISO, paquete, '%s (se probo con %d.%d o mas)'
                  % (ver, minima[0], minima[1]))
            avisos.append('%s esta en %s; puede dar resultados distintos' % (paquete, ver))
        else:
            linea(OK, paquete, ver)
    except ImportError:
        linea(MAL, paquete, 'no instalada')
        faltan.append(paquete)
if faltan:
    fallas.append('pip install ' + ' '.join(faltan))

# --- 3. Archivos del proyecto ---
for f in ('lib.py', 'etiquetar.py', 'entrenar.py'):
    if os.path.exists(os.path.join(AQUI, f)):
        linea(OK, f)
    else:
        linea(MAL, f, 'no esta')
        fallas.append('Falta %s. Clona el repositorio otra vez.' % f)

# --- 4. Carpetas de trabajo ---
for d in ('videos_sin_etiquetar', 'etiquetas'):
    ruta = os.path.join(AQUI, d)
    if os.path.isdir(ruta):
        linea(OK, d + os.sep)
    else:
        os.makedirs(ruta, exist_ok=True)
        linea(OK, d + os.sep, 'no estaba, se acaba de crear')

# --- 5. El modelo carga y trae sus columnas ---
mp = os.path.join(AQUI, 'modelo_fst.joblib')
if not os.path.exists(mp):
    linea(AVISO, 'modelo_fst.joblib', 'no esta; podras sacar rasgos pero no etiquetar')
    avisos.append('Sin modelo. Entrena uno con: py entrenar.py --carpeta etiquetas --etiquetas mano')
elif not faltan:
    try:
        import warnings
        import joblib
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            paq = joblib.load(mp)
            incompat = any('version' in str(x.message).lower() for x in w)
        cols = paq.get('caracteristicas')
        clases = paq.get('clases')
        if not cols or not clases:
            linea(MAL, 'modelo_fst.joblib', 'no guarda sus columnas')
            fallas.append('El modelo esta incompleto. Reentrenalo.')
        elif incompat:
            linea(AVISO, 'modelo_fst.joblib',
                  '%d columnas, pero se guardo con otra version de scikit-learn' % len(cols))
            avisos.append('El modelo se guardo con scikit-learn distinto al tuyo (%s). '
                          'Reentrenalo para que las predicciones sean fiables:\n'
                          '     py entrenar.py --carpeta etiquetas --etiquetas mano' % sk_ver)
        else:
            linea(OK, 'modelo_fst.joblib',
                  '%d columnas, clases: %s' % (len(cols), ', '.join(clases)))
    except Exception as e:
        linea(MAL, 'modelo_fst.joblib', str(e)[:40])
        fallas.append('El modelo no carga. Reentrenalo con entrenar.py')

# --- 6. Lectura de video ---
if 'cv2' not in faltan and not any('opencv' in f for f in faltan):
    import cv2
    vids = [f for f in os.listdir(os.path.join(AQUI, 'videos_sin_etiquetar'))
            if os.path.splitext(f)[1].lower() in ('.mov', '.mp4', '.avi', '.mkv')]
    if not vids:
        linea(AVISO, 'lectura de video', 'no hay videos para probar')
        avisos.append('Deja un video en videos_sin_etiquetar y vuelve a correr esto.')
    else:
        for f in sorted(vids):
            ruta = os.path.join(AQUI, 'videos_sin_etiquetar', f)
            cap = cv2.VideoCapture(ruta)
            if not cap.isOpened():
                linea(MAL, f[:40], 'no se deja abrir')
                fallas.append('%s no se abre. Prueba a reconvertirlo.' % f)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            dec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Un archivo truncado declara mas fotogramas de los que tiene. Se
            # muestrean 20 posiciones repartidas en vez de leerlo entero: leer
            # un video de 1 GB completo tardaria varios minutos.
            puntos = [int(dec * k / 20) for k in range(20)] if dec else [0]
            buenos = 0
            for p in puntos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                bien, _ = cap.read()
                buenos += bool(bien)
            cap.release()
            pct = 100.0 * buenos / len(puntos)
            info = '%dx%d  %.1f fps  %.0f s  %d/%d sondeos (%.0f%%)' % (
                ancho, alto, fps, dec / fps if fps else 0,
                buenos, len(puntos), pct)
            if pct < 90:
                linea(MAL, f[:40], info)
                fallas.append('%s esta incompleto o corrupto: solo responde el %.0f%% '
                              'del video. Copialo de nuevo desde el origen.' % (f, pct))
            else:
                linea(OK, f[:40], info)

# --- resumen ---
print('=' * 72)
if fallas:
    print('  NO se puede usar todavia. Arregla esto:\n')
    for i, f in enumerate(fallas, 1):
        print('  %d. %s' % (i, f))
elif avisos:
    print('  Funciona, con reservas:\n')
    for i, a in enumerate(avisos, 1):
        print('  %d. %s' % (i, a))
else:
    print('  Todo listo. Ya puedes etiquetar:\n')
    print('  py etiquetar.py videos_sin_etiquetar/TUVIDEO.MOV '
          '--modelo=modelo_fst.joblib --tubos=4')
print('=' * 72)
sys.exit(1 if fallas else 0)
