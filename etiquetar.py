"""Etiquetador automatico de FST. Un video entra, un CSV etiquetado sale.

    py etiquetar.py video1.MOV video2.MOV ... --modelo=modelo_fst.joblib --tubos=4

Tambien acepta CSV de rasgos ya sacados, para volver a proponer etiquetas con
un modelo nuevo sin reprocesar el video (segundos en vez de diez minutos):

    py etiquetar.py etiquetas/IMG_0840_rasgos.csv --modelo=modelo_fst.joblib

Opciones:
    --modelo M.joblib   clasificador entrenado; sin el, solo saca rasgos
    --tubos N           cuantos cilindros CON RATA hay (recomendado: ponlo siempre)
    --paso N            analizar 1 de cada N fotogramas. 12 por defecto: es el
                        que mejor resultado dio y con el que se entreno el modelo
    --umbral U          confianza minima para dar por buena la etiqueta (0.80)
    --salida DIR        carpeta de resultados (por defecto ./etiquetas)
    --rotar R           endereza un video grabado con la camara girada:
                        izquierda, derecha o 180. Elige hacia donde hay que
                        girar la imagen para que los cilindros queden de pie.
                        Si el video sale en vertical, el programa lo avisa.

Por cada video deja:
    <nombre>_rasgos.csv        un renglon por bloque de 5 s y por especimen
    <nombre>_control.jpg       MIRALA. Si las lineas no caen sobre el agua y
                               el suelo del tubo, ese video no sirve.
    <nombre>_cuadros.csv       las medidas de cada fotograma (para no tener que
                               volver a leer el video nunca)
    <nombre>_segundos.csv      con --modelo: la conducta de cada rata en cada
                               segundo (lo abre revisor_segundos.html)
    <nombre>_resumen.csv       con --modelo: segundos de cada conducta por rata,
                               por minuto y en total

Conducta activa. En el diseño de Porsolt solo se distingue inmovil de no
inmovil; Detke separo despues lo activo en nado y escalamiento. Cuando el
modelo sabe que la rata NO esta inmovil pero duda entre nadar y trepar, la
propuesta es 'activa': una etiqueta mas gruesa pero fiable. Asi:

    1. si una conducta pasa el umbral               -> esa conducta
    2. si no, pero nado + escalamiento lo pasan     -> activa
    3. si ninguna de las dos cosas                  -> dudoso (usar = 0)

La columna 'usar' vale 1 en los casos 1 y 2. Las probabilidades de cada
conducta quedan en p_inmovilidad, p_nado y p_escalamiento.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import lib

ACTIVAS = ('nado', 'escalamiento')


def proponer(bl, paq, umbral, log=print):
    """Rellena clase, confianza, usar y p_* con la regla de tres pasos."""
    mod, feats = paq['modelo'], paq['caracteristicas']
    falta = [c for c in feats if c not in bl.columns]
    if falta:
        raise SystemExit('el modelo pide columnas que no existen: %s' % falta)
    bl = bl.drop(columns=[c for c in bl.columns
                          if c in ('clase', 'confianza', 'usar') or c.startswith('p_')])
    X = bl[feats].to_numpy(float)
    val = ~np.isnan(X).any(1)
    clases = list(mod.classes_)
    P = np.full((len(bl), len(clases)), np.nan)
    if val.any():
        P[val] = mod.predict_proba(X[val])
    for j, c in enumerate(clases):
        bl['p_' + c] = P[:, j].round(3)

    mejor = np.where(val, np.nanargmax(np.where(val[:, None], P, 0), axis=1), -1)
    conf_mejor = np.where(val, np.nanmax(np.where(val[:, None], P, 0), axis=1), np.nan)
    activa = sum(P[:, clases.index(c)] for c in ACTIVAS if c in clases)

    clase, conf, usar = [], [], []
    for i in range(len(bl)):
        if not val[i]:
            clase.append(''); conf.append(np.nan); usar.append(0)
        elif conf_mejor[i] >= umbral:
            clase.append(clases[mejor[i]]); conf.append(conf_mejor[i]); usar.append(1)
        elif activa[i] >= umbral:
            clase.append('activa'); conf.append(activa[i]); usar.append(1)
        else:
            clase.append(clases[mejor[i]]); conf.append(conf_mejor[i]); usar.append(0)
    bl['clase'] = clase
    bl['confianza'] = np.round(conf, 3)
    bl['usar'] = usar

    u = bl[bl.usar == 1]
    log('\n  reparto de conductas (solo las filas usar=1):')
    for c, k in u['clase'].value_counts().items():
        log('    %-14s %3d bloques  (%4.0f s)' % (c, k, k * lib.BLOCK_S))
    n_act = int((bl.clase == 'activa').sum())
    if n_act:
        log('  de esos, %d son "activa": no inmovil, sin decidir si nada o trepa' % n_act)
    log('  aprovechables: %d de %d bloques (%.0f%%)   dudosos: %d'
        % (len(u), len(bl), 100 * len(u) / max(len(bl), 1), len(bl) - len(u)))
    return bl


def una(ruta, args, paq, log=print):
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    t0 = time.time()
    log('\n%s' % ('=' * 66))
    log('%s' % nombre)
    log('=' * 66)

    fps, n, W, H = lib.info_video(ruta)
    log('  %dx%d  %.2f fps  %d fotogramas  %.1f s%s'
        % (W, H, fps, n, n / fps, '   (enderezado %d grados)' % lib.ROT if lib.ROT else ''))
    if H > W:
        # El aparato es mas ancho que alto: un fotograma vertical casi siempre
        # es una camara girada. No se corrige solo porque la direccion del
        # giro no se puede adivinar con seguridad.
        log('  AVISO: el video queda en vertical. Si en la imagen de control los\n'
            '  cilindros salen acostados, repitelo con --rotar=izquierda o\n'
            '  --rotar=derecha (hacia donde haya que girarlo para enderezarlo).')

    log('  1/5 alineando la camara')
    lut, WH, fps = lib.registrar(ruta, paso=args.paso, log=log)
    log('  2/5 modelando el fondo')
    fondo, occ = lib.fondo_y_ocupacion(ruta, lut, WH, fps, log=log)
    log('  3/5 buscando los tubos y la linea de agua')
    g = lib.detectar_geometria(fondo, occ, n_tubos=args.tubos, log=log)

    ctrl = os.path.join(args.salida, nombre + '_control.jpg')
    lib.dibujar_geometria(ruta, lut, WH, g, ctrl)
    log('      control -> %s' % ctrl)

    log('  4/5 midiendo al animal')
    filas = lib.extraer(ruta, lut, WH, fondo, g, fps, log=log)
    log('  5/5 agregando en bloques de 5 s')
    bl = lib.bloques(filas, g, fps, log=log)

    # Las medidas de cada fotograma se guardan tal cual: con ellas se calculan
    # los segundos (segundos.py) o cualquier otra ventana sin volver a leer el
    # video, que es lo que cuesta diez minutos.
    cu = pd.DataFrame(filas)
    cu['gx0'] = cu['s'].map(lambda s: g['tubos'][s][0])
    cu['gx1'] = cu['s'].map(lambda s: g['tubos'][s][1])
    cu['g_agua'], cu['g_fondo'] = g['y_agua'], g['y_fondo']
    cu['escala'] = float(np.nanmedian(cu['largo']))     # la misma que usa bloques()
    cu.insert(0, 'paso', args.paso)
    dest_cu = os.path.join(args.salida, nombre + '_cuadros.csv')
    cu.round(4).to_csv(dest_cu, index=False, encoding='utf-8')
    log('      medidas por fotograma -> %s' % dest_cu)

    bl = lib.cajas_crudas(bl, lut, g, fps, WH)
    bl.insert(0, 'video', nombre)
    bl.insert(2, 'rotar', lib.ROT)
    # Queda anotado porque tres rasgos (path, rng, spanx) dependen de cada
    # cuantos fotogramas se mide, y mezclar pasos degrada el clasificador sin
    # dar ningun error. Medido en IMG_0826: paso 12 acierta 0.880 y paso 2
    # solo 0.770, porque con mas muestras el temblor del recorte se acumula.
    bl.insert(1, 'paso', args.paso)
    perdidos = int((bl['n_visto'] < 5).sum())
    if perdidos:
        log('  AVISO: %d bloques sin animal visible (quedan sin rasgos)' % perdidos)

    if paq:
        bl = proponer(bl, paq, args.umbral, log=log)
    else:
        log('\n  sin --modelo: solo se sacaron los rasgos, sin etiquetar')

    dest = os.path.join(args.salida, nombre + '_rasgos.csv')
    bl.to_csv(dest, index=False, encoding='utf-8-sig')
    log('  -> %s   (%.0f s)' % (dest, time.time() - t0))

    if paq:
        # Con el mismo modelo, una ventana que se corre de segundo en segundo:
        # el conteo por segundo que pidio el laboratorio (ver segundos.py)
        import argparse as _ap
        import segundos
        segundos.proponer(_ap.Namespace(archivos=[dest_cu], modelo=args.modelo,
                                        ventana=segundos.VENTANA, umbral=args.umbral))
    return dest


def reproponer(ruta, args, paq, log=print):
    """Vuelve a proponer sobre un _rasgos.csv ya existente."""
    log('\n%s\n%s   (solo propuestas, sin reprocesar el video)\n%s'
        % ('=' * 66, os.path.basename(ruta), '=' * 66))
    if not paq:
        raise SystemExit('para volver a proponer sobre un CSV hace falta --modelo')
    bl = pd.read_csv(ruta, encoding='utf-8-sig')
    paso_csv = int(bl['paso'].iloc[0]) if 'paso' in bl.columns else None
    if paq.get('paso') and paso_csv and paso_csv != paq['paso']:
        raise SystemExit('%s se extrajo con --paso=%d y el modelo es de --paso=%d. '
                         'No son comparables.' % (ruta, paso_csv, paq['paso']))
    bl = proponer(bl, paq, args.umbral, log=log)
    bl.to_csv(ruta, index=False, encoding='utf-8-sig')
    log('  -> %s' % ruta)
    return ruta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('videos', nargs='+', help='videos, o _rasgos.csv ya sacados')
    p.add_argument('--modelo', default=None)
    p.add_argument('--tubos', type=int, default=None)
    p.add_argument('--paso', type=int, default=12)
    p.add_argument('--umbral', type=float, default=0.80)
    p.add_argument('--salida', default='etiquetas')
    p.add_argument('--rotar', default='0',
                   help='endereza un video grabado girado: derecha, izquierda o 180')
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    giros = {'0': 0, 'no': 0, 'derecha': 90, '90': 90, '180': 180,
             'izquierda': 270, '270': 270, '-90': 270}
    if args.rotar.lower() not in giros:
        raise SystemExit('--rotar acepta: derecha, izquierda o 180 (recibi %r)' % args.rotar)
    lib.ROT = giros[args.rotar.lower()]

    # Se comprueba ANTES de tocar ningun video: procesar uno cuesta diez
    # minutos y no tiene sentido gastarlos para fallar al final.
    paq = None
    if args.modelo:
        import joblib
        paq = joblib.load(args.modelo)
        paso_mod = paq.get('paso')
        hay_videos = any(not v.lower().endswith('.csv') for v in args.videos)
        if hay_videos and paso_mod and paso_mod != args.paso:
            raise SystemExit(
                'Este modelo se entreno con --paso=%d y pediste --paso=%d.\n'
                'Los rasgos path, rng y spanx cambian con el paso, asi que las\n'
                'etiquetas saldrian sesgadas sin dar ningun error.\n'
                'Vuelve a lanzarlo con --paso=%d.' % (paso_mod, args.paso, paso_mod))

    hechos, rotos = [], []
    for v in args.videos:
        try:
            if v.lower().endswith('.csv'):
                hechos.append(reproponer(v, args, paq))
            else:
                hechos.append(una(v, args, paq))
        except SystemExit:
            raise
        except Exception as e:
            rotos.append((v, str(e)))
            print('  FALLO en %s: %s' % (v, e))

    print('\n%s' % ('=' * 66))
    print('listos: %d   fallidos: %d' % (len(hechos), len(rotos)))
    for v, e in rotos:
        print('  %s -> %s' % (os.path.basename(v), e))
    if any(not v.lower().endswith('.csv') for v in args.videos):
        print('\nAHORA: abre todos los *_control.jpg y descarta los videos donde las')
        print('lineas no esten sobre el agua y el suelo. Eso toma 5 s por video y')
        print('es lo unico que evita meter basura al entrenamiento.')


if __name__ == '__main__':
    main()
