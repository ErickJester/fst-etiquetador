"""Etiquetador automatico de FST. Un video entra, un CSV etiquetado sale.

    py etiquetar.py video1.MOV video2.MOV ...

Opciones:
    --modelo M.joblib   clasificador entrenado; sin el, solo saca rasgos
    --tubos N           cuantos cilindros esperar (recomendado: ponlo siempre)
    --paso N            analizar 1 de cada N fotogramas (2 = ~15/s, por defecto)
    --umbral U          confianza minima para dar por buena la etiqueta (0.80)
    --salida DIR        carpeta de resultados (por defecto ./etiquetas)

Por cada video deja:
    <nombre>_rasgos.csv        un renglon por bloque de 5 s y por especimen
    <nombre>_control.jpg       MIRALA. Si las lineas no caen sobre el agua y
                               el suelo del tubo, ese video no sirve.

La columna 'usar' vale 1 solo si el clasificador decidio con confianza
suficiente. Para entrenar, quedate nada mas con las filas usar=1: menos
etiquetas pero mas limpias sale mejor que muchas etiquetas dudosas.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import lib


def una(ruta, args, log=print):
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    t0 = time.time()
    log('\n%s' % ('=' * 66))
    log('%s' % nombre)
    log('=' * 66)

    fps, n, W, H = lib.info_video(ruta)
    log('  %dx%d  %.2f fps  %d fotogramas  %.1f s' % (W, H, fps, n, n / fps))

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

    bl.insert(0, 'video', nombre)
    # Queda anotado porque tres rasgos (path, rng, spanx) dependen de cada
    # cuantos fotogramas se mide, y mezclar pasos degrada el clasificador sin
    # dar ningun error. Medido en IMG_0826: paso 12 acierta 0.880 y paso 2
    # solo 0.770, porque con mas muestras el temblor del recorte se acumula.
    bl.insert(1, 'paso', args.paso)
    perdidos = int((bl['n_visto'] < 5).sum())
    if perdidos:
        log('  AVISO: %d bloques sin animal visible (quedan sin rasgos)' % perdidos)

    if args.modelo:
        import joblib
        paq = joblib.load(args.modelo)
        mod, feats = paq['modelo'], paq['caracteristicas']
        falta = [c for c in feats if c not in bl.columns]
        if falta:
            raise SystemExit('el modelo pide columnas que no existen: %s' % falta)
        X = bl[feats].to_numpy(float)
        val = ~np.isnan(X).any(1)
        bl['clase'] = ''
        bl['confianza'] = np.nan
        if val.any():
            P = mod.predict_proba(X[val])
            bl.loc[val, 'clase'] = mod.classes_[P.argmax(1)]
            bl.loc[val, 'confianza'] = P.max(1).round(3)
        bl['usar'] = ((bl['confianza'] >= args.umbral) & val).astype(int)
        log('\n  reparto de conductas (solo las filas usar=1):')
        u = bl[bl.usar == 1]
        if len(u):
            for c, k in u['clase'].value_counts().items():
                log('    %-14s %3d bloques  (%4.0f s)' % (c, k, k * lib.BLOCK_S))
        log('  aprovechables: %d de %d bloques (%.0f%%)'
            % (len(u), len(bl), 100 * len(u) / max(len(bl), 1)))
    else:
        log('\n  sin --modelo: solo se sacaron los rasgos, sin etiquetar')

    dest = os.path.join(args.salida, nombre + '_rasgos.csv')
    bl.to_csv(dest, index=False, encoding='utf-8-sig')
    log('  -> %s   (%.0f s)' % (dest, time.time() - t0))
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('videos', nargs='+')
    p.add_argument('--modelo', default=None)
    p.add_argument('--tubos', type=int, default=None)
    p.add_argument('--paso', type=int, default=2)
    p.add_argument('--umbral', type=float, default=0.80)
    p.add_argument('--salida', default='etiquetas')
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    # Se comprueba ANTES de tocar ningun video: procesar uno cuesta diez
    # minutos y no tiene sentido gastarlos para fallar al final.
    if args.modelo:
        import joblib
        paso_mod = joblib.load(args.modelo).get('paso')
        if paso_mod and paso_mod != args.paso:
            raise SystemExit(
                'Este modelo se entreno con --paso=%d y pediste --paso=%d.\n'
                'Los rasgos path, rng y spanx cambian con el paso, asi que las\n'
                'etiquetas saldrian sesgadas sin dar ningun error.\n'
                'Vuelve a lanzarlo con --paso=%d.' % (paso_mod, args.paso, paso_mod))

    hechos, rotos = [], []
    for v in args.videos:
        try:
            hechos.append(una(v, args))
        except Exception as e:
            rotos.append((v, str(e)))
            print('  FALLO en %s: %s' % (v, e))

    print('\n%s' % ('=' * 66))
    print('listos: %d   fallidos: %d' % (len(hechos), len(rotos)))
    for v, e in rotos:
        print('  %s -> %s' % (os.path.basename(v), e))
    print('\nAHORA: abre todos los *_control.jpg y descarta los videos donde las')
    print('lineas no esten sobre el agua y el suelo. Eso toma 5 s por video y')
    print('es lo unico que evita meter basura al entrenamiento.')


if __name__ == '__main__':
    main()
