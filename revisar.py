"""Prepara la revision humana de un video ya etiquetado.

    py revisar.py etiquetas/IMG_0840_rasgos.csv videos_sin_etiquetar/IMG_0840.MOV

Deja dos cosas:

  revision/<video>/          un videito de 5 s por cada bloque a revisar,
                             con el tubo marcado en rojo y la propuesta escrita
  etiquetas/<video>_mano.csv la hoja donde escribes tu respuesta

Por defecto solo saca los bloques donde el programa dudo (usar = 0). Con
--todos saca los 5 minutos completos, por si quieres revisar de cero.

Como se rellena la hoja: los bloques seguros vienen ya escritos con la
propuesta del programa. Los dudosos vienen con la casilla 'clase' VACIA. Ves su
videito, escribes una de las tres palabras, y guardas.

    inmovilidad    flota, solo mueve lo justo para respirar, no cambia de sitio
    nado           se desplaza, cruza al menos dos cuartos del cilindro
    escalamiento   patea la pared con las patas delanteras, rompe la superficie

Si un bloque no se entiende ni viendolo, dejalo vacio: se descarta y no
contamina el entrenamiento. Es mejor una etiqueta menos que una inventada.
"""
import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd

BLOQUE_S = 5.0
CLASES = ['inmovilidad', 'nado', 'escalamiento']


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rasgos', help='el CSV que dejo etiquetar.py')
    p.add_argument('video')
    p.add_argument('--todos', action='store_true',
                   help='sacar todos los bloques, no solo los dudosos')
    p.add_argument('--salida', default='revision')
    p.add_argument('--ancho', type=int, default=900,
                   help='ancho del videito resultante')
    args = p.parse_args()

    d = pd.read_csv(args.rasgos, encoding='utf-8-sig')
    nombre = os.path.splitext(os.path.basename(args.video))[0]
    if 'clase' not in d.columns:
        raise SystemExit('ese CSV no tiene etiquetas; corre etiquetar.py con --modelo')

    dudosos = d if args.todos else d[d.get('usar', 1) == 0]
    dudosos = dudosos.sort_values(['bloque', 'especimen'])
    print('%d bloques en total, %d a revisar' % (len(d), len(dudosos)))
    if not len(dudosos):
        print('No hay ninguno dudoso. Puedes entrenar directo con este video.')

    carpeta = os.path.join(args.salida, nombre)
    os.makedirs(carpeta, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('no se pudo abrir el video: %s' % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    esc = args.ancho / W
    dim = (int(W * esc), int(H * esc))
    hay_geo = all(c in d.columns for c in ('gx0', 'gx1', 'g_agua', 'g_fondo'))
    if not hay_geo:
        print('AVISO: el CSV no trae la geometria, los videitos saldran sin marcar')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for k, (_, r) in enumerate(dudosos.iterrows(), 1):
        t0 = (int(r.bloque) - 1) * BLOQUE_S
        destino = os.path.join(carpeta, 'b%03d_tubo%d.mp4'
                               % (int(r.bloque), int(r.especimen)))
        out = cv2.VideoWriter(destino, fourcc, fps, dim)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
        for _ in range(int(BLOQUE_S * fps)):
            ok, fr = cap.read()
            if not ok:
                break
            if hay_geo:
                cv2.rectangle(fr, (int(r.gx0), int(r.g_agua) - 200),
                              (int(r.gx1), int(r.g_fondo)), (0, 0, 255), 4)
            fr = cv2.resize(fr, dim, interpolation=cv2.INTER_AREA)
            cv2.rectangle(fr, (0, 0), (dim[0], 34), (0, 0, 0), -1)
            cv2.putText(fr, 'bloque %d  tubo %d  %.0f-%.0f s   propuesta: %s (%.2f)'
                        % (r.bloque, r.especimen, t0, t0 + BLOQUE_S,
                           r.clase, r.get('confianza', float('nan'))),
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            out.write(fr)
        out.release()
        if k % 10 == 0 or k == len(dudosos):
            print('  %d/%d videitos' % (k, len(dudosos)))
    cap.release()

    # --- la hoja ---
    hoja = d[['bloque', 'especimen']].copy()
    hoja['inicio_s'] = (d['bloque'] - 1) * BLOQUE_S
    hoja['propuesta'] = d['clase']
    hoja['seguridad'] = d.get('confianza')
    a_revisar = d.get('usar', pd.Series(1, index=d.index)) == 0
    if args.todos:
        a_revisar[:] = True
    # los seguros ya vienen escritos; los dudosos, en blanco para que los pongas tu
    hoja['clase'] = d['clase'].where(~a_revisar, '')
    hoja['videito'] = np.where(
        a_revisar, ['b%03d_tubo%d.mp4' % (b, s)
                    for b, s in zip(d['bloque'], d['especimen'])], '')
    destino = os.path.join(os.path.dirname(args.rasgos) or '.',
                           nombre + '_mano.csv')
    if os.path.exists(destino):
        destino = destino.replace('.csv', '_NUEVO.csv')
        print('AVISO: ya habia un _mano.csv, este se guarda como %s'
              % os.path.basename(destino))
    hoja.to_csv(destino, index=False, encoding='utf-8-sig')

    print('\nvideitos -> %s' % carpeta)
    print('hoja     -> %s' % destino)
    print('\nQUE HACER AHORA:')
    print('  1. Abre la hoja. Filtra las filas con la casilla "clase" vacia.')
    print('  2. Por cada una, abre el videito que dice la columna "videito".')
    print('  3. Escribe en "clase" una de estas tres palabras:')
    for c in CLASES:
        print('       %s' % c)
    print('  4. Si no se entiende ni viendolo, dejalo vacio. Se descarta.')
    print('  5. Guarda en el mismo sitio y con el mismo nombre.')
    print('\nDespues:  py entrenar.py --carpeta etiquetas --etiquetas mano')


if __name__ == '__main__':
    main()
