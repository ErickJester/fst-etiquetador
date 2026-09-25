"""Entrena el clasificador con uno o varios videos ya etiquetados a mano.

    py entrenar.py rasgos1.csv=etiquetas1.csv rasgos2.csv=etiquetas2.csv ...
    py entrenar.py --carpeta etiquetas --etiquetas mano

El archivo de rasgos lo produce etiquetar.py. El de etiquetas es tu puntuacion
manual y necesita al menos: bloque, especimen, clase. Si trae una columna
'confianza', por defecto se descartan las filas de confianza baja: una etiqueta
en la que tu dudaste no ensena nada, solo mete ruido.

La validacion es POR VIDEO, no por bloque ni por animal. Dos bloques seguidos
del mismo animal son casi el mismo fotograma, asi que si uno esta en
entrenamiento y el otro en prueba, la nota sale regalada. Con un solo video
solo puede agrupar por animal, y lo avisa.

Guarda el modelo junto con la lista de columnas y en el orden correcto, para
que predecir nunca pueda usar columnas desordenadas en silencio.
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import lib


def limpia(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace('é', 'e').replace('í', 'i')
                  .replace('ó', 'o').replace('á', 'a').replace('ú', 'u')
                  for c in df.columns]
    return df


def cargar(pares, usar_dudosas):
    trozos = []
    for rr, re_ in pares:
        r = limpia(pd.read_csv(rr, encoding='utf-8-sig'))
        e = limpia(pd.read_csv(re_, encoding='utf-8-sig'))
        for c in ('bloque', 'especimen'):
            if c not in r.columns or c not in e.columns:
                raise SystemExit('falta la columna "%s" en %s o %s' % (c, rr, re_))
        if 'clase' not in e.columns:
            raise SystemExit('%s no tiene columna "clase"' % re_)
        cols = ['bloque', 'especimen', 'clase']
        if 'confianza' in e.columns:
            cols.append('confianza')
        j = r.merge(e[cols], on=['bloque', 'especimen'], how='inner')
        if 'video' not in j.columns:
            j['video'] = os.path.splitext(os.path.basename(rr))[0]
        n0 = len(j)
        if 'confianza' in j.columns and not usar_dudosas:
            j = j[~j['confianza'].astype(str).str.lower().str.startswith('baj')]
        print('  %-34s %3d etiquetas cruzadas, %d usadas'
              % (os.path.basename(rr), n0, len(j)))
        trozos.append(j)
    d = pd.concat(trozos, ignore_index=True)
    d = d.dropna(subset=lib.RASGOS)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pares', nargs='*', help='rasgos.csv=etiquetas.csv')
    p.add_argument('--carpeta', default=None,
                   help='carpeta con *_rasgos.csv; busca el CSV de etiquetas al lado')
    p.add_argument('--etiquetas', default='etiquetas',
                   help='sufijo de los archivos de etiquetas dentro de --carpeta')
    p.add_argument('--usar-dudosas', action='store_true',
                   help='incluir tambien las etiquetas de confianza baja')
    p.add_argument('--salida', default='modelo_fst.joblib')
    args = p.parse_args()

    pares = []
    for x in args.pares:
        if '=' not in x:
            raise SystemExit('formato: rasgos.csv=etiquetas.csv   (recibi %r)' % x)
        a, b = x.split('=', 1)
        pares.append((a, b))
    if args.carpeta:
        for rr in sorted(glob.glob(os.path.join(args.carpeta, '*_rasgos.csv'))):
            cand = rr.replace('_rasgos.csv', '_%s.csv' % args.etiquetas)
            if os.path.exists(cand):
                pares.append((rr, cand))
    if not pares:
        raise SystemExit('no hay nada que entrenar')

    print('cargando:')
    d = cargar(pares, args.usar_dudosas)
    print('\n%d bloques utiles, %d videos, %d clases'
          % (len(d), d['video'].nunique(), d['clase'].nunique()))
    print(d['clase'].value_counts().to_string())
    if len(d) < 60:
        print('\nAVISO: con menos de 60 bloques el numero de validacion no dice nada.')

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 classification_report, confusion_matrix)

    X = d[lib.RASGOS].to_numpy(float)
    y = d['clase'].to_numpy()
    if d['video'].nunique() > 1:
        grupos, como = d['video'].to_numpy(), 'video'
    else:
        grupos = ('v' + d['video'].astype(str) + '_a' + d['especimen'].astype(str)).to_numpy()
        como = 'animal (solo hay 1 video; la nota sale optimista)'
    k = min(pd.unique(grupos).size, 5)

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=15, l2_regularization=1.0,
                                         random_state=0)
    print('\nvalidando por %s, en %d partes' % (como, k))
    pred = cross_val_predict(clf, X, y, groups=grupos, cv=GroupKFold(n_splits=k))
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    clases = sorted(pd.unique(y))

    print('\n  adivinar la clase mas comun : %.3f'
          % pd.Series(y).value_counts(normalize=True).max())
    print('  exactitud                   : %.3f' % accuracy_score(y, pred))
    print('  kappa                       : %+.3f' % cohen_kappa_score(y, pred))
    for c in clases:
        print('  acierto en %-14s   : %.3f' % (c, rep[c]['f1-score']))

    cm = confusion_matrix(y, pred, labels=clases)
    print('\n  confusion (fila = tu etiqueta, columna = la del modelo):')
    print('  ' + ' ' * 14 + ''.join('%12s' % c[:10] for c in clases))
    for i, c in enumerate(clases):
        print('  %-14s' % c + ''.join('%12d' % v for v in cm[i]))

    # con que confianza acierta: define el umbral de --umbral en etiquetar.py
    P = cross_val_predict(clf, X, y, groups=grupos, cv=GroupKFold(n_splits=k),
                          method='predict_proba')
    cl = np.array(clf.fit(X, y).classes_)
    pr, cf = cl[P.argmax(1)], P.max(1)
    print('\n  si descartas los bloques donde el modelo duda:')
    print('  %-8s %8s %8s %10s' % ('umbral', 'quedan', '% total', 'exactitud'))
    for u in (0.0, 0.6, 0.7, 0.8, 0.9):
        m = cf >= u
        if m.sum() < 10:
            continue
        print('  %-8.2f %8d %7.0f%% %10.3f'
              % (u, m.sum(), 100 * m.mean(), accuracy_score(y[m], pr[m])))

    import joblib
    pasos = sorted(pd.unique(d['paso'])) if 'paso' in d.columns else []
    if len(pasos) > 1:
        raise SystemExit(
            'Los rasgos vienen con pasos distintos (%s) y no son comparables:\n'
            'path, rng y spanx cambian con el paso. Vuelve a extraer todos los\n'
            'videos con el mismo --paso antes de entrenar.'
            % ', '.join(str(p) for p in pasos))
    if pasos:
        print('\nextraidos con --paso=%d' % pasos[0])
    joblib.dump(dict(modelo=clf, caracteristicas=list(lib.RASGOS),
                     paso=int(pasos[0]) if pasos else None,
                     clases=list(clf.classes_), n_bloques=len(d),
                     videos=sorted(pd.unique(d['video']).tolist())), args.salida)
    print('\nmodelo -> %s' % args.salida)
    print('usalo asi:  py etiquetar.py nuevo.MOV --modelo %s --tubos 4' % args.salida)


if __name__ == '__main__':
    main()
