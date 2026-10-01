"""Conducta SEGUNDO A SEGUNDO, con el mismo modelo que etiqueta por bloques.

    py segundos.py proponer etiquetas/VIDEO_cuadros.csv
    py segundos.py validar

Necesita los _cuadros.csv que deja etiquetar.py (las medidas de cada
fotograma). Un video procesado antes de que existieran hay que pasarlo otra
vez por etiquetar.py, una sola vez.

La idea. Para decidir el segundo k se miden los mismos 15 rasgos de siempre en
una ventana centrada en el (por defecto 4 s, de k-1.5 a k+2.5) y se le pasa al
modelo de bloques, modelo_fst.joblib. La ventana se corre de segundo en
segundo, asi que cada segundo tiene su propia decision y aparecen las
conductas cortas que la regla de la mayoria de Porsolt escondia.

No hace falta entrenar nada nuevo: el modelo aprendio de los bloques de 5 s y
una ventana de 4-5 s se mide igual que un bloque. Medido en IMG_0826,
IMG_0840 y AcuN1, cada video con un modelo que no lo vio, al juntar los
segundos de cada bloque y tomar la mayoria:

    modelo de bloques, como siempre       exactitud 0.852  kappa +0.720
    ventana de 5 s, segundo a segundo     exactitud 0.857  kappa +0.732
    ventana de 4 s, segundo a segundo     exactitud 0.843  kappa +0.709
    ventana de 3 s                        demasiado corta: ~7 mediciones con
                                          --paso 12, el acierto se desploma

Una ventana mas corta ve conductas mas breves, pero mide con menos datos. 4 s
es el punto medio: detecta conductas de unos 2 s.

proponer   deja VIDEO_segundos.csv (un renglon por rata y segundo; lo abre
           revisor_segundos.html) y VIDEO_resumen.csv (segundos de cada
           conducta por rata, por minuto y en total)
validar    repite la medida de arriba con todos los videos que tengan
           _cuadros.csv y _mano.csv
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import lib

ACTIVAS = ('nado', 'escalamiento')
VENTANA = 4.0


def tiempo(s):
    return '%d:%02d' % (s // 60, s % 60)


def regla(P, clases, umbral):
    """La misma regla de tres pasos que etiquetar.py, segundo a segundo."""
    act = sum(P[:, clases.index(c)] for c in ACTIVAS if c in clases)
    clase, conf = [], []
    for p, a in zip(P, act):
        j = int(np.argmax(p))
        if p[j] >= umbral:
            clase.append(clases[j]); conf.append(p[j])
        elif a >= umbral:
            clase.append('activa'); conf.append(a)
        else:
            clase.append(clases[j]); conf.append(p[j])
    return clase, np.round(conf, 3)


def medir(cuadros_csv, ventana, paso_modelo=None):
    cu = pd.read_csv(cuadros_csv)
    if paso_modelo and 'paso' in cu.columns and int(cu['paso'].iloc[0]) != paso_modelo:
        raise SystemExit('%s se saco con --paso=%d y el modelo es de --paso=%d'
                         % (cuadros_csv, cu['paso'].iloc[0], paso_modelo))
    return lib.segundos(cu, ventana)


def resumen(out):
    cols = ['nado', 'inmovilidad', 'escalamiento', 'activa']
    out = out.assign(minuto=out['inicio_s'] // 60 + 1)
    filas = []
    for e, de in out.groupby('especimen'):
        for mi, d in list(de.groupby('minuto')) + [('total', de)]:
            v = d['clase'].value_counts()
            filas.append(dict(especimen=e, minuto=mi, **{k: int(v.get(k, 0)) for k in cols},
                              sin_decidir=int((d['clase'] == '').sum())))
    return pd.DataFrame(filas)


def proponer(a):
    import joblib
    paq = joblib.load(a.modelo)
    clases, feats = list(paq['clases']), paq['caracteristicas']
    for cu in a.archivos:
        nombre = os.path.basename(cu)[:-len('_cuadros.csv')]
        S = medir(cu, a.ventana, paq.get('paso'))
        X = S[feats].to_numpy(float)
        ok = ~np.isnan(X).any(axis=1)
        P = np.full((len(S), len(clases)), np.nan)
        if ok.any():
            P[ok] = paq['modelo'].predict_proba(X[ok])
        c, conf = regla(np.nan_to_num(P), clases, a.umbral)
        out = S[['especimen', 'segundo', 'inicio_s']].copy()
        out.insert(3, 'tiempo', [tiempo(int(s)) for s in out['inicio_s']])
        out['clase'] = np.where(ok, c, '')
        out['confianza'] = np.where(ok, conf, np.nan)
        for j, k in enumerate(clases):
            out['p_' + k] = np.round(P[:, j], 3)
        out['ventana_s'] = a.ventana
        carpeta = os.path.dirname(cu) or '.'
        dest = os.path.join(carpeta, nombre + '_segundos.csv')
        out.to_csv(dest, index=False, encoding='utf-8-sig')
        res = resumen(out)
        dres = os.path.join(carpeta, nombre + '_resumen.csv')
        res.to_csv(dres, index=False, encoding='utf-8-sig')

        print('\n%s: %d segundos por rata, ventana de %.0f s' % (nombre, out['segundo'].max(), a.ventana))
        print('segundos de cada conducta:')
        print(res[res['minuto'] == 'total'].drop(columns='minuto').to_string(index=False))
        print('-> %s\n-> %s' % (dest, dres))


def validar(a):
    """Cada video se predice con un modelo de bloques entrenado sin el; se
    juntan sus segundos por bloque y se compara la mayoria con tu bloque."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support as prfs
    clases = ['escalamiento', 'inmovilidad', 'nado']

    def bloques_de(v):
        r = pd.read_csv(os.path.join(a.carpeta, v + '_rasgos.csv'), encoding='utf-8-sig')
        r = r.drop(columns=[c for c in ('clase', 'confianza', 'usar') if c in r.columns])
        m = pd.read_csv(os.path.join(a.carpeta, v + '_mano.csv'), encoding='utf-8-sig')
        d = r.merge(m[['bloque', 'especimen', 'clase']], on=['bloque', 'especimen'])
        d['clase'] = d['clase'].fillna('')
        return d[d['clase'].isin(clases)].dropna(subset=lib.RASGOS)

    todos = sorted(os.path.basename(f)[:-len('_mano.csv')]
                   for f in glob.glob(os.path.join(a.carpeta, '*_mano.csv'))
                   if os.path.exists(f.replace('_mano.csv', '_rasgos.csv')))
    con_cu = [v for v in todos if os.path.exists(os.path.join(a.carpeta, v + '_cuadros.csv'))]
    if not con_cu:
        raise SystemExit('Ningun video tiene _cuadros.csv todavia: pasalos por etiquetar.py.')
    print('se prueban %d videos con _cuadros.csv; se entrena con los %d que tienen etiquetas'
          % (len(con_cu), len(todos)))
    B = {v: bloques_de(v) for v in todos}
    res = []
    for v in con_cu:
        tr = pd.concat([B[o] for o in todos if o != v])
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=0)
        mdl.fit(tr[lib.RASGOS].to_numpy(float), tr['clase'])
        S = medir(os.path.join(a.carpeta, v + '_cuadros.csv'), a.ventana)
        S = S[S[lib.RASGOS].notna().all(axis=1)].copy()
        S['pred'] = mdl.predict(S[lib.RASGOS].to_numpy(float))
        m = pd.read_csv(os.path.join(a.carpeta, v + '_mano.csv'), encoding='utf-8-sig')
        lab = {(e, b): c for e, b, c in zip(m['especimen'], m['bloque'], m['clase'].fillna(''))}
        S['bloque'] = S['inicio_s'] // lib.BLOCK_S + 1
        S['tuya'] = [lab.get((e, b), '') for e, b in zip(S['especimen'], S['bloque'])]
        S['video'] = v
        res.append(S[S['tuya'].isin(clases)])
        print('  %-16s listo' % v)
    D = pd.concat(res)
    g = D.groupby(['video', 'especimen', 'bloque'])
    may = g.agg(tuya=('tuya', 'first'), maq=('pred', lambda x: x.value_counts().idxmax()))
    f1 = prfs(may['tuya'], may['maq'], labels=clases, zero_division=0)[2]
    print('\nventana de %.0f s, mayoria de los segundos de cada bloque contra tu bloque (%d bloques):'
          % (a.ventana, len(may)))
    print('  exactitud %.3f   kappa %+.3f' % ((may['tuya'] == may['maq']).mean(),
                                            cohen_kappa_score(may['tuya'], may['maq'])))
    for c, f in zip(clases, f1):
        print('  F1 %-13s %.3f' % (c, f))
    mez = g['pred'].nunique()
    print('\nbloques donde aparece mas de una conducta: %d de %d (%.0f%%)'
          % ((mez > 1).sum(), len(mez), 100 * (mez > 1).mean()))
    print('segundos por conducta     tus bloques (x5)   segundo a segundo')
    vb, vs = D['tuya'].value_counts(), D['pred'].value_counts()
    for c in clases:
        print('  %-22s %10d %18d' % (c, vb.get(c, 0), vs.get(c, 0)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('accion', choices=['proponer', 'validar'])
    p.add_argument('archivos', nargs='*', help='para proponer: uno o varios *_cuadros.csv')
    p.add_argument('--modelo', default='modelo_fst.joblib')
    p.add_argument('--ventana', type=float, default=VENTANA,
                   help='segundos que se miran alrededor de cada segundo (4)')
    p.add_argument('--umbral', type=float, default=0.80)
    p.add_argument('--carpeta', default='etiquetas', help='para validar')
    a = p.parse_args()
    if a.accion == 'validar':
        validar(a)
    else:
        if not a.archivos:
            p.error('dime que _cuadros.csv proponer')
        proponer(a)


if __name__ == '__main__':
    main()
