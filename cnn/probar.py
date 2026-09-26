"""Pasa la CNN entrenada por un video y deja sus etiquetas para verlas en revisor.html.

    py cnn/probar.py etiquetas/NUEVO_rasgos.csv --red fst_cnn.pt

Corre en la PC, sin GPU: solo predice, no entrena (unos 2-4 min por video).

Necesita el _rasgos.csv del video porque ahi estan los recuadros de los tubos.
Si el video es nuevo, primero:

    py etiquetar.py videos_sin_etiquetar/NUEVO.MOV --tubos 4

Deja etiquetas/NUEVO_cnn.csv, con el mismo formato que un _rasgos.csv pero con
las propuestas de la CNN en clase, confianza, usar y p_*. Abrelo en
revisor.html junto con el video para ver lo que dice la CNN bloque por bloque.

Si ademas existe NUEVO_mano.csv, compara la CNN contra tus etiquetas.

Opciones:
    --red R.pt       red guardada por la celda 5 del cuaderno (fst_cnn.pt)
    --video V        el video; si no, lo busca por nombre en videos_sin_etiquetar
    --umbral U       confianza minima, igual que en etiquetar.py (0.80)
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
import fst_cnn
import recortar

ACTIVAS = ('nado', 'escalamiento')


def cargar_red(ruta):
    paq = torch.load(ruta, map_location='cpu', weights_only=False)
    fst_cnn.ENTRADA = paq.get('entrada', 'trios')
    med = paq.get('medidas')
    m = fst_cnn.red(preentrenada=False, n_med=len(med) if med else None)
    m.load_state_dict(paq['pesos'])
    m.eval()
    print('red %s: version %s, entrada %s, %d fotogramas, entrenada con %d clips'
          % (os.path.basename(ruta), paq.get('version', '?'), fst_cnn.ENTRADA,
             paq['frames'], paq.get('n_clips', 0)))
    return m, paq


def proponer(P, clases, umbral):
    """La misma regla de tres pasos que etiquetar.py."""
    act = sum(P[:, clases.index(c)] for c in ACTIVAS)
    clase, conf, usar = [], [], []
    for p, a in zip(P, act):
        j = int(p.argmax())
        if p[j] >= umbral:
            clase.append(clases[j]); conf.append(p[j]); usar.append(1)
        elif a >= umbral:
            clase.append('activa'); conf.append(a); usar.append(1)
        else:
            clase.append(clases[j]); conf.append(p[j]); usar.append(0)
    return clase, np.round(conf, 3), usar


def comparar(d, mano_csv, vistos):
    from sklearn.metrics import cohen_kappa_score
    m = pd.read_csv(mano_csv, encoding='utf-8-sig')[['bloque', 'especimen', 'clase']]
    j = d[['bloque', 'especimen', 'p_inmovilidad']].merge(m, on=['bloque', 'especimen'])
    j['clase'] = j['clase'].fillna('').astype(str).str.strip().str.lower()
    tres = j[j['clase'].isin(fst_cnn.CLASES)]
    print('\ncontra tu %s:' % os.path.basename(mano_csv))
    if vistos:
        print('  OJO: la red se entreno con este video. La nota sale inflada;')
        print('  para medirla de verdad usa un video que no este en la lista.')
    # la CNN puede proponer 'activa'; para las 3 conductas se usa su mejor clase
    mejor = d.set_index(['bloque', 'especimen'])[['p_' + c for c in fst_cnn.CLASES]]
    # los bloques que no se pudieron recortar no tienen prediccion
    mejor = mejor.dropna().idxmax(axis=1).str[2:]
    tres = tres[tres.set_index(['bloque', 'especimen']).index.isin(mejor.index)]
    j = j.dropna(subset=['p_inmovilidad'])
    pred = tres.set_index(['bloque', 'especimen']).index.map(mejor)
    print('  exactitud (3 conductas) : %.3f   (%d bloques)' % ((pred == tres['clase']).mean(), len(tres)))
    print('  kappa                   : %+.3f' % cohen_kappa_score(tres['clase'], pred))
    ri = j['clase'] == 'inmovilidad'
    pi = j['p_inmovilidad'] >= 0.5
    ok = j['clase'].isin(fst_cnn.CLASES + ['activa'])
    print('  Porsolt, inmovil/activa : %.3f' % (ri[ok] == pi[ok]).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rasgos')
    p.add_argument('--red', default='fst_cnn.pt')
    p.add_argument('--video', default=None)
    p.add_argument('--umbral', type=float, default=0.80)
    a = p.parse_args()

    nombre = os.path.basename(a.rasgos).replace('_rasgos.csv', '')
    carpeta = os.path.dirname(a.rasgos) or '.'
    video = a.video or recortar.buscar_video(nombre, ['videos_sin_etiquetar', '.'])
    if not video:
        raise SystemExit('no encuentro el video de %s; pasalo con --video' % nombre)
    m, paq = cargar_red(a.red)

    d = pd.read_csv(a.rasgos, encoding='utf-8-sig')
    if 'rotar' not in d.columns:
        d['rotar'] = 0
    t = time.time()
    print('recortando %d bloques de %s' % (len(d), os.path.basename(video)))
    X, visto = recortar.recortar(video, d, paq['frames'])
    completos = visto.all(1)
    print('prediciendo (%.0f s)' % (time.time() - t))
    P = np.full((len(d), len(fst_cnn.CLASES)), np.nan)
    if completos.any():
        with torch.no_grad():
            M = None
            if fst_cnn.ENTRADA == 'fusion':
                # mismas medidas que al entrenar: rasgos del CSV + reglas del clip
                R = d.loc[completos, fst_cnn.RASGOS].to_numpy(np.float32)
                M = np.hstack([R, fst_cnn.reglas(X[completos])])
            P[completos] = fst_cnn.predecir(m, X[completos], M=M)

    d = d.drop(columns=[c for c in d.columns
                        if c in ('clase', 'confianza', 'usar') or c.startswith('p_')])
    for j, c in enumerate(fst_cnn.CLASES):
        d['p_' + c] = P[:, j].round(3)
    clase, conf, usar = proponer(np.nan_to_num(P), fst_cnn.CLASES, a.umbral)
    d['clase'] = np.where(completos, clase, '')
    d['confianza'] = np.where(completos, conf, np.nan)
    d['usar'] = np.where(completos, usar, 0)

    dest = os.path.join(carpeta, nombre + '_cnn.csv')
    d.to_csv(dest, index=False, encoding='utf-8-sig')

    u = d[d.usar == 1]
    print('\nreparto (solo usar=1):')
    for c, k in u['clase'].value_counts().items():
        print('  %-14s %3d bloques' % (c, k))
    print('aprovechables: %d de %d   dudosos: %d' % (len(u), len(d), len(d) - len(u)))
    print('-> %s   (%.0f s)' % (dest, time.time() - t))

    mano = os.path.join(carpeta, nombre + '_mano.csv')
    if os.path.exists(mano):
        comparar(d, mano, nombre in paq.get('videos', []))

    print('\nPara verlo: abre revisor.html, arrastra el video y %s' % os.path.basename(dest))


if __name__ == '__main__':
    main()
