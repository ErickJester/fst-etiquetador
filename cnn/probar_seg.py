"""Pasa la CNN POR SEGUNDO por un video y deja la conducta de cada segundo.

    py cnn/probar_seg.py etiquetas/NUEVO_rasgos.csv --red fst_cnn_seg.pt

Corre en la PC, sin GPU: solo predice.

Necesita, del mismo video:
    NUEVO_rasgos.csv   los recuadros de los tubos (bx0..by1)
    NUEVO_cuadros.csv  las medidas por fotograma, para los 15 rasgos
Los dos salen de:  py etiquetar.py videos_sin_etiquetar/NUEVO.MOV --tubos 4

Deja:
    NUEVO_cnnseg.csv          un renglon por rata y segundo, mismo formato que
                              _segundos.csv: abrelo en revisor_segundos.html
                              como "propuesta por segundo"
    NUEVO_cnnseg_resumen.csv  segundos de cada conducta por rata, por minuto
                              y en total (el reporte que pide el laboratorio)

Si existen hojas tuyas (NUEVO_seg*.csv), compara contra los segundos que
marcaste y avisa si el video estaba en el entrenamiento.

Opciones:
    --red R.pt       red guardada por fst_cnn_seg.final (fst_cnn_seg.pt)
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
import fst_cnn_seg
import recortar_seg
import segundos


def cargar_red(ruta):
    paq = torch.load(ruta, map_location='cpu', weights_only=False)
    if paq.get('unidad') != 'segundo':
        raise SystemExit('%s es una red por bloques; para esa usa cnn/probar.py' % ruta)
    if (paq['fps_clip'], paq['ventana']) != (recortar_seg.FPS_CLIP, recortar_seg.VENTANA):
        raise SystemExit('la red se entreno con %s fps y ventana de %s s; recortar_seg usa %s y %s'
                         % (paq['fps_clip'], paq['ventana'], recortar_seg.FPS_CLIP, recortar_seg.VENTANA))
    m = fst_cnn.red(preentrenada=False, n_med=len(paq['medidas']))
    m.load_state_dict(paq['pesos'])
    m.eval()
    print('red %s: entrenada con %d segundos de %d videos'
          % (os.path.basename(ruta), paq['n_clips'], len(paq['videos'])))
    return m, paq


def comparar(out, nombre, carpeta, vistos):
    from sklearn.metrics import cohen_kappa_score
    rutas = recortar_seg.hojas(carpeta).get(nombre, [])
    h = recortar_seg.humanas(rutas) if rutas else None
    if h is None or h.empty:
        return
    j = h.merge(out[['especimen', 'segundo'] + ['p_' + c for c in fst_cnn.CONDUCTAS]],
                on=['especimen', 'segundo']).dropna()
    j['cnn'] = j[['p_' + c for c in fst_cnn.CONDUCTAS]].idxmax(axis=1).str[2:]
    tres = j[j['conducta'].isin(fst_cnn.CONDUCTAS)]
    print('\ncontra tus segundos (%s):' % ', '.join(os.path.basename(r) for r in rutas))
    if vistos:
        print('  OJO: la red se entreno con este video. La nota sale inflada.')
    print('  segundo a segundo, 3 conductas: exactitud %.3f   kappa %+.3f   (%d segundos)'
          % ((tres['conducta'] == tres['cnn']).mean(), cohen_kappa_score(tres['conducta'], tres['cnn']),
             len(tres)))
    ri, pi = j['conducta'] == 'inmovilidad', j['p_inmovilidad'] >= 0.5
    print('  Porsolt, inmovil/activa       : exactitud %.3f' % (ri == pi).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rasgos')
    p.add_argument('--red', default='fst_cnn_seg.pt')
    p.add_argument('--video', default=None)
    p.add_argument('--umbral', type=float, default=0.80)
    a = p.parse_args()

    nombre = os.path.basename(a.rasgos).replace('_rasgos.csv', '')
    carpeta = os.path.dirname(a.rasgos) or '.'
    video = a.video or recortar_seg.buscar_video(nombre, ['videos_sin_etiquetar', '.'])
    if not video:
        raise SystemExit('no encuentro el video de %s; pasalo con --video' % nombre)
    m, paq = cargar_red(a.red)

    R, paso = recortar_seg.rasgos_seg(carpeta, nombre)
    if R is None:
        raise SystemExit('falta %s_cuadros.csv: pasa el video por etiquetar.py' % nombre)
    rasgos = pd.read_csv(a.rasgos, encoding='utf-8-sig')
    esp = sorted(int(e) for e in rasgos['especimen'].unique())
    t = time.time()
    print('recortando %d tubos de %s' % (len(esp), os.path.basename(video)))
    F, n_seg = recortar_seg.peliculas(video, rasgos, esp)

    d = pd.DataFrame([(e, s) for e in esp for s in range(1, n_seg + 1)],
                     columns=['especimen', 'segundo'])
    d = d.merge(R[['especimen', 'segundo'] + fst_cnn.RASGOS], on=['especimen', 'segundo'], how='left')
    Fk = {'%s|%d' % (nombre, e): v for e, v in F.items()}
    X = fst_cnn_seg.armar(Fk, np.full(len(d), nombre), d['especimen'].to_numpy(),
                          d['segundo'].to_numpy(), recortar_seg.FPS_CLIP, recortar_seg.VENTANA)
    print('prediciendo %d segundos (%.0f s)' % (len(d), time.time() - t))
    M = np.hstack([d[fst_cnn.RASGOS].to_numpy(np.float32), fst_cnn.reglas(X)])
    P = fst_cnn.predecir(m, X, M=M)

    out = d[['especimen', 'segundo']].copy()
    out['inicio_s'] = out['segundo'] - 1
    out.insert(3, 'tiempo', [segundos.tiempo(int(s)) for s in out['inicio_s']])
    out['conducta'], out['confianza'] = segundos.regla(P, fst_cnn.CONDUCTAS, a.umbral)
    for j, c in enumerate(fst_cnn.CONDUCTAS):
        out['p_' + c] = P[:, j].round(3)
    out['ventana_s'] = recortar_seg.VENTANA
    dest = os.path.join(carpeta, nombre + '_cnnseg.csv')
    out.to_csv(dest, index=False, encoding='utf-8-sig')
    res = segundos.resumen(out)
    dres = os.path.join(carpeta, nombre + '_cnnseg_resumen.csv')
    res.to_csv(dres, index=False, encoding='utf-8-sig')

    print('\nsegundos de cada conducta por rata:')
    print(res[res['minuto'] == 'total'].drop(columns='minuto').to_string(index=False))
    print('-> %s\n-> %s   (%.0f s)' % (dest, dres, time.time() - t))
    comparar(out, nombre, carpeta, nombre in paq['videos'])
    print('\nPara verlo: abre revisor_segundos.html con el video, %s_rasgos.csv y %s'
          % (nombre, os.path.basename(dest)))


if __name__ == '__main__':
    main()
