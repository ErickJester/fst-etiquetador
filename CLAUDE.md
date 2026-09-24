# Contexto de esta carpeta

Léelo antes de tocar nada.

## Para qué existe

Etiquetar automáticamente videos del test de nado forzado (FST). Cada video
tiene hasta 4 ratas, cada una en su cilindro. Hay que decir, por cada bloque de
5 segundos y cada rata, cuál de tres conductas hizo:

- **inmovilidad** — flota, solo mueve lo justo para respirar, no cambia de sitio
- **nado** — se desplaza, cruza al menos 2 de los 4 cuadrantes del cilindro
- **escalamiento** — patadas rápidas contra la pared que rompen la superficie

Esto **no es el objetivo final**. Es la herramienta que fabrica el conjunto de
entrenamiento para el clasificador definitivo del proyecto de tesis. El usuario
es estudiante de ingeniería y no tiene acceso a anotadores expertos: él mismo es
el etiquetador de referencia.

## Cómo se usa

```bash
py etiquetar.py videos_sin_etiquetar/*.MOV --modelo modelo_fst.joblib --tubos 4
py entrenar.py --carpeta etiquetas --etiquetas mano
```

`--salida` ya apunta a `etiquetas/` por defecto, así que no hay que pasarlo.

El detalle está en `COMO_USARLO.md`.

## Qué hay dentro

| Archivo | Qué es |
|---|---|
| `lib.py` | Registro, fondo, geometría automática, extracción, rasgos |
| `etiquetar.py` | Video → CSV de rasgos (+ etiquetas si se da `--modelo`) |
| `entrenar.py` | CSVs etiquetados → `modelo_fst.joblib` |
| `modelo_fst.joblib` | Modelo actual: 100 bloques, solo `IMG_0826` |
| `videos_sin_etiquetar/` | Videos crudos que el usuario aún no ha procesado |
| `etiquetas/` | Salidas: `*_rasgos.csv`, `*_control.jpg` y `*_mano.csv` |

Convención de nombres dentro de `etiquetas/`, y `entrenar.py --carpeta` depende
de ella: `X_rasgos.csv` son los rasgos que saca el programa y `X_mano.csv` la
puntuación corregida por el usuario para ese mismo video.

## Estado medido (no supuesto)

Sobre `IMG_0826`, validando por animal, contra las 100 etiquetas de confianza
alta y media del usuario:

| | |
|---|---|
| Exactitud | 0.890 |
| Kappa | +0.819 |
| f1 inmovilidad | 0.935 |
| f1 nado | 0.884 |
| f1 escalamiento | 0.786 |
| Exactitud filtrando a confianza ≥0.80 | 0.929 (queda el 85%) |

**Esa nota es optimista.** Sale de un solo video, así que solo puede agrupar por
animal. Nunca se ha probado en un video que el modelo no haya visto. No la
presentes como capacidad de generalizar.

## Decisiones de diseño que NO hay que deshacer

1. **Fondo por percentil 10, no mediana.** La mediana deja incrustado al animal
   inmóvil dentro del fondo y luego no se le puede segmentar.
2. **Registro ORB + RANSAC con similitud**, no solo traslación. La cámara de
   este montaje cambia ~5% de escala; con traslación sola el fondo sale borroso.
3. **La energía de movimiento se mide solo dentro del animal**, no en toda la
   región. El oleaje del agua satura la medida.
4. **Se erosiona la cola antes de medir la postura.** Sin eso los momentos de
   imagen miden la cola, no la verticalidad del cuerpo.
5. **El centroide se suaviza con mediana móvil (k=5) antes** de calcular
   trayectorias. El centroide crudo salta varios px solo por ruido.
6. **Las distancias se dividen por el largo del cuerpo** y los tiempos por
   segundos. Es lo único que hace comparables dos videos con encuadres
   distintos. Si alguien vuelve a medir en píxeles, el modelo aprende el
   encuadre en vez de la conducta.
7. **Las filas de rasgos se agrupan por VIDEO al validar**, no por bloque. Dos
   bloques seguidos del mismo animal son casi el mismo fotograma.
8. **Las etiquetas de confianza baja se descartan al entrenar.** El usuario
   marcó 140 de 240 bloques como dudosos; incluirlos bajaba el acierto.

## Cosas aprendidas a golpes (no repetirlas)

- La geometría se detecta desde la banda **estrecha** donde aparece el animal,
  no de pared a pared. Incluir las paredes metía bordes que tapaban el del
  suelo: el error pasó de 16 px a 45 px.
- **Ensanchar los tubos de los extremos subió el acierto de escalamiento de
  0.55 a 0.79.** La rata trepa pegada a la pared exterior, justo el trozo que
  el recuadro recortaba.
- El último bloque casi nunca cabe completo (un video de 301 s deja 1 s suelto).
  Se descarta; un bloque a medias no es comparable.
- Un intento previo con correlación de fase (solo traslación) y otro con
  perfiles de intensidad por columna fallaron. No reintentarlos.
- El rotulador rojo del pizarrón tiene el mismo gris que el aparato y engaña a
  la detección de bordes por umbral. Por eso la geometría sale del mapa de
  ocupación, no de un umbral sobre la imagen.

## Lo que falta

1. Probar con videos que no sean `IMG_0826`. **Nunca se ha hecho.** Es el
   riesgo abierto: no se sabe si encuentra los tubos en otro encuadre.
2. Etiquetar el video 2 en bola de nieve: el programa propone, el usuario
   corrige solo las filas con `usar = 0` (~20%).
3. Reentrenar con 2 o más videos. Ahí `entrenar.py` ya agrupa por video solo y
   la nota pasa a ser honesta.
4. Escalamiento sigue siendo la conducta más floja. Solo hay 14 ejemplos claros
   en todo `IMG_0826`. Se arregla con más videos, no con más código.

## Cómo hablarle al usuario

Tiene TDAH y pidió explícitamente respuestas sin tecnicismos. Empieza por la
acción, numera los pasos, da tiempos concretos, nada de preámbulos ni de cierres
de cortesía. Listas de 5 como mucho. Termina siempre con una sola cosa que pueda
hacer ahora.

Y no le vendas números que no se hayan medido.
