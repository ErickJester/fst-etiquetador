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
| `comprobar.py` | Diagnóstico del entorno. Lo primero que se corre en una máquina nueva |
| `revisor.html` | Revisión humana en el navegador. Lee `*_rasgos.csv` + video, escribe `*_mano.csv`. Sustituye a `revisar.py` |
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

## El parametro --paso: medido, no supuesto

Mas detalle NO es mejor. Sobre `IMG_0826`, mismos bloques y mismas etiquetas:

| | paso 12 | paso 2 |
|---|---|---|
| Exactitud | **0.880** | 0.770 |
| f1 escalamiento | **0.786** | 0.476 |
| Filtrado a 0.80 | **0.929** | 0.857 |

Causa: `path` y `rng` acumulan el temblor del centroide. Con 75 muestras por
bloque en vez de 12, ese temblor se suma seis veces mas. `path` sale 2.06 veces
mayor a paso 2, y la rata no anduvo el doble.

**Usar siempre `--paso=12`.** El modelo guarda el paso con el que se entreno y
`etiquetar.py` se niega a correr con otro, antes de procesar nada.

Rasgos que SI son independientes del paso tras el arreglo de `DT_REF`
(razon entre paso 2 y paso 12): `me` 1.00, `iou` 1.00, `dice` 1.00,
`vert` 1.00, `area` 1.00, `elong` 1.00.

Rasgos que siguen dependiendo: `path` 2.06, `rng` 1.29, `spanx` 1.23, `nq` 1.11.
Se podrian arreglar remuestreando la trayectoria a una rejilla temporal fija
antes de medirla. No se hizo porque paso 12 ya es el mejor y el candado impide
mezclar.

## revisor.html: lo que hay que saber antes de tocarlo

- Todo el analisis vive en el marco ESTABILIZADO (`gx0`, `gx1`, `g_agua`,
  `g_fondo`). El video que ve el usuario es el crudo. Para dibujar sobre el hay
  que usar `bx0..by1`, que `lib.cajas_crudas` calcula deshaciendo el registro
  en la mitad de cada bloque. En `IMG_0840` la camara se desplaza hasta 574 px
  en vertical: con `gx*` el marco cae fuera de la rata.
- El oscurecido alrededor del tubo son cuatro divs. Un box-shadow u outline de
  9999 px deja de pintarse en Chrome mientras el `<video>` se reproduce.
- La hoja que guarda trae `origen`: `humano_confirma`, `humano_corrige`,
  `humano_duda` (casilla vacia) o `maquina` (bloque seguro que nadie miro).
- `entrenar.py` tira `clase/confianza/usar` de los rasgos antes de cruzar. Si
  no, pandas renombra a `clase_x/clase_y` y la etiqueta humana se pierde.
- Con pandas 3.0, `astype(str)` conserva los vacios como nulos. Hay que hacer
  `fillna('')` antes, o el "no se" del revisor rompe el entrenamiento.

Pendiente conocido: al validar por video, las filas `maquina` del video de
prueba son predicciones del propio modelo y inflan la nota. La cifra honesta
es la calculada solo sobre filas `humano_*`.

## Conducta activa

Cuarta etiqueta valida: `activa` = no inmovil, sin decidir entre nado y
escalamiento. Es el nivel de Porsolt (inmovil / no inmovil); Detke separo
despues lo activo en nado y escalamiento. OJO: el usuario la describio una vez
como "nado e inmovilidad"; se implemento como nado + escalamiento porque es lo
unico coherente con Porsolt, y se le aviso.

- `etiquetar.py` propone `activa` cuando ninguna conducta pasa el umbral pero
  p_nado + p_escalamiento si. En IMG_0840 bajo los dudosos de 23 a 7.
  Guarda tambien `p_inmovilidad`, `p_nado`, `p_escalamiento`.
- `etiquetar.py` acepta un `_rasgos.csv` en lugar de un video para volver a
  proponer con otro modelo sin reprocesar (segundos, no diez minutos).
- `entrenar.py` saca las filas `activa` del modelo de tres conductas y las usa
  en una segunda medida, inmovil contra activa, validada igual por grupos.
- El `--paso` por defecto de `etiquetar.py` es 12 (antes 2, que contradecia
  lo medido y chocaba con el candado del modelo).

`revisor.html`: teclas 1 nado, 2 inmovilidad, 3 escalamiento, 4 activa. Enter
ya no hace nada durante la revision. Una flecha = bloque de al lado; dos
flechas en menos de 400 ms = siguiente dudoso contando desde donde estabas.
Modo "desde cero" oculta todo rastro de la maquina (cartel, colores de la linea
de tiempo, contadores) y guarda su progreso aparte; sus filas salen con
`origen = humano_ciego` y los bloques sin mirar quedan en blanco.

## CNN (carpeta cnn/)

La tesis pide una CNN. El etiquetador (Random Forest sobre rasgos) fabrico el
conjunto de datos; la CNN es el clasificador final y se compara contra el.

- `cnn/recortar.py` corre donde estan los videos (CPU, ~1 min por video). Por
  cada bloque de `*_mano.csv` recorta el tubo con `bx0..by1` del `_rasgos.csv`,
  8 fotogramas repartidos en los 5 s, gris 128x96, enderezado segun `rotar`.
  Deja `clips/X.npz` + `X_muestra.jpg` para revisar a ojo.
- `cnn/fst_cnn.py`: ResNet18 preentrenada; los 3 canales son 3 fotogramas del
  bloque (asi ve movimiento). Trio al azar al entrenar, promedio de trios fijos
  al predecir. Validacion GroupKFold por video, misma salida que entrenar.py.
- `cnn/fst_cnn.ipynb`: cuaderno de Colab. El repo es PRIVADO: no se clona;
  lee `MyDrive/fst/fst_cnn.py` y `MyDrive/fst/clips` desde Drive.
- Probado en local (CPU) con 3 videos, 8 epocas: kappa +0.24. Solo prueba que
  el circuito funciona; con 2 videos de entrenamiento por ronda no dice nada.
- Colab, 14 videos, validando por video: entrada 'trios' (3 fotogramas crudos)
  kappa +0.476 a 12 epocas y +0.430 a 40. No era falta de entrenamiento: la
  red aprendia el fondo de cada video.
- Entrada 'movimiento' (postura normalizada + desviacion temporal + cambio
  entre fotogramas): en local, 3 videos, 8 epocas, kappa +0.24 -> +0.41 e
  inmovilidad 0.06 -> 0.58 frente a 'trios' en el mismo experimento.
- Colab, 14 videos: 'movimiento' kappa +0.574 (8 fotogramas) y +0.580 (16).
  Muestrear mas denso no ayudo.
- Version 4 (`VERSION` se imprime al cargar, para saber que Colab lee la
  copia nueva de Drive): ENTRADA 'video3d', R(2+1)D-18 de torchvision
  preentrenada en Kinetics-400, 16 fotogramas, 15 epocas, lote 16, lr 1e-4.
  Cada clip normalizado por su propio brillo. En CPU ~1 s por clip y epoca.
- Colab v4: kappa +0.625, Porsolt 0.896 (kappa +0.703). La mejor hasta ahora.
- `cnn/probar.py X_rasgos.csv --red fst_cnn.pt`: aplica la red en la PC sin
  GPU (~2-4 min por video) y deja `X_cnn.csv` para abrir en revisor.html. Si
  hay `X_mano.csv` compara y avisa si el video estaba en el entrenamiento.
  Ojo: revisor guarda como `X_mano.csv` y pisaria el existente.
