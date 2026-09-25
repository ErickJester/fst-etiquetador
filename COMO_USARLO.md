# Etiquetador automático de FST

Sirve para convertir muchos videos en etiquetas, y con esas etiquetas entrenar
el clasificador final.

Tres archivos:

| Archivo o carpeta | Para qué |
|---|---|
| `videos_sin_etiquetar/` | **Aquí dejas tus videos crudos** |
| `etiquetas/` | Aquí salen los resultados |
| `modelo_fst.joblib` | El cerebro entrenado |
| `etiquetar.py` | Video entra, CSV sale |
| `entrenar.py` | CSVs etiquetados entran, modelo sale |

`lib.py` son las tripas. No lo tocas.

---

## Qué cambia respecto al pipeline anterior

**1. Encuentra los tubos y la línea del agua solo.**
No hay números escritos a mano. Cada video se mide a sí mismo.

**2. Mide en largos de rata, no en píxeles.**
"Se movió 3 veces su propio largo" significa lo mismo con la cámara cerca o
lejos. Sin esto, el modelo aprende el encuadre en lugar de la conducta.

**3. Cuenta el tiempo en segundos, no en fotogramas.**
Así aguanta videos grabados a distinta velocidad.

---

## Empezar desde cero

### Paso 1 — sacar los rasgos del video que ya etiquetaste

```bash
py etiquetar.py ../IMG_0826.MOV --tubos 4 --salida etiquetas
```

Te deja dos archivos:
- `IMG_0826_rasgos.csv` — los números
- `IMG_0826_control.jpg` — **ábrela**

En la imagen de control, la línea azul tiene que caer sobre el agua y la verde
sobre el suelo del tubo. Si no, ese video no sirve y no hay que insistir.

### Paso 2 — entrenar con él

Pon tu puntuación manual al lado, llamada `IMG_0826_mano.csv`, con las columnas
`bloque`, `especimen`, `clase` y (si la tienes) `confianza`.

```bash
py entrenar.py --carpeta etiquetas --etiquetas mano
```

Imprime la nota real y guarda `modelo_fst.joblib`.

### Paso 3 — etiquetar los videos nuevos

Déjalos en `videos_sin_etiquetar/` y lánzalos todos de una:

```bash
py etiquetar.py videos_sin_etiquetar/*.MOV --modelo modelo_fst.joblib --tubos 4
```

Si son `.mp4`, cambia esa parte. Si uno falla, los demás siguen y al final te
dice cuáles fallaron.

Cada CSV trae dos columnas extra:
- `confianza` — qué tan seguro está, de 0 a 1
- `usar` — vale 1 solo si pasó el umbral (0.80 por defecto)

### Paso 4 — bola de nieve

No etiquetes a mano el video entero. Revisa **nada más los bloques donde la
máquina dudó**, que son como uno de cada cinco.

Para eso está **`revisor.html`**. Ábrelo con doble clic (Chrome o Edge) y
arrastra encima el video y su `_rasgos.csv`. Es como un juego:

- El video ocupa casi toda la pantalla, con el tubo a mirar iluminado
- Debajo, la línea de tiempo de cada tubo, como en un editor de video
- Debajo, cinco botones:

| Botón | Tecla | Qué hace |
|---|---|---|
| Nado | `1` | Esa es la conducta |
| Inmovilidad | `2` | Esa es la conducta |
| Escalamiento | `3` | Esa es la conducta |
| Repetir | `R` | Vuelve a ver esos 5 segundos |
| Conducta activa | `4` | Se mueve, pero no sabes si nada o trepa |

**Conducta activa** es el nivel de Porsolt: no inmóvil, o sea nado más
escalamiento. Úsala cuando está claro que la rata no está quieta pero no
distingues si nada o trepa. Es una respuesta válida, no un "no sé".

Si no lo ves claro ni repitiendo, pulsa `0`: el bloque queda como dudoso y no
entra al entrenamiento.

Para moverte: una flecha `→` va al bloque de al lado; dos flechas rápidas
`→→` saltan al siguiente dudoso. Clic en cualquier bloque de la línea de tiempo
para ir a él. El progreso se guarda solo en el navegador; si cierras, al volver
te ofrece continuar.

Al empezar eliges entre tres modos:

| Modo | Qué revisas |
|---|---|
| Solo donde la máquina dudó | Los bloques dudosos, viendo su propuesta |
| Todo el video | Todos, viendo su propuesta |
| Desde cero | Todos, **sin ver** lo que dice la máquina |

"Desde cero" es el que da la comparación honesta entre tú y la máquina: al
final te dice en qué porcentaje coincidiste sin que ella te influyera. Lleva su
propio progreso, separado de los otros dos.

Al terminar, "Guardar la hoja" te propone el nombre `video2_mano.csv`.
Guárdalo en `etiquetas/` y vuelve al Paso 2 con los dos videos:

```bash
py entrenar.py --carpeta etiquetas --etiquetas mano
```

Cada video nuevo te cuesta menos que el anterior. El primero: una hora. El
quinto: quince minutos.

---

## Cosas que te van a morder

**Las etiquetas dudosas se tiran por defecto.** Si tu CSV manual tiene una
columna `confianza` con valores "baja", esas filas no entran al entrenamiento.
Es a propósito: una etiqueta en la que tú dudaste no le enseña nada al modelo.
Si las quieres dentro, agrega `--usar-dudosas`.

**Pon siempre `--tubos N`.** Sin eso, si un video tiene un reflejo raro, puede
creer que hay 5 tubos y desordenar la numeración de los especímenes.

**La numeración de los tubos es de izquierda a derecha.** Si en un video las
jaulas están en otro orden, las etiquetas se cruzan. Revísalo en la imagen de
control.

**Si solo tienes un video, la nota sale optimista** y el programa te lo avisa.
Con dos o más videos valida por video, que es lo honesto.

**Velocidad.** Unos 10 a 15 minutos por video de 5 minutos. Con `--paso 4` baja
a la mitad, midiendo 7 veces por segundo en vez de 15. Para inmovilidad y nado
alcanza; para escalamiento pierde detalle.
