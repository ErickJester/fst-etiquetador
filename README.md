# Etiquetador automático del test de nado forzado (FST)

Convierte videos del test de nado forzado en etiquetas de conducta, para
construir el conjunto de entrenamiento de un clasificador.

Cada video tiene varias ratas, cada una en su propio cilindro. El programa
divide el video en bloques de 5 segundos y asigna a cada bloque y cada animal
una de tres conductas:

| Conducta | Definición operativa |
|---|---|
| **Inmovilidad** | Flota pasiva. Solo los movimientos necesarios para mantener el hocico fuera del agua. No cambia de cuadrante. |
| **Nado** | Se desplaza cruzando al menos 2 de los 4 cuadrantes del cilindro. Bucear cuenta como nado. |
| **Escalamiento** | Movimientos rápidos de las patas delanteras contra la pared, rompiendo la superficie del agua. |

Basado en el esquema de Detke, Rickels y Lucki (1995), con las definiciones
operativas de Craft, Kostick, Rogers y Tsutsui (2010) y la validación de
intervalos de Álvarez-Suárez et al. (2015).

## Qué lo distingue

**No hay medidas escritas a mano.** El programa deduce de cada video dónde
están las paredes de los cilindros, la línea del agua y el suelo. Eso permite
usarlo con grabaciones de distintos días, encuadres y resoluciones.

**Las distancias se miden en largos de cuerpo, no en píxeles.** Un animal
grabado de cerca y otro de lejos producen el mismo número. Sin esta
normalización el clasificador aprende el encuadre en lugar de la conducta.

**Cada etiqueta lleva su confianza.** La columna `usar` marca con 1 solo los
bloques donde el modelo decidió con seguridad. Filtrar por ella deja menos
etiquetas pero considerablemente más limpias.

## Instalación en una máquina nueva

Requiere Python 3.9 o superior. **No requiere GPU** ni ninguna tarjeta
especial: todo corre en procesador.

```bash
git clone https://github.com/ErickJester/fst-etiquetador.git
cd fst-etiquetador
pip install -r requirements.txt
py comprobar.py
```

`comprobar.py` tarda un minuto y verifica que esté todo: versión de Python,
librerías, archivos del proyecto, que el modelo cargue con sus columnas, y que
los videos que hayas dejado en `videos_sin_etiquetar/` se puedan leer completos.
Si algo falta, dice exactamente qué comando lo arregla.

Córrelo antes de procesar nada. Detecta en un minuto problemas que si no
descubrirías a los diez minutos de estar procesando.

En Linux o macOS usa `python3` en lugar de `py` en todos los comandos.

### Sobre la versión de scikit-learn

`modelo_fst.joblib` se guardó con scikit-learn 1.9.1. Cargarlo con otra serie
funciona pero emite un aviso y puede cambiar las predicciones. `comprobar.py`
lo detecta. La solución es reentrenar, que toma segundos:

```bash
py entrenar.py --carpeta etiquetas --etiquetas mano
```

## Uso

Etiquetar videos:

```bash
py etiquetar.py videos_sin_etiquetar/*.MOV --modelo=modelo_fst.joblib --tubos=4
```

Reentrenar con etiquetas corregidas:

```bash
py entrenar.py --carpeta etiquetas --etiquetas mano
```

Instrucciones completas en [COMO_USARLO.md](COMO_USARLO.md).

## Flujo de trabajo

El etiquetado crece en bola de nieve. No hace falta puntuar cada video entero
a mano:

1. Puntúa un video completo a mano. Es la semilla.
2. Entrena con él.
3. Pasa el modelo al video siguiente. Revisa **solo** las filas con `usar = 0`,
   alrededor de una de cada cinco.
4. Guarda las correcciones y vuelve al paso 2 con los dos videos.

El primer video cuesta cerca de una hora. El quinto, unos quince minutos.

## Estado actual

Medido sobre `IMG_0826`, validando por animal, contra las 100 etiquetas de
confianza alta y media del anotador humano:

| Métrica | Valor |
|---|---|
| Exactitud | 0.890 |
| Kappa de Cohen | +0.819 |
| f1 inmovilidad | 0.935 |
| f1 nado | 0.884 |
| f1 escalamiento | 0.786 |
| Exactitud filtrando a confianza ≥ 0.80 | 0.929 (conserva el 85%) |

**Estas cifras son optimistas.** Provienen de un solo video, de modo que la
validación solo puede agrupar por animal. El sistema todavía no se ha evaluado
sobre un video que no haya visto: hasta que eso ocurra, no hay evidencia de que
generalice.

El escalamiento es la conducta más débil. En `IMG_0826` hay apenas 14 ejemplos
claros, insuficientes para aprender el patrón. Se espera que mejore al añadir
videos, no al cambiar el código.

## Archivos

| Archivo | Contenido |
|---|---|
| `comprobar.py` | Verifica que la máquina pueda correr todo. Se ejecuta primero |
| `revisor.html` | Página para revisar las propuestas como un juego: video, línea de tiempo y cinco botones. Se abre con doble clic |
| `lib.py` | Registro de cámara, modelo de fondo, geometría automática, extracción de rasgos |
| `etiquetar.py` | Video → CSV de rasgos y etiquetas |
| `entrenar.py` | CSV etiquetados → modelo, con validación agrupada por video |
| `modelo_fst.joblib` | Modelo entrenado, junto con los nombres y el orden de sus columnas |
| `etiquetas/` | Rasgos extraídos y puntuación manual de referencia |

## Verificación obligatoria

Cada ejecución deja un `*_control.jpg` con la geometría detectada dibujada
sobre un fotograma. **Revísalo antes de usar las etiquetas de ese video.** Si
la línea azul no cae sobre la superficie del agua o los recuadros no cubren los
cilindros completos, la detección falló y esas etiquetas no sirven.

Un recuadro que recortaba el cilindro del extremo costó 24 puntos de f1 en
escalamiento, porque el animal trepa pegado a la pared exterior: justo el trozo
que quedaba fuera. Ese error solo era visible en la imagen de control.
