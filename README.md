# panini-extractor

Extrae automáticamente cada sticker de un PDF del álbum Panini FIFA World Cup
2026 y lo guarda como un PNG independiente, usando únicamente visión por
computadora tradicional (OpenCV) — sin modelos de IA.

## Cómo funciona

1. **`pdf_loader.py`** renderiza cada página del PDF a una imagen (vía
   Poppler/`pdf2image`).
2. **`detector.py`** detecta automáticamente la cuadrícula de stickers. En un
   PDF real, el espacio entre stickers es de apenas 1 píxel a 300 DPI y no
   siempre es continuo (el propio arte de un sticker puede "sangrar" un par
   de píxeles hacia el hueco), así que un detector de contornos por sticker
   sobre el mapa de bordes completo no es confiable: el contenido interno de
   cada sticker (fotos, logos, texto) genera muchísimos más contornos que la
   propia cuadrícula. En su lugar, la detección se trata como un problema de
   *inferencia de cuadrícula*:
   1. Se calcula el área de contenido de la página (bounding box).
   2. Se muestrea la página en varias bandas perpendiculares buscando algunos
      huecos (gutters) inequívocamente blancos — con solo 2 o 3 confirmados
      alcanza, porque el espaciado de la cuadrícula es perfectamente uniforme.
   3. De ese espaciado se infiere cuántas filas/columnas hay, y se generan
      líneas de cuadrícula equiespaciadas a lo largo de toda el área de
      contenido — así se tolera que algún hueco puntual esté "sangrado" en
      alguna fila/columna, porque lo que se infiere es la cantidad de celdas,
      no la posición exacta de cada línea.
   4. Se recorta la página según esa cuadrícula, saltando cualquier celda
      esencialmente en blanco (un espacio vacío en una página parcialmente
      llena).

   No se asume ninguna posición, tamaño o cantidad de stickers: todo se
   deriva a partir del contenido real renderizado.

   La única excepción es *de qué página* sale ese conteo: como todas las
   páginas de un mismo álbum comparten la misma plantilla de cuadrícula,
   `extract.py` primero le pregunta a **cada** página cuántas filas/columnas
   detecta y con qué confianza (cuántos huecos realmente confirmó, no una
   estimación de respaldo), y usa el conteo que gana por mayoría entre las
   páginas más confiables para procesar el álbum completo — incluidas las
   pocas páginas donde el arte impreso sangra tanto que ni una sola fila u
   columna muestra un hueco limpio.
3. **`cropper.py`** recorta cada sticker con una transformación de
   perspectiva (corrige inclinaciones leves), añadiendo un margen
   configurable sin salirse de los límites de la página.
4. **`extract.py`** orquesta todo el proceso: carga el PDF, calcula la
   cuadrícula de consenso, numera los stickers de forma continua a través de
   todas las páginas y los guarda como `0001.png`, `0002.png`, ...

## Instalación

### 1. Python

Requiere Python 3.12.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Poppler (necesario para `pdf2image`)

`pdf2image` no convierte PDFs por sí mismo: delega en las utilidades de
línea de comandos de **Poppler** (`pdftoppm`/`pdfinfo`). Hay que instalar
Poppler y asegurarse de que esté en el `PATH`.

**Windows:**
1. Descarga los binarios de Poppler para Windows, por ejemplo desde
   https://github.com/oschwartz10612/poppler-windows/releases/ (elige el
   release más reciente).
2. Descomprime el `.zip` en, por ejemplo, `C:\poppler`.
3. Agrega `C:\poppler\Library\bin` a la variable de entorno `PATH`.
4. Verifica la instalación abriendo una nueva terminal y ejecutando:
   ```powershell
   pdftoppm -v
   ```

**macOS:**
```bash
brew install poppler
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install poppler-utils
```

## Uso

```bash
python extract.py album.pdf output/
```

Esto genera:

```
output/
    0001.png
    0002.png
    ...
```

Si el PDF tiene varias páginas, la numeración continúa sin reiniciarse en
cada página.

### Opciones de línea de comandos

| Flag       | Descripción                                             |
|------------|----------------------------------------------------------|
| `--dpi`    | Sobrescribe el DPI de renderizado (por defecto `300`)     |
| `--margin` | Sobrescribe el margen de recorte en píxeles                |
| `--debug`  | Genera overlays de depuración en `output/debug/`           |

Ejemplo con mayor resolución y modo debug:

```bash
python extract.py album.pdf output/ --dpi 600 --debug
```

## Modo DEBUG

Con `--debug`, por cada página se guarda una imagen en `output/debug/`
(`page_001.png`, `page_002.png`, ...) mostrando:

- la cuadrícula/bounding box de cada celda detectada (verde)
- el centro de cada sticker (azul)
- la numeración asignada a cada sticker (rojo)

Es la primera herramienta a revisar si algún sticker no se detecta
correctamente o el orden de numeración no es el esperado.

## Ajuste de parámetros

Todos los parámetros relevantes están centralizados en `config.py`
(`Config`), entre ellos:

- `DPI`, `MIN_CARD_WIDTH`, `MIN_CARD_HEIGHT`, `MARGIN`, `OUTPUT_FORMAT`,
  `DEBUG`
- `WHITE_THRESHOLD`: valor de gris a partir del cual un píxel se considera
  fondo de página en blanco (y no parte de un sticker).
- `NUM_PROFILE_BANDS` / `GUTTER_FRACTION_THRESHOLD`: cuántas bandas se
  muestrean y qué tan "limpio" (poco contenido) debe verse un hueco para
  confirmarlo como separación real entre stickers.
- `MAX_GRID_SIZE`: tope de seguridad para la cantidad de filas/columnas
  inferidas por página.
- `EMPTY_CELL_FILL_RATIO`: umbral para considerar una celda vacía (páginas
  parcialmente llenas).
- `MIN_FOOTER_GAP` / `FOOTER_GAP_THRESHOLD`: distinguen un hueco real entre
  stickers de un bloque de contenido ajeno a la cuadrícula (p. ej. una marca
  de agua o pie de página) que aparece después de la última fila/columna.

Si algunos stickers no se detectan correctamente, lo primero a intentar es:

1. Revisar `output/debug/` para entender qué está fallando (¿la cuadrícula
   se ve desalineada? ¿faltan celdas al final de una página?).
2. Si la página no rinde el número de filas/columnas esperado, bajar
   `GUTTER_FRACTION_THRESHOLD` (huecos menos "limpios") o revisar
   `WHITE_THRESHOLD` si el fondo de página no es blanco puro.
3. Ajustar `MIN_CARD_WIDTH`/`MIN_CARD_HEIGHT` si celdas válidas se están
   descartando por tamaño.

## Estructura del proyecto

```
panini-extractor/
    extract.py       # punto de entrada / CLI
    config.py         # parámetros centralizados
    pdf_loader.py      # PDF -> imágenes
    detector.py         # detección automática de stickers (OpenCV)
    cropper.py           # recorte + corrección de inclinación
    utils.py               # helpers compartidos (geometría, overlays de debug)
    output/                  # PNGs generados (y output/debug/ si --debug)
    samples/                  # PDFs de ejemplo para pruebas
    requirements.txt
    README.md
```

## Limitaciones conocidas

- El algoritmo asume que **todas las páginas del PDF comparten la misma
  cuadrícula** (mismo número de filas/columnas). Esto es válido para un
  álbum Panini estándar, pero si el PDF mezclara páginas con layouts
  genuinamente distintos (p. ej. una portada de texto sin stickers, o una
  página con una cuadrícula de otro tamaño), esa página se forzaría al
  mismo conteo que el resto y el resultado sería incorrecto para ella.
- Si ninguna página del PDF logra suficiente confianza en ningún eje (caso
  extremo, no observado en la práctica), no hay conteo de consenso al que
  recurrir y cada página vuelve a su propia estimación local, que puede
  fallar si su arte sangra a través de todos los huecos candidatos.
- Revisa `output/debug/` si sospechas que alguna página no siguió el
  conteo esperado.
