# Habitia — precio de venta de viviendas en Madrid

Trabajo de Fin de Máster (Máster en Ciencia de Datos, Big Data & IA, Universidad Complutense
de Madrid). Modelo predictivo del precio de venta de viviendas en Madrid ciudad, entrenado con
los anuncios de idealista de 2018 (`idealista18`) enriquecidos con fuentes públicas del
Ayuntamiento y del INE, y aplicado a anuncios actuales descargados de la API de idealista.

El recorrido completo — de los datos crudos al modelo en producción — está en `notebooks/`,
en orden. Toda la implementación vive en `src/`; los notebooks muestran la evidencia y las
decisiones.

## Estructura

```
.
├── data/
│   ├── raw/            fuentes externas, sin modificar (ver data/README.md)
│   ├── interim/        dataset base construido por el notebook 00
│   ├── processed/      matrices de modelado y objetivo (04)
│   ├── results/        registros de experimentos y modelos seleccionados (05, 06)
│   ├── models/         modelos entrenados (07) y paquete de producción exportado
│   └── api_idealista/  respuestas guardadas de la API y predicciones (08)
├── docs/               documentos de decisión (qué variables entran y cómo)
├── notebooks/          cadena de análisis, 00 → 08
├── src/                código del pipeline; src/rutas.py fija todas las rutas
└── requirements.txt
```

Todas las rutas se definen en [`src/rutas.py`](src/rutas.py). Los notebooks y los módulos las
importan de ahí (`from src import rutas`), de modo que ningún fichero lleva rutas escritas a mano.

## Instalación

Python 3.11 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate en Linux/macOS)
pip install -r requirements.txt
```

## Cadena de notebooks

Se leen y se ejecutan en orden. Cada notebook localiza la raíz del repositorio en su primera
celda, así que da igual desde qué carpeta se lance Jupyter.

| # | notebook | pregunta que responde | escribe en |
|---|---|---|---|
| 00 | [`00_construccion_dataset`](notebooks/00_construccion_dataset.ipynb) | ¿de dónde sale el dataset? cruce de `idealista18` con barrio, alquiler, equipamientos, criminalidad y vulnerabilidad | `data/interim/` |
| 01 | [`01_eda`](notebooks/01_eda.ipynb) | ¿qué hay en los datos y qué problemas tienen? | — |
| 02 | [`02_transformaciones`](notebooks/02_transformaciones.ipynb) | ¿cómo entra cada variable? | — |
| 03 | [`03_seleccion_variables`](notebooks/03_seleccion_variables.ipynb) | ¿qué variables entran en el modelo? | — |
| 04 | [`04_datasets_modelado`](notebooks/04_datasets_modelado.ipynb) | ejecución del plan: split train/test y variantes de matriz | `data/processed/` |
| 05 | [`05_modelado_baseline`](notebooks/05_modelado_baseline.ipynb) | ¿qué familia de modelo y qué variante de dataset? | `data/results/resultados_baseline.csv` |
| 06 | [`06_fine_tuning`](notebooks/06_fine_tuning.ipynb) | ¿qué hiperparámetros, para el modelo completo y para el de producción? | `data/results/resultados_fine_tuning.csv`, `modelos_seleccionados_06.json` |
| 07 | [`07_evaluacion_final`](notebooks/07_evaluacion_final.ipynb) | ¿cuánto aciertan en test los modelos elegidos? | `data/models/*.joblib` |
| 08 | [`08_produccion_api`](notebooks/08_produccion_api.ipynb) | ¿cuánto vale hoy un piso anunciado en idealista? | `data/api_idealista/` |

Todas las salidas intermedias están incluidas, así que cualquier notebook se puede abrir o
ejecutar sin haber ejecutado los anteriores.

- **06** tarda del orden de una hora (búsquedas de Random Forest y XGBoost sobre dos variantes).
- **08** no llama a la API por defecto: lee las respuestas guardadas en `data/api_idealista/`.
  Para descargar anuncios nuevos, pon `LLAMAR_API = True` en su celda de configuración y define
  las variables de entorno `IDEALISTA_API_KEY` e `IDEALISTA_API_SECRET`. Cada búsqueda consume
  cuota de la API.

**Por qué transformaciones (02) va antes que selección (03):** decidir qué variables entran
requiere conocer antes su forma funcional — `CADCONSTRUCTIONYEAR` en crudo parece irrelevante
(01 §8), pero binnada por épocas es una de las señales más fuertes del dataset.

**Modelo completo y modelo de producción.** Desde el 04 conviven dos variantes: `arboles`, con
todas las variables, y `arboles_desplegable`, sin las que no se pueden obtener para un anuncio
de la API (catastro, `CADDWELLINGCOUNT`, `ISINTOPFLOOR`). El 06 ajusta ambas con el mismo
esfuerzo, el 07 mide la diferencia en test y el 08 aplica la de producción.

## Código

| módulo | responsabilidad |
|---|---|
| `src/rutas.py` | todas las rutas del proyecto |
| `src/aux_functions.py` | mapeo sección censal → barrio y agregación de incidencias por barrio |
| `src/poi_distances.py` | distancias a equipamientos municipales (GeoRSS) |
| `src/crime_indices.py` | incidencias de policía por 10.000 habitantes, por barrio |
| `src/vulnerability_indices.py` | índice de vulnerabilidad por barrio |
| `src/eda_utils.py` | descriptivos, correlaciones, VIF, V de Cramer, outliers |
| `src/limpieza.py` | carga y limpieza determinista (segura antes del split) |
| `src/imputacion.py` | imputación de `FLOORCLEAN` y `CADASTRALQUALITYID` (ajustada en train) |
| `src/transformaciones.py` | formas funcionales; vías `lineal` y `arboles`; variables no disponibles en producción |
| `src/analisis_seleccion.py` | ΔR² único (bootstrap y validación cruzada), bloques, forward, redundancia, interacciones |
| `src/modelado.py` | validación cruzada, espacios de búsqueda con valores de apagado, registro de experimentos |
| `src/api_extractor.py` | cliente de la API de idealista (OAuth2, búsqueda paginada) |
| `src/amenidades_descripcion.py` | instalaciones desde la descripción del anuncio (mencionado / negado / sin mención) |
| `src/indices_precio.py` | índice de venta por distrito desde 2018 y factor precio → renta mensual |
| `src/produccion.py` | del anuncio de la API a la matriz del modelo, predicción y traslado de precio |
| `src/predictor.py` | predictor exportable, sin datos crudos (`python -m src.predictor`) |

## Predictor exportable

Desde la raíz del repositorio, con `data/models/produccion.joblib` ya generado por el 07:

```bash
python -m src.predictor exportar      # congela el paquete en data/models/paquete_produccion/
python -m src.predictor predecir data/api_idealista/madrid_ciudad_venta.json -o predicciones.csv
python -m src.predictor empaquetar    # dist/habitia_predictor.zip, autocontenido
```

## Documentos de decisión

- [`docs/plan_inclusion_variables.md`](docs/plan_inclusion_variables.md) — **qué** variables entran
- [`docs/plan_transformaciones.md`](docs/plan_transformaciones.md) — **cómo** entra cada una

Razonan por extenso lo que los notebooks 02 y 03 ejecutan. Sus cifras se calcularon antes de
que existiera `src/analisis_seleccion.py`; **las cifras válidas son las de los notebooks**,
que son las reproducibles.

## Datos

Fuentes, estructura de `data/` y diccionario del dataset base: [`data/README.md`](data/README.md).

Dataset principal: Rey-Blanco, D., Arbués, P., López, F. A. y Páez, A. (2024). *A geo-referenced
micro-data set of real estate listings for Spain's three largest cities*. Environment and
Planning B: Urban Analytics and City Science. Paquete R:
[paezha/idealista18](https://github.com/paezha/idealista18).
