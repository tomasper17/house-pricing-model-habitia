"""
Rutas del proyecto. Es el único sitio donde se fijan: los módulos de `src/` y los notebooks
las importan de aquí, así que reorganizar `data/` solo obliga a tocar este fichero.

    data/
      raw/            fuentes externas tal como se descargan (no se modifican)
      interim/        dataset base (notebook 00) y selección de variables (notebook 03)
      processed/      matrices de modelado y objetivo (notebook 04)
      results/        registros de experimentos y modelos seleccionados (05, 06)
      models/         modelos entrenados (07) y paquete de producción exportado
      api_idealista/  respuestas de la API guardadas y predicciones (08)
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESADOS = DATA / "processed"
RESULTADOS = DATA / "results"
MODELOS = DATA / "models"
API_IDEALISTA = DATA / "api_idealista"
DIST = RAIZ / "dist"

# ------------------------------------------------------------------------------
# data/raw — fuentes
# ------------------------------------------------------------------------------
# idealista18 (Rey-Blanco et al., 2024): anuncios de venta de 2018, zonas y puntos de interés
DIR_IDEALISTA18 = RAW / "idealista18"
RUTA_MADRID_SALE = DIR_IDEALISTA18 / "Madrid_Sale.rda"
RUTA_MADRID_POLYGONS = DIR_IDEALISTA18 / "Madrid_Polygons.rda"
RUTA_MADRID_POIS = DIR_IDEALISTA18 / "Madrid_POIS.rda"

# INE: seccionado censal a 1 de enero de 2018
RUTA_SECCIONES_2018 = RAW / "ine_seccionado_2018" / "SECC_CE_20180101.shp"

# Ayuntamiento de Madrid: sección censal -> barrio/distrito (2026) y altas/bajas de secciones
RUTA_MAPEO_SECCIONES = RAW / "madrid_seccionado" / "Secciones_Censales_mappings.json"
RUTA_CAMBIOS_SECCIONES = RAW / "madrid_seccionado" / "secciones_mapeos_post_2016.xlsx"

# Ayuntamiento de Madrid: alquiler por sección censal (SERPAVI, una hoja por año)
RUTA_ALQUILER = RAW / "madrid_alquiler" / "alquiler_madrid_historico.xlsx"

# Ministerio de Vivienda y Agenda Urbana (SEIR): alquiler por distrito, dato oficial (no
# agregado desde secciones), una sola hoja con el último año disponible (2024)
RUTA_ALQUILER_DISTRITO = RAW / "madrid_alquiler" / "alquiler_madrid_historico_distritoi.xlsx"

# Ayuntamiento de Madrid: precio medio declarado de compraventa por distrito y barrio (registral)
RUTA_COMPRAVENTA = RAW / "madrid_compraventa" / "serie_compraventa_madrid.xlsx"

# Ayuntamiento de Madrid: incidencias de Policía Municipal, población y vulnerabilidad por barrio
DIR_INCIDENCIAS = RAW / "madrid_incidencias_2026"
RUTA_POBLACION = RAW / "madrid_poblacion" / "poblacion_distrito_barrio_2026.csv"
RUTA_VULNERABILIDAD = RAW / "madrid_vulnerabilidad" / "vulnerabilidad_barrios_2020.csv"

# datos.madrid.es: equipamientos (GeoRSS) para las distancias a puntos de interés
DIR_EQUIPAMIENTOS = RAW / "madrid_equipamientos"

# ------------------------------------------------------------------------------
# data/interim — salida del notebook 00
# ------------------------------------------------------------------------------
RUTA_DATASET_BASE = INTERIM / "habitia_madrid_2018.csv"
RUTA_DICCIONARIO = INTERIM / "diccionario_datos.csv"
RUTA_ASSET_DISTRITO = INTERIM / "asset_distrito.parquet"

# ------------------------------------------------------------------------------
# data/interim — salida del notebook 03 (selección de variables, para el 04)
# ------------------------------------------------------------------------------
RUTA_SELECCION_VARIABLES = INTERIM / "seleccion_variables_03.json"

# ------------------------------------------------------------------------------
# data/results — salidas de 05 y 06
# ------------------------------------------------------------------------------
RUTA_RESULTADOS_BASELINE = RESULTADOS / "resultados_baseline.csv"
RUTA_RESULTADOS_AJUSTE = RESULTADOS / "resultados_fine_tuning.csv"
RUTA_SELECCION_MODELOS = RESULTADOS / "modelos_seleccionados_06.json"

# ------------------------------------------------------------------------------
# data/models — salidas de 07 y del predictor exportable
# ------------------------------------------------------------------------------
RUTA_MODELO_GANADOR = MODELOS / "ganador_global.joblib"
RUTA_MODELO_PRODUCCION = MODELOS / "produccion.joblib"
DIR_PAQUETE_PRODUCCION = MODELOS / "paquete_produccion"
