"""
Transformaciones de variables, según `docs/plan_transformaciones.md`.

La mayoría de los cortes son umbrales fijos de dominio (nudo de superficie en 50 m²,
épocas constructivas, tramos de habitaciones/planta/altura) y no dependen de la
distribución de los datos, así que se aplican igual a train y test sin ajuste previo.
Los dos únicos elementos que sí se estiman sobre los datos -- los quintiles de
`delitos_per_10k` y la media/desviación para estandarizar -- se ajustan solo sobre
train con `fit_transformaciones` y se aplican con `aplicar_transformaciones` para
evitar fuga de información de test (ver plan_transformaciones.md §8.1).

Expone `transformar_variables(df, familia, params)` con dos vías (plan §7):

- `familia="lineal"`: aplica todas las transformaciones y devuelve variables listas
  para dummificar/estandarizar.
- `familia="arboles"`: devuelve las variables crudas (las transformaciones monótonas
  son inertes para un árbol); solo se retira lo que también se descarta en la vía de
  árboles (las 4 distancias a POI).

Los grupos de columnas al final del módulo (`COLS_INCLUIR`, `COLS_CAD`) permiten
construir el conjunto final de variables y su variante desplegable (sin campos
catastrales) por selección de columnas sobre el resultado de `transformar_variables`,
sin tener que materializar cada combinación por separado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POI_DESCARTAR = [
    "dist_parque_m",
    "dist_centro_medico_m",
    "dist_centro_educativo_m",
    "dist_espacio_deporte_m",
]

ORIENTACIONES_DESCARTAR = [
    "HASNORTHORIENTATION",
    "HASSOUTHORIENTATION",
    "HASEASTORIENTATION",
    "HASWESTORIENTATION",
]

_COLS_ESTANDARIZAR = ["alq_mediana_eur_m2_barrio", "indice_vulnerabilidad"]


# ==============================================================================
# Ajuste (fit) de los elementos que dependen de los datos -- solo sobre train
# ==============================================================================
@dataclass
class ParametrosTransformacion:
    bordes_delitos: np.ndarray  # bordes de pd.qcut ajustados en train
    medias: dict[str, float]
    desviaciones: dict[str, float]


def fit_transformaciones(df_train: pd.DataFrame) -> ParametrosTransformacion:
    _, bordes = pd.qcut(df_train["delitos_per_10k_barrio"], 5, retbins=True, duplicates="drop")
    bordes = bordes.copy()
    bordes[0], bordes[-1] = -np.inf, np.inf  # cubrir valores de test fuera del rango de train

    medias = {c: float(df_train[c].mean()) for c in _COLS_ESTANDARIZAR}
    desviaciones = {c: float(df_train[c].std(ddof=0)) for c in _COLS_ESTANDARIZAR}

    return ParametrosTransformacion(bordes_delitos=bordes, medias=medias, desviaciones=desviaciones)


# ==============================================================================
# Transformaciones de umbral fijo (sin ajuste, plan_transformaciones.md §4)
# ==============================================================================
def _transformar_area(df: pd.DataFrame) -> pd.DataFrame:
    df["log_area"] = np.log(df["CONSTRUCTEDAREA"].astype("float64"))
    df["log_area_50"] = np.maximum(df["log_area"] - np.log(50), 0)
    return df.drop(columns=["CONSTRUCTEDAREA"])


def _transformar_distancias_centro(df: pd.DataFrame) -> pd.DataFrame:
    df["log_dist_centro"] = np.log1p(df["DISTANCE_TO_CITY_CENTER"])
    df["log_dist_castellana"] = np.log1p(df["DISTANCE_TO_CASTELLANA"])
    df["log_dist_metro"] = np.log1p(df["DISTANCE_TO_METRO"])
    return df.drop(columns=["DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_CASTELLANA", "DISTANCE_TO_METRO"])


def _transformar_viviendas_edificio(df: pd.DataFrame) -> pd.DataFrame:
    df["log_viviendas_edif"] = np.log1p(df["CADDWELLINGCOUNT"].astype("float64"))
    return df.drop(columns=["CADDWELLINGCOUNT"])


def _reordenar_con_referencia(s: pd.Series, referencia) -> pd.Series:
    """Pone `referencia` primera en el orden de categorías, para que get_dummies(...,
    drop_first=True) la use siempre como referencia -- igual en train y en test,
    porque el orden de categorías es fijo por construcción (pd.cut), no observado."""
    resto = [c for c in s.cat.categories if c != referencia]
    return s.cat.reorder_categories([referencia] + resto)


def _transformar_habitaciones(df: pd.DataFrame) -> pd.DataFrame:
    cat = pd.cut(
        df["ROOMNUMBER"].astype("float64"),
        bins=[-1, 0, 1, 2, 3, 4, np.inf],
        labels=["0", "1", "2", "3", "4", "5+"],
    )
    df["habitaciones_cat"] = _reordenar_con_referencia(cat, "0")
    return df.drop(columns=["ROOMNUMBER"])


def _transformar_epoca(df: pd.DataFrame) -> pd.DataFrame:
    cat = pd.cut(
        df["CADCONSTRUCTIONYEAR"].astype("float64"),
        bins=[0, 1900, 1940, 1960, 1975, 1990, 2000, 2008, 2100],
        labels=["<1900", "1900-40", "1940-60", "1960-75", "1975-90", "1990-2000", "2000-08", ">=2008"],
    )
    # Referencia = "1960-75" (el mínimo de precio): así todos los coeficientes de
    # época salen positivos y se leen directamente como prima sobre el peor tramo.
    df["epoca_construccion"] = _reordenar_con_referencia(cat, "1960-75")
    return df.drop(columns=["CADCONSTRUCTIONYEAR"])


def _transformar_planta(df: pd.DataFrame) -> pd.DataFrame:
    floor = df["FLOORCLEAN"].astype("float64")
    df["es_sotano_bajo"] = (floor <= 0).astype(int)
    cat = pd.cut(floor, bins=[-2, 0, 4, 8, np.inf], labels=["sotano/bajo", "1-4", "5-8", "9+"])
    df["planta_cat"] = _reordenar_con_referencia(cat, "sotano/bajo")
    return df.drop(columns=["FLOORCLEAN"])


def _transformar_altura_edificio(df: pd.DataFrame) -> pd.DataFrame:
    cadmax = df["CADMAXBUILDINGFLOOR"].astype("float64")
    df["sin_match_catastral"] = (cadmax == 0).astype(int)
    cadmax_valido = cadmax.mask(cadmax == 0)
    cat = pd.cut(cadmax_valido, bins=[0, 3, 6, 9, np.inf], labels=["1-3", "4-6", "7-9", "10+"])
    df["alturaedif_cat"] = _reordenar_con_referencia(cat, "1-3")
    return df.drop(columns=["CADMAXBUILDINGFLOOR"])


def _tope_banos(df: pd.DataFrame, tope: int = 5) -> pd.DataFrame:
    df["BATHNUMBER"] = df["BATHNUMBER"].astype("float64").clip(upper=tope)
    return df


def _transformar_delitos(df: pd.DataFrame, params: ParametrosTransformacion) -> pd.DataFrame:
    cat = pd.cut(
        df["delitos_per_10k_barrio"], bins=params.bordes_delitos, labels=[f"Q{i}" for i in range(1, 6)]
    )
    df["delitos_q"] = _reordenar_con_referencia(cat, "Q1")
    return df.drop(columns=["delitos_per_10k_barrio"])


# Columnas categóricas creadas por las transformaciones de arriba: sus categorías
# (incluida la referencia) son fijas por construcción, así que get_dummies produce
# las mismas columnas en train y test sin necesidad de un fit adicional.
CATEGORICAS_NUEVAS = ["habitaciones_cat", "epoca_construccion", "planta_cat", "alturaedif_cat", "delitos_q"]


def dummificar(df: pd.DataFrame) -> pd.DataFrame:
    """get_dummies(drop_first=True) sobre las categóricas nuevas presentes en df."""
    columnas = [c for c in CATEGORICAS_NUEVAS if c in df.columns]
    return pd.get_dummies(df, columns=columnas, drop_first=True)


def _estandarizar(df: pd.DataFrame, params: ParametrosTransformacion) -> pd.DataFrame:
    for c in _COLS_ESTANDARIZAR:
        df[c] = (df[c] - params.medias[c]) / params.desviaciones[c]
    return df


def _transformar_poi(df: pd.DataFrame) -> pd.DataFrame:
    """
    log1p de las 4 distancias a POI, en igualdad de condiciones con el resto de
    distancias del modelo.

    `transformar_variables` retira estas 4 columnas por defecto (`POI_DESCARTAR`): es una
    decisión estructural, no de ajuste (ruido de anonimización por debajo de su
    escala, notebook 03 §4.1), pero sigue siendo una decisión tomada mirando un OLS
    aditivo. Esta función existe para que `retener_poi_orientacion=True` pueda
    devolverlas, en la misma forma funcional que el resto de la vía lineal, a
    cualquier modelo que quiera confirmarla o contradecirla con su propio criterio.
    """
    df = df.copy()
    for c in POI_DESCARTAR:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].astype("float64"))
            df = df.drop(columns=[c])
    return df


# ==============================================================================
# Punto de entrada
# ==============================================================================
def transformar_variables(
    df: pd.DataFrame,
    familia: str,
    params: ParametrosTransformacion,
    estandarizar: bool = False,
    retener_poi_orientacion: bool = False,
) -> pd.DataFrame:
    """
    familia: "lineal" | "arboles". `df` debe venir ya limpio (`src.limpieza`) e
    imputado (`src.imputacion`). No incluye el target ni lo dummifica.

    `retener_poi_orientacion=True` desactiva el descarte estructural por defecto de
    las 4 distancias a POI y las 4 orientaciones (notebook 03 §4.1/§4.3) y las deja en
    la matriz. Existe para que un modelo que no comparte los supuestos de un OLS
    aditivo -- un árbol, un GBM, una selección stepwise -- pueda confirmar o
    contradecir esa decisión con su propio criterio en vez de heredarla sin poder
    cuestionarla (notebook 03 §8): con "lineal" las distancias entran en log1p, igual
    que el resto; con "arboles" van crudas, como todo lo demás en esa vía. Por
    defecto (`False`) el comportamiento es idéntico al de siempre.
    """
    if familia not in {"lineal", "arboles"}:
        raise ValueError(f"familia debe ser 'lineal' o 'arboles', recibido: {familia!r}")

    if retener_poi_orientacion:
        if familia == "lineal":
            df = _transformar_poi(df)
        # familia == "arboles": no se transforma nada, van crudas -- igual que el
        # resto de columnas en esa vía.
    else:
        df = df.drop(columns=POI_DESCARTAR + ORIENTACIONES_DESCARTAR)

    if familia == "arboles":
        return df

    df = _transformar_area(df)
    df = _transformar_distancias_centro(df)
    df = _transformar_viviendas_edificio(df)
    df = _transformar_habitaciones(df)
    df = _transformar_epoca(df)
    df = _transformar_planta(df)
    df = _transformar_altura_edificio(df)
    df = _tope_banos(df)
    df = _transformar_delitos(df, params)

    if estandarizar:
        df = _estandarizar(df, params)

    return df



# Variables no disponibles en producción (API de idealista): CAD* + tipología constructiva,
# que exigen el enriquecimiento catastral (plan_inclusion_variables.md §4.5), e
# ISINTOPFLOOR, que exige conocer la altura del edificio. Las instalaciones no están aquí: se
# reconstruyen desde la descripción del anuncio (`src/amenidades_descripcion.py`).
# Quitar estas columnas del resultado de transformar_variables da la variante "desplegable";
# dejarlas da la "enriquecida". Este eje (disponibilidad en producción) es independiente
# de la decisión de inclusión de arriba.
COLS_CAD = [
    "epoca_construccion", "alturaedif_cat", "sin_match_catastral", "log_viviendas_edif",
    "CADASTRALQUALITYID", "BUILTTYPEID_1", "BUILTTYPEID_2",
    "ISINTOPFLOOR",
]

# Equivalente crudo de COLS_CAD sobre la vía de árboles: mismo eje (disponibilidad en
# producción), sobre las variables sin transformar. `epoca_construccion` es
# `CADCONSTRUCTIONYEAR` sin binnear; `alturaedif_cat`/`sin_match_catastral` son las dos
# lecturas (categoría y flag de no-match) de la misma `CADMAXBUILDINGFLOOR` cruda, así
# que ahí basta un solo nombre. `CADASTRALQUALITYID`/`BUILTTYPEID_1`/`BUILTTYPEID_2` no
# cambian de nombre entre vías.
COLS_CAD_ARBOLES = [
    "CADCONSTRUCTIONYEAR", "CADMAXBUILDINGFLOOR", "CADDWELLINGCOUNT",
    "CADASTRALQUALITYID", "BUILTTYPEID_1", "BUILTTYPEID_2",
    "ISINTOPFLOOR",
]
