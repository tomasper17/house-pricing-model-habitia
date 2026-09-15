"""
Imputación de `FLOORCLEAN` y `CADASTRALQUALITYID`, ajustada solo sobre train.

Reproduce la lógica del EDA original (sección 4, "Imputación de FLOORCLEAN y
CADASTRALQUALITYID"): mediana de `FLOORCLEAN` condicional a tipo de inmueble
(`BUILTTYPEID_1`/`BUILTTYPEID_2`) y tramo de altura del edificio
(`CADMAXBUILDINGFLOOR`, tratando el centinela `== 0` como altura desconocida), con
la mediana global como respaldo; moda para `CADASTRALQUALITYID`.

A diferencia del EDA (exploratorio, calculado sobre todo el dataset), aquí los
parámetros se ajustan con `fit_imputacion(df_train)` y se aplican igual a train y a
test con `aplicar_imputacion`, para no filtrar información de test al pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BINS_ALTURA = [-1, 2, 5, 8, np.inf]
LABELS_ALTURA = ["0-2", "3-5", "6-8", "9+"]


def _tramo_altura(cadmax: pd.Series) -> pd.Series:
    """CADMAXBUILDINGFLOOR == 0 es centinela de 'sin match catastral', no altura real."""
    cadmax_valido = cadmax.astype("float64").mask(cadmax == 0)
    return pd.cut(cadmax_valido, bins=BINS_ALTURA, labels=LABELS_ALTURA)


@dataclass
class ParametrosImputacion:
    medianas_floorclean: pd.Series  # index: (BUILTTYPEID_1, BUILTTYPEID_2, tramo_altura)
    mediana_floorclean_global: float
    moda_cadastralqualityid: float


def fit_imputacion(df_train: pd.DataFrame) -> ParametrosImputacion:
    floor = df_train["FLOORCLEAN"].astype("float64")
    tramo = _tramo_altura(df_train["CADMAXBUILDINGFLOOR"])
    grupo = [df_train["BUILTTYPEID_1"], df_train["BUILTTYPEID_2"], tramo]

    medianas = floor.groupby(grupo, observed=True).median()

    return ParametrosImputacion(
        medianas_floorclean=medianas,
        mediana_floorclean_global=float(floor.median()),
        moda_cadastralqualityid=float(df_train["CADASTRALQUALITYID"].mode().iloc[0]),
    )


def aplicar_imputacion(df: pd.DataFrame, params: ParametrosImputacion) -> pd.DataFrame:
    df = df.copy()

    tramo = _tramo_altura(df["CADMAXBUILDINGFLOOR"])
    grupo = pd.MultiIndex.from_arrays(
        [df["BUILTTYPEID_1"], df["BUILTTYPEID_2"], tramo]
    )
    medianas_por_fila = params.medianas_floorclean.reindex(grupo).to_numpy()

    floor = df["FLOORCLEAN"].astype("float64")
    floor_imputado = floor.fillna(pd.Series(medianas_por_fila, index=df.index))
    df["FLOORCLEAN"] = floor_imputado.fillna(params.mediana_floorclean_global)

    df["CADASTRALQUALITYID"] = df["CADASTRALQUALITYID"].fillna(
        params.moda_cadastralqualityid
    )

    return df
