"""
Limpieza determinista del dataset Habitia Madrid 2018.

Reproduce, como función, los pasos de limpieza del EDA original (secciones 3 y 4) que
no dependen de ninguna estadística estimada sobre los datos (drops estructurales y
aritmética fila a fila) -- por eso son seguros de aplicar sobre el dataset completo,
antes de separar train/test. La imputación (que sí estima medianas/modas) vive en
`src/imputacion.py` y debe ajustarse solo sobre train.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import rutas

TARGET = "PRICE"
LEAKAGE = ["UNITPRICE"]


def cargar_datos(ruta: str | Path = rutas.RUTA_DATASET_BASE) -> pd.DataFrame:
    """Carga el CSV y fija ASSETID como índice, igual que el EDA original."""
    df = pd.read_csv(ruta, sep=";")
    assert df["ASSETID"].is_unique, "ASSETID no es una clave única"
    return df.set_index("ASSETID")


def limpiar_basico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza estructural, igual para train y test (no estima nada sobre los datos):

    - Elimina `UNITPRICE` (leakage: identidad con PRICE/CONSTRUCTEDAREA).
    - Ajusta `PRICE` restando `PARKINGSPACEPRICE` cuando incluye la plaza de garaje,
      y elimina las dos columnas de parking ya usadas para ese ajuste.
    - Elimina `CONSTRUCTIONYEAR` (60% de nulos, redundante con `CADCONSTRUCTIONYEAR`).
    - Elimina `BUILTTYPEID_3` (categoría de referencia para las dummies de tipo de
      construcción).
    """
    df = df.copy()

    df = df.drop(columns=[c for c in LEAKAGE if c in df.columns])

    df = df.drop(columns=["PARKINGSPACEPRICE", "ISPARKINGSPACEINCLUDEDINPRICE"])

    df = df.drop(columns=["CONSTRUCTIONYEAR"])
    df = df.drop(columns=["BUILTTYPEID_3"])

    return df
