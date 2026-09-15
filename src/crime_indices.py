"""
Carga y procesamiento de índices de criminalidad por barrio.

Módulo para cargar incidencias de la Policía Municipal de Madrid,
agregar por barrio, ponderar por población, y unir al dataset de anuncios.
"""

import re
import unicodedata
from pathlib import Path
from typing import Tuple
import pandas as pd
import geopandas as gpd
from src.aux_functions import crear_indice_criminalidad


def _normalizar(texto: str) -> str:
    """Normaliza texto: uppercase, sin acentos, sin puntuación, para matching.

    Quitar la puntuación es necesario porque el mismo barrio se escribe de
    forma distinta en cada fuente (p.ej. "Villaverde Alto, Casco Histórico de
    Villaverde" vs "VILLAVERDE ALTO - CASCO HISTORICO DE VILLAVERDE").
    """
    if pd.isna(texto):
        return None
    texto = str(texto).upper()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r"[^A-Z0-9]+", " ", texto).strip()
    return texto


def cargar_poblacion_por_barrio(ruta: str, ano: int = 2026) -> pd.DataFrame:
    """
    Carga población por código de barrio desde CSV del Ayuntamiento.

    El fichero trae una fila por barrio y año (2018-2026); se toma solo un
    año para no sumar la misma población varias veces. `num_personas` usa
    "." como separador de miles, no decimal, así que hay que quitarlo antes
    de convertir a número (si no, pandas lee "23.410" como 23.41).

    Args:
        ruta: Ruta al CSV de población (poblacion_distrito_barrio_2026.csv)
        ano: Año a extraer (por defecto el más reciente, 2026)

    Returns:
        DataFrame con "barrio_code" (str, 3 dígitos), "barrio", "distrito_code",
        "poblacion_total"
    """
    # utf-8-sig: el fichero es UTF-8 con BOM (leerlo como latin-1 corrompe
    # los nombres de barrio con tilde, p.ej. "Pacífico" -> "PacÃ­fico").
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", dtype=str)

    df = df[df["fecha"] == f"1 de enero de {ano}"].copy()

    df["num_personas"] = (
        df["num_personas"].str.replace(".", "", regex=False).astype(float)
    )

    # cod_barrio es el código de barrio a nivel de todo el municipio
    # (distrito + barrio local, p.ej. "131" = distrito 13, barrio 1), igual
    # que barrio_code en el dataset principal -- solo hace falta el zfill(3)
    # para los distritos de un solo dígito ("11" -> "011").
    df["barrio_code"] = df["cod_barrio"].astype(str).str.zfill(3)
    df["distrito_code"] = df["cod_distrito"].astype(str).str.zfill(2)

    poblacion = (
        df.groupby(["barrio_code", "barrio", "distrito_code"], as_index=False)
        ["num_personas"].sum()
        .rename(columns={"num_personas": "poblacion_total"})
    )

    return poblacion


def enriquecer_con_incidencias(
    datos: gpd.GeoDataFrame,
    data_dir: str,
    ruta_poblacion: str,
    barrio_code_col: str = "barrio_code",
) -> gpd.GeoDataFrame:
    """
    Enriquece un GeoDataFrame de anuncios con incidencias ponderadas por
    población, a nivel de barrio.

    Une incidencias y población por nombre de barrio normalizado -- el campo
    Distrito de las incidencias no es fiable (barrios de un distrito aparecen
    bajo otros distritos), así que Barrio es la única georreferencia
    utilizable -- calcula tasa de delitos por 10.000 habitantes, y une al
    dataset principal por barrio_code.

    Args:
        datos: GeoDataFrame con anuncios (debe tener columna 'barrio_code')
        data_dir: Ruta al directorio con CSVs de incidencias
        ruta_poblacion: Ruta al CSV de población
        barrio_code_col: Nombre de la columna de código de barrio (default: "barrio_code")

    Returns:
        GeoDataFrame enriquecido con columnas:
        - "no_delitos_barrio": número total de delitos por barrio
        - "poblacion_barrio": población del barrio
        - "delitos_per_10k_barrio": tasa de delitos por 10.000 habitantes
    """
    print(f"\nEnriqueciendo con incidencias ponderadas por población...")

    # Cargar población
    print("  Cargando población por barrio...")
    poblacion = cargar_poblacion_por_barrio(ruta_poblacion)

    # Cargar incidencias
    print("  Cargando incidencias...")
    agg_by_barrio, _ = crear_indice_criminalidad(data_dir)

    # Crear mapeo normalizado: nombre_normalizado -> barrio_code, a partir de
    # la tabla de población (fuente autorizada de nombre + código de barrio)
    barrio_mapping = poblacion[[barrio_code_col, "barrio"]].dropna().drop_duplicates()
    barrio_mapping["barrio_norm"] = barrio_mapping["barrio"].apply(_normalizar)
    mapeo_dict = dict(zip(barrio_mapping["barrio_norm"], barrio_mapping[barrio_code_col]))

    # Normalizar nombres de barrio en los datos de incidencias y mapear a código
    agg_by_barrio["barrio_norm"] = agg_by_barrio["Barrio"].apply(_normalizar)
    agg_by_barrio[barrio_code_col] = agg_by_barrio["barrio_norm"].map(mapeo_dict)

    print(f"  Barrios mapeados: {agg_by_barrio[barrio_code_col].notna().sum()}/{len(agg_by_barrio)}")

    # Unir incidencias con población por código de barrio
    crimen_poblacion = agg_by_barrio.merge(
        poblacion,
        on=barrio_code_col,
        how="left",
    )

    # Calcular tasa de delitos por 10.000 habitantes
    crimen_poblacion["delitos_per_10k_barrio"] = (
        (crimen_poblacion["Total_Incidentes"] / crimen_poblacion["poblacion_total"]) * 10000
    ).round(2)

    datos_copy = datos.copy()

    # Join con el dataset principal por barrio_code
    crimen_poblacion_renamed = crimen_poblacion.rename(columns={
        "Total_Incidentes": "no_delitos_barrio",
        "poblacion_total": "poblacion_barrio",
    })

    datos_copy = datos_copy.merge(
        crimen_poblacion_renamed[[
            barrio_code_col,
            "no_delitos_barrio",
            "poblacion_barrio",
            "delitos_per_10k_barrio",
        ]].dropna(subset=[barrio_code_col]).drop_duplicates(subset=[barrio_code_col]),
        on=barrio_code_col,
        how="left",
    )

    mean_delitos_raw = datos_copy["no_delitos_barrio"].mean()
    mean_delitos_per_10k = datos_copy["delitos_per_10k_barrio"].mean()
    count_with_data = datos_copy["delitos_per_10k_barrio"].notna().sum()

    print(f"  Unidas incidencias para {count_with_data:,} anuncios")
    print(f"  Media de delitos por barrio: {mean_delitos_raw:.0f}")
    print(f"  Media de delitos por 10k habitantes: {mean_delitos_per_10k:.1f}")

    # Mostrar tasa de criminalidad por barrio (top/bottom 10 -- 131 barrios es
    # demasiado para listar entero)
    tasa_por_barrio = (
        crimen_poblacion_renamed[[
            barrio_code_col, "Barrio", "no_delitos_barrio", "poblacion_barrio", "delitos_per_10k_barrio"
        ]]
        .dropna(subset=["delitos_per_10k_barrio"])
        .drop_duplicates(subset=[barrio_code_col])
        .sort_values("delitos_per_10k_barrio", ascending=False)
    )

    print(f"\n  Tasa de criminalidad por barrio (delitos/10k habitantes) -- top 10:")
    for _, row in tasa_por_barrio.head(10).iterrows():
        print(f"    {row['Barrio']:40s} {row['delitos_per_10k_barrio']:8.1f} (n={row['no_delitos_barrio']:6.0f}, pop={row['poblacion_barrio']:7.0f})")

    print(f"\n  Tasa de criminalidad por barrio (delitos/10k habitantes) -- bottom 10:")
    for _, row in tasa_por_barrio.tail(10).iterrows():
        print(f"    {row['Barrio']:40s} {row['delitos_per_10k_barrio']:8.1f} (n={row['no_delitos_barrio']:6.0f}, pop={row['poblacion_barrio']:7.0f})")

    return datos_copy
