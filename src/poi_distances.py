"""
Carga y procesamiento de datos de POI (Points of Interest) de Madrid.

Módulo para extraer puntos de interés de feeds Atom/GeoRSS del Ayuntamiento
de Madrid (parques, centros educativos, espacios deportivos, etc.) y calcular
distancias euclídeas desde cualquier conjunto de puntos (p.ej., anuncios
inmobiliarios).
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional
import geopandas as gpd
import pandas as pd

from src import rutas


def cargar_georss(ruta: str, ns_override: Optional[Dict] = None) -> gpd.GeoDataFrame:
    """
    Carga un feed Atom/GeoRSS y lo convierte a GeoDataFrame de puntos.

    Args:
        ruta: Ruta al archivo .geo (Atom/GeoRSS)
        ns_override: Namespaces alternativos (default: estándar Atom + geo)

    Returns:
        GeoDataFrame con columnas 'nombre' y 'geometry' (EPSG:4326)
    """
    NS = ns_override or {
        "atom": "http://www.w3.org/2005/Atom",
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    }

    try:
        root = ET.parse(ruta).getroot()
    except ET.ParseError as e:
        raise ValueError(f"No se pudo parsear {ruta}: {e}")

    registros = []
    for entry in root.findall("atom:entry", NS):
        titulo = entry.find("atom:title", NS)
        lon_elem = entry.find("geo:long", NS)
        lat_elem = entry.find("geo:lat", NS)

        if titulo is None or lon_elem is None or lat_elem is None:
            continue

        try:
            registros.append({
                "nombre": titulo.text,
                "lon": float(lon_elem.text),
                "lat": float(lat_elem.text),
            })
        except (ValueError, TypeError):
            continue

    if not registros:
        raise ValueError(f"No se encontraron puntos válidos en {ruta}")

    df = pd.DataFrame(registros)
    gdf = gpd.GeoDataFrame(
        df[["nombre"]],
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326",
    )
    return gdf


def cargar_centros_educativos(data_dir: Path = None) -> gpd.GeoDataFrame:
    """Carga todos los centros educativos (públicos, concertados, privados)."""
    if data_dir is None:
        data_dir = rutas.DIR_EQUIPAMIENTOS
    ruta = data_dir / "300614-4-centros-educativos-geo.geo"
    gdf = cargar_georss(str(ruta))
    gdf["poi_type"] = "centro_educativo"
    return gdf


def cargar_parques(data_dir: Path = None) -> gpd.GeoDataFrame:
    """Carga parques y jardines públicos."""
    if data_dir is None:
        data_dir = rutas.DIR_EQUIPAMIENTOS
    ruta = data_dir / "200761-1-parques-jardines-geo.geo"
    gdf = cargar_georss(str(ruta))
    gdf["poi_type"] = "parque"
    return gdf


def cargar_espacios_deporte(data_dir: Path = None) -> gpd.GeoDataFrame:
    """Carga espacios deportivos públicos."""
    if data_dir is None:
        data_dir = rutas.DIR_EQUIPAMIENTOS
    ruta = data_dir / "212808-2-espacio-deporte-geo.geo"
    gdf = cargar_georss(str(ruta))
    gdf["poi_type"] = "espacio_deporte"
    return gdf


def cargar_centros_medicos(data_dir: Path = None) -> gpd.GeoDataFrame:
    """Carga centros de atención médica."""
    if data_dir is None:
        data_dir = rutas.DIR_EQUIPAMIENTOS
    ruta = data_dir / "212769-4-atencion-medica-geo.geo"
    gdf = cargar_georss(str(ruta))
    gdf["poi_type"] = "centro_medico"
    return gdf


def calcular_distancias_poi(
    puntos_utm: gpd.GeoDataFrame,
    poi_type: str,
    poi_gdf_utm: gpd.GeoDataFrame,
) -> pd.Series:
    """
    Calcula distancia euclídea (UTM) al POI más cercano de un tipo dado.

    Args:
        puntos_utm: GeoDataFrame de puntos (p.ej., anuncios), ya en CRS_UTM
        poi_type: Tipo de POI (p.ej., 'parque', 'colegio_publico')
        poi_gdf_utm: GeoDataFrame de POI, ya en CRS_UTM

    Returns:
        Series con distancias en metros, indexada igual que puntos_utm
    """
    result = gpd.sjoin_nearest(
        puntos_utm[["geometry"]],
        poi_gdf_utm[["geometry"]],
        how="left",
        distance_col=f"dist_{poi_type}_m",
    )
    result = result[~result.index.duplicated(keep="first")]
    return result.loc[puntos_utm.index, f"dist_{poi_type}_m"].round(1)


def enriquecer_con_distancias_poi(
    datos_utm: gpd.GeoDataFrame,
    data_dir: Path = None,
    crs_utm: int = 25830,
    tipos_poi: Tuple[str, ...] = (
        "centro_educativo",
        "parque",
        "espacio_deporte",
        "centro_medico",
    ),
) -> gpd.GeoDataFrame:
    """
    Enriquece un GeoDataFrame de anuncios con distancias a diversos POI.

    Args:
        datos_utm: GeoDataFrame con anuncios, geometrías en CRS_UTM
        data_dir: Directorio con archivos .geo (default: data/raw/madrid_equipamientos)
        crs_utm: CRS métrico para cálculo de distancias
        tipos_poi: Tupla de tipos de POI a cargar y procesar

    Returns:
        GeoDataFrame enriquecido con columnas dist_<tipo>_m

    Ejemplo:
        datos_enriched = enriquecer_con_distancias_poi(
            datos_utm,
            data_dir=rutas.DIR_EQUIPAMIENTOS,
            tipos_poi=("parque", "centro_medico"),
        )
    """
    if data_dir is None:
        data_dir = rutas.DIR_EQUIPAMIENTOS

    loaders = {
        "centro_educativo": cargar_centros_educativos,
        "parque": cargar_parques,
        "espacio_deporte": cargar_espacios_deporte,
        "centro_medico": cargar_centros_medicos,
    }

    datos_copy = datos_utm.copy()

    for poi_type in tipos_poi:
        if poi_type not in loaders:
            print(f"  ⚠ Tipo de POI desconocido: {poi_type}")
            continue

        try:
            print(f"  Cargando {poi_type}...", end=" ")
            loader = loaders[poi_type]
            poi_gdf = loader(data_dir)
            poi_gdf_utm = poi_gdf.to_crs(crs_utm)

            col_name = f"dist_{poi_type}_m"
            datos_copy[col_name] = calcular_distancias_poi(
                datos_utm, poi_type, poi_gdf_utm
            )

            mean_dist = datos_copy[col_name].mean()
            count = len(poi_gdf)
            print(f"✓ ({count} puntos, media {mean_dist:.0f} m)")

        except FileNotFoundError as e:
            print(f"✗ Archivo no encontrado: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")

    return datos_copy
