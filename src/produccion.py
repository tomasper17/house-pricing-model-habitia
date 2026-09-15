"""
Del anuncio de la API de idealista a la matriz del modelo de producción, y del precio
predicho (nivel de 2018) al año de destino.

Cada variable se reconstruye como se construyó en los datos de entrenamiento de 2018; el
notebook 08 documenta y verifica cada paso:

- Distancias: haversine a Puerta del Sol, a la estación de metro más cercana y al punto
  más cercano del eje Castellana, con los puntos de `Madrid_POIS.rda` (idealista18). En
  km, como `DISTANCE_TO_*`.
- Barrio: unión espacial contra los barrios que resultan de disolver el seccionado censal
  de 2018 con el mapeo sección→barrio de 2026, sin mapeo inverso. Los puntos que caen en
  un hueco del mapeo se asignan al barrio más cercano (hasta 500 m).
- Alquiler de barrio: renta mediana de vivienda colectiva por sección (SERPAVI) del año de
  destino, agregada a barrio con media ponderada por nº de viviendas en alquiler y llevada
  al nivel de 2018 (`tabla_variables_barrio`).
- Delitos y vulnerabilidad: los valores de entrenamiento, que ya salen de las fuentes
  actuales; los barrios sin anuncios en entrenamiento se calculan desde esas fuentes.
- Instalaciones: desde la descripción (`src/amenidades_descripcion.py`); ascensor y garaje,
  desde los campos que la API sí devuelve.
"""

import contextlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.amenidades_descripcion import AMENIDADES_API_NO_DEVUELVE
from src.amenidades_descripcion import aplicar as aplicar_amenidades
from src.amenidades_descripcion import normalizar

from src import rutas

RAIZ = rutas.RAIZ
RUTA_POIS = rutas.RUTA_MADRID_POIS
RUTA_SECCIONES = rutas.RUTA_SECCIONES_2018
RUTA_MAPEO = rutas.RUTA_MAPEO_SECCIONES
RUTA_ALQUILER = rutas.RUTA_ALQUILER
RUTA_VULNERABILIDAD = rutas.RUTA_VULNERABILIDAD
DIR_INCIDENCIAS = rutas.DIR_INCIDENCIAS
RUTA_POBLACION = rutas.RUTA_POBLACION
RUTA_MODELADO = rutas.RUTA_DATASET_BASE
RUTA_BARRIO_ENTRENAMIENTO = rutas.RUTA_ASSET_DISTRITO

LOCATION_ID_MADRID = "0-EU-ES-28-07-001-079"
CRS_UTM = 25830
RADIO_TIERRA_KM = 6371.0088

# Filtros de búsqueda de la API por amenity. Portero y jardín no tienen filtro.
# Códigos de planta de idealista sin número. Entreplanta se asimila a bajo.
PLANTAS_TEXTO = {"bj": 0, "en": 0, "ss": -1, "st": -1}

VARIABLES_BARRIO = ["alq_mediana_eur_m2_barrio", "delitos_per_10k_barrio", "indice_vulnerabilidad"]


# ------------------------------------------------------------------------------
# Distancias
# ------------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def cargar_pois(ruta: Path = RUTA_POIS) -> dict[str, pd.DataFrame]:
    """`City_Center` (1 punto), `Metro` (estaciones) y `Castellana` (puntos del eje), en lon/lat."""
    import rdata  # solo para leer el .rda; el predictor exportado lee los puntos desde parquet

    return rdata.conversion.convert(rdata.parser.parse_file(ruta))["Madrid_POIS"]


def cargar_anuncios_2018(columnas: list[str], ruta: Path = rutas.RUTA_MADRID_SALE) -> pd.DataFrame:
    """
    Anuncios originales de idealista18, una fila por `ASSETID` (la primera), con `columnas`
    (debe incluir `ASSETID`). Sirve para comprobar contra 2018 las variables que se
    reconstruyen en producción y que el dataset base ya no trae (coordenadas).
    """
    import warnings

    import rdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # clases de `sf` sin constructor en rdata; no afectan a las columnas
        venta = rdata.conversion.convert(rdata.parser.parse_file(ruta))["Madrid_Sale"]
    venta = pd.DataFrame(venta[columnas])
    return venta.assign(ASSETID=venta["ASSETID"].astype(str)).drop_duplicates("ASSETID")


def _distancia_minima(lat: np.ndarray, lon: np.ndarray, puntos: pd.DataFrame, bloque: int = 2000) -> np.ndarray:
    plat, plon = puntos["Lat"].to_numpy()[None, :], puntos["Lon"].to_numpy()[None, :]
    salida = np.empty(len(lat))
    for i in range(0, len(lat), bloque):
        tramo = slice(i, i + bloque)
        salida[tramo] = haversine_km(lat[tramo, None], lon[tramo, None], plat, plon).min(axis=1)
    return salida


def calcular_distancias(lat, lon, pois: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    centro = pois["City_Center"]
    return pd.DataFrame({
        "DISTANCE_TO_CITY_CENTER": haversine_km(lat, lon, centro["Lat"].iloc[0], centro["Lon"].iloc[0]),
        "DISTANCE_TO_METRO": _distancia_minima(lat, lon, pois["Metro"]),
        "DISTANCE_TO_CASTELLANA": _distancia_minima(lat, lon, pois["Castellana"]),
    })


# ------------------------------------------------------------------------------
# Barrio
# ------------------------------------------------------------------------------
def cargar_mapeo_secciones(ruta: Path = RUTA_MAPEO) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def cargar_barrios(ruta_secciones: Path = RUTA_SECCIONES, mapeo: dict | None = None) -> gpd.GeoDataFrame:
    mapeo = cargar_mapeo_secciones() if mapeo is None else mapeo
    secciones = gpd.read_file(ruta_secciones)
    secciones = secciones[secciones["CUSEC"].str[:5] == "28079"].to_crs(CRS_UTM)
    secciones["barrio_code"] = secciones["CUSEC"].str[5:].map(lambda s: mapeo.get(s, {}).get("COD_BAR"))
    barrios = (secciones.dropna(subset=["barrio_code"])
               .dissolve(by="barrio_code").reset_index()[["barrio_code", "geometry"]])
    barrios["distrito_code"] = barrios["barrio_code"].str[:2]
    barrios.attrs["secciones_sin_barrio"] = int(secciones["barrio_code"].isna().sum())
    return barrios


def asignar_barrio(lat, lon, barrios: gpd.GeoDataFrame, max_rescate_m: float = 500) -> pd.DataFrame:
    """Barrio y distrito por punto; `barrio_rescatado` marca los asignados por cercanía."""
    puntos = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)), crs=4326,
    ).to_crs(CRS_UTM)
    capa = barrios[["barrio_code", "geometry"]]
    dentro = gpd.sjoin(puntos, capa, how="left", predicate="within")
    barrio = dentro[~dentro.index.duplicated()]["barrio_code"].reindex(puntos.index)
    fuera = barrio.isna()
    if fuera.any():
        cercano = gpd.sjoin_nearest(puntos.loc[fuera], capa, how="left", max_distance=max_rescate_m)
        barrio.loc[fuera] = cercano[~cercano.index.duplicated()]["barrio_code"]
    return pd.DataFrame({
        "barrio_code": barrio.to_numpy(),
        "distrito_code": barrio.str[:2].to_numpy(),
        "barrio_rescatado": (fuera & barrio.notna()).to_numpy(),
    })


# ------------------------------------------------------------------------------
# Variables de barrio
# ------------------------------------------------------------------------------
def _alquiler_por_seccion(ano: int, ruta: Path = RUTA_ALQUILER) -> pd.DataFrame:
    df = pd.read_excel(ruta, sheet_name=str(ano), header=5)
    codigo = df.iloc[:, 0].astype(str).str.strip()
    df = df[codigo.str.fullmatch(r"\d{1,5}")]
    secciones = pd.DataFrame({
        "seccion": df.iloc[:, 0].astype(str).str.strip().str.zfill(5),
        "renta": pd.to_numeric(df["Mediana.1"].astype(str).str.replace(",", ".", regex=False), errors="coerce"),
        "viviendas": pd.to_numeric(df.iloc[:, 1].astype(str).str.replace(",", ".", regex=False), errors="coerce"),
    })
    # descarta cabeceras/pies con código numérico suelto (p. ej. el año): no traen renta
    return secciones[secciones["renta"].notna() & (secciones["viviendas"] > 0)]


def _alquiler_por_barrio(ano: int, mapeo: dict) -> tuple[pd.Series, float]:
    secciones = _alquiler_por_seccion(ano)
    secciones["barrio_code"] = secciones["seccion"].map(lambda s: mapeo.get(s, {}).get("COD_BAR"))
    con_barrio = secciones.dropna(subset=["barrio_code"])
    g = (con_barrio.assign(ponderada=con_barrio["renta"] * con_barrio["viviendas"])
         .groupby("barrio_code")[["ponderada", "viviendas"]].sum())
    ciudad = float(np.average(secciones["renta"], weights=secciones["viviendas"]))
    return g["ponderada"] / g["viviendas"], ciudad


def valores_barrio_entrenamiento(ruta_modelado: Path = RUTA_MODELADO,
                                 ruta_barrio: Path = RUTA_BARRIO_ENTRENAMIENTO) -> pd.DataFrame:
    """Valor de las variables de barrio que vio el modelo, por `barrio_code` (constantes por barrio)."""
    columnas = ["ASSETID"] + VARIABLES_BARRIO
    df = pd.read_csv(ruta_modelado, sep=";", usecols=columnas).merge(pd.read_parquet(ruta_barrio), on="ASSETID")
    return df.groupby("barrio_code")[VARIABLES_BARRIO].first()


def _delitos_y_vulnerabilidad_desde_fuentes(codigos: list[str]) -> pd.DataFrame:
    from src.crime_indices import enriquecer_con_incidencias
    from src.vulnerability_indices import cargar_vulnerabilidad

    base = pd.DataFrame({"barrio_code": codigos})
    with contextlib.redirect_stdout(io.StringIO()):
        delitos = enriquecer_con_incidencias(base, str(DIR_INCIDENCIAS), str(RUTA_POBLACION))
    vulnerabilidad = cargar_vulnerabilidad(str(RUTA_VULNERABILIDAD))
    vulnerabilidad["barrio_code"] = vulnerabilidad["barrio_code"].str.zfill(3)
    return (delitos.set_index("barrio_code")[["delitos_per_10k_barrio"]]
            .join(vulnerabilidad.set_index("barrio_code")["vulnerabilidad_indice"].rename("indice_vulnerabilidad")))


def _ultimo_ano_alquiler(ruta: Path = RUTA_ALQUILER) -> int:
    return max(int(hoja) for hoja in pd.ExcelFile(ruta).sheet_names if str(hoja).isdigit())


def tabla_variables_barrio(ano_destino: int = 2024, ano_base: int = 2018, mapeo: dict | None = None,
                           entrenamiento: pd.DataFrame | None = None, ano_inicio_tendencia: int = 2018) -> pd.DataFrame:
    """
    Variables de barrio para anuncios actuales, una fila por `barrio_code`.

    `alq_mediana_eur_m2_barrio` = alquiler de barrio del año de destino ÷ índice de alquiler
    de la ciudad (destino / base) × calibración de método. La calibración es la mediana, entre
    barrios, de valor de entrenamiento / valor de `ano_base` calculado con este mismo método:
    corrige que el agregado de entrenamiento se hizo con otro mapeo e imputación, de modo que
    un barrio cuyo alquiler relativo no cambia recibe el valor que vio el modelo.

    Si `ano_destino` pasa del último año del fichero de alquiler, el alquiler de cada barrio y
    el de la ciudad se proyectan desde ese último año con su tendencia log-lineal desde
    `ano_inicio_tendencia` (`crecimiento_tendencia`, columna `crecimiento_alquiler_anual`); un
    barrio con menos de dos años con dato toma la tasa de la ciudad. Como la variable se
    deflacta con el índice de la ciudad, solo cuenta cuánto se separa cada barrio de la
    tendencia de la ciudad.
    """
    from src.indices_precio import crecimiento_tendencia

    mapeo = cargar_mapeo_secciones() if mapeo is None else mapeo
    entrenamiento = valores_barrio_entrenamiento() if entrenamiento is None else entrenamiento

    base_barrio, base_ciudad = _alquiler_por_barrio(ano_base, mapeo)
    ultimo = _ultimo_ano_alquiler()
    crecimiento_barrio = None
    if ano_destino > ultimo:
        barrios_por_ano, ciudad_por_ano = {}, {}
        for ano in range(ano_inicio_tendencia, ultimo + 1):
            barrios_por_ano[ano], ciudad_por_ano[ano] = _alquiler_por_barrio(ano, mapeo)
        serie_barrio = pd.DataFrame(barrios_por_ano)
        crecimiento_ciudad = float(crecimiento_tendencia(pd.DataFrame([ciudad_por_ano]), ano_inicio_tendencia, ultimo).iloc[0])
        crecimiento_barrio = crecimiento_tendencia(serie_barrio, ano_inicio_tendencia, ultimo).fillna(crecimiento_ciudad)
        destino_barrio = serie_barrio[ultimo].dropna() * (1 + crecimiento_barrio) ** (ano_destino - ultimo)
        destino_ciudad = ciudad_por_ano[ultimo] * (1 + crecimiento_ciudad) ** (ano_destino - ultimo)
    else:
        destino_barrio, destino_ciudad = _alquiler_por_barrio(ano_destino, mapeo)
    indice_ciudad = destino_ciudad / base_ciudad
    comunes = entrenamiento.index.intersection(base_barrio.index)
    calibracion = float(np.median(entrenamiento.loc[comunes, "alq_mediana_eur_m2_barrio"] / base_barrio.loc[comunes]))

    codigos = sorted(set(destino_barrio.index) | set(entrenamiento.index))
    out = pd.DataFrame(index=pd.Index(codigos, name="barrio_code"))
    out[f"alquiler_{ano_destino}_crudo"] = destino_barrio.reindex(out.index)
    out["alq_mediana_eur_m2_barrio"] = out[f"alquiler_{ano_destino}_crudo"] / indice_ciudad * calibracion
    if crecimiento_barrio is not None:
        out["crecimiento_alquiler_anual"] = crecimiento_barrio.reindex(out.index)

    sin_entrenamiento = [c for c in codigos if c not in entrenamiento.index]
    out[["delitos_per_10k_barrio", "indice_vulnerabilidad"]] = (
        entrenamiento[["delitos_per_10k_barrio", "indice_vulnerabilidad"]].reindex(out.index))
    out["fuente_delitos_vulnerabilidad"] = "entrenamiento"
    if sin_entrenamiento:
        fuentes = _delitos_y_vulnerabilidad_desde_fuentes(sin_entrenamiento)
        out.loc[sin_entrenamiento, ["delitos_per_10k_barrio", "indice_vulnerabilidad"]] = (
            fuentes.reindex(sin_entrenamiento)[["delitos_per_10k_barrio", "indice_vulnerabilidad"]].to_numpy())
        out.loc[sin_entrenamiento, "fuente_delitos_vulnerabilidad"] = "fuentes"

    out.attrs.update(ano_base=ano_base, ano_destino=ano_destino, ultimo_ano_alquiler=ultimo,
                     ano_inicio_tendencia=ano_inicio_tendencia, indice_alquiler_ciudad=indice_ciudad,
                     calibracion_metodo=calibracion, alquiler_ciudad_base=base_ciudad,
                     alquiler_ciudad_destino=destino_ciudad)
    return out


# ------------------------------------------------------------------------------
# Matriz del modelo
# ------------------------------------------------------------------------------
def _columna(df: pd.DataFrame, nombre: str, defecto=np.nan) -> pd.Series:
    return df[nombre] if nombre in df.columns else pd.Series(defecto, index=df.index, dtype=object)


def parsear_planta(planta) -> float:
    """Código de planta de idealista a número; NaN si no se puede interpretar."""
    if not isinstance(planta, str):
        return np.nan
    texto = planta.strip().lower()
    if texto in PLANTAS_TEXTO:
        return float(PLANTAS_TEXTO[texto])
    try:
        return float(int(texto))
    except ValueError:
        return np.nan


def construir_variables(anuncios: pd.DataFrame, columnas_modelo: list[str], tabla_barrio: pd.DataFrame,
                        pois: dict[str, pd.DataFrame], barrios: gpd.GeoDataFrame,
                        mediana_planta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    `anuncios`: respuesta de la API aplanada con `pd.json_normalize`.
    Devuelve la matriz con las columnas y el orden del modelo, y un `contexto` por anuncio
    (identificación, precio anunciado, barrio y marcas de cómo se construyó cada fila).
    """
    df = anuncios.reset_index(drop=True)
    X = pd.DataFrame(index=df.index)
    X["CONSTRUCTEDAREA"] = pd.to_numeric(_columna(df, "size"), errors="coerce")
    X["ROOMNUMBER"] = pd.to_numeric(_columna(df, "rooms"), errors="coerce")
    X["BATHNUMBER"] = pd.to_numeric(_columna(df, "bathrooms"), errors="coerce")

    amenidades = aplicar_amenidades(pd.DataFrame({"description": _columna(df, "description", None)}))
    for amenidad in AMENIDADES_API_NO_DEVUELVE:
        X[amenidad] = amenidades[amenidad].astype(int)

    ascensor_api = _columna(df, "hasLift")
    X["HASLIFT"] = ascensor_api.where(ascensor_api.notna(), amenidades["HASLIFT"]).astype(bool).astype(int)
    garaje_api = _columna(df, "parkingSpace.hasParkingSpace")
    X["HASPARKINGSPACE"] = garaje_api.where(garaje_api.notna(), False).astype(bool).astype(int)

    subtipo = _columna(df, "detailedType.subTypology", "")
    tipo = _columna(df, "propertyType", "")
    X["ISDUPLEX"] = (subtipo.eq("duplex") | tipo.eq("duplex")).astype(int)
    X["ISSTUDIO"] = (subtipo.eq("studio") | tipo.eq("studio")).astype(int)

    planta = _columna(df, "floor", None).map(parsear_planta)
    X["FLOORCLEAN"] = planta.fillna(mediana_planta)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    distancias = calcular_distancias(lat, lon, pois)
    for columna in distancias.columns:
        X[columna] = distancias[columna].to_numpy()

    barrio = asignar_barrio(lat, lon, barrios)
    valores = tabla_barrio.reindex(barrio["barrio_code"])[VARIABLES_BARRIO]
    for columna in VARIABLES_BARRIO:
        X[columna] = valores[columna].to_numpy()

    faltan = set(columnas_modelo) - set(X.columns)
    if faltan:
        raise ValueError(f"variables del modelo sin construir: {sorted(faltan)}")

    # Obra nueva: una promoción suele anunciar cada vivienda por separado, así que sus errores no
    # son independientes.
    obra_nueva = _columna(df, "newDevelopment", False).fillna(False).astype(bool)

    contexto = pd.DataFrame({
        "propertyCode": _columna(df, "propertyCode"),
        "precio_anunciado": pd.to_numeric(_columna(df, "price"), errors="coerce"),
        "barrio_code": barrio["barrio_code"],
        "distrito_code": barrio["distrito_code"],
        "barrio_rescatado": barrio["barrio_rescatado"],
        "sin_barrio": barrio["barrio_code"].isna(),
        "planta_imputada": planta.isna(),
        "ascensor_desde_descripcion": ascensor_api.isna(),
        "sin_descripcion": amenidades["sin_descripcion"],
        "obra_nueva": obra_nueva,
    })
    return X[list(columnas_modelo)], contexto


# Tipologías que el modelo no sabe valorar: no tiene variable de parcela ni de vivienda unifamiliar.
TIPOS_CASA = {"chalet", "countryHouse"}
SUBTIPOS_CASA = {"independantHouse", "semidetachedHouse", "terracedHouse"}


# Estado y ocupación, leídos de la descripción normalizada (`normalizar`: minúsculas, sin tildes).
# Un anuncio queda marcado si menciona el patrón y no contiene en ningún punto su anulación.
PATRON_A_REFORMAR = (r"\b(a|para) (reformar|actualizar)\b"
                     r"|\b(necesita|precisa|requiere) (de )?(una |un poco de |algo de )?(reforma|actualizacion)\b")
ANULA_A_REFORMAR = (r"sin necesidad de (reforma|reformar|obras|actualizar)"
                    r"|no (necesita|precisa|requiere) (de )?(ninguna |una )?(reforma|obras|actualizacion)")
# "nuda propiedad - valoracion..." es la lista de servicios del pie de algunas agencias, no el anuncio.
PATRON_OCUPADA = (r"\b(ocupad[oa]|okupad[oa]|okupas?)\b|sin posesion|nuda propiedad(?!\s*-\s*valoracion)|usufructo|renta antigua"
                  r"|\bcon inquilin[oa]s?\b|\binquilino actual|\b(actualmente|se encuentra|esta|vende) alquilad[oa]\b"
                  r"|\balquilad[oa] (actualmente|con contrato|hasta)\b|contrato de alquiler (en vigor|vigente)")
ANULA_OCUPADA = r"libre de (inquilin|ocupant|cargas y ocupant)|sin inquilin|desocupad"


def _descripcion_menciona(anuncios: pd.DataFrame, patron: str, anula: str) -> np.ndarray:
    texto = _columna(anuncios, "description", None).map(normalizar)
    return (texto.str.contains(patron, regex=True) & ~texto.str.contains(anula, regex=True)).to_numpy(dtype=bool)


def motivos_fuera_de_dominio(anuncios: pd.DataFrame, X: pd.DataFrame, area_max: float) -> pd.DataFrame:
    """
    Por anuncio, las razones para no valorarlo aunque tenga todos los datos:
    `tipologia_casa` (chalet o casa: el modelo no ve la parcela), `superficie_fuera_de_dominio`
    (superficie por encima de `area_max`, el percentil 99 de train: sin soporte y los árboles
    no extrapolan), `a_reformar` (la descripción dice que está a reformar o para actualizar: el
    modelo no ve el estado) y `ocupada` (la descripción dice que está ocupada, alquilada o sin
    posesión: se vende con descuento que el modelo no ve). Las dos últimas salen del texto.
    """
    df = anuncios.reset_index(drop=True)
    casa = (_columna(df, "propertyType", "").isin(TIPOS_CASA)
            | _columna(df, "detailedType.subTypology", "").isin(SUBTIPOS_CASA))
    return pd.DataFrame({
        "tipologia_casa": casa.to_numpy(dtype=bool),
        "superficie_fuera_de_dominio": (X["CONSTRUCTEDAREA"] > area_max).to_numpy(dtype=bool),
        "a_reformar": _descripcion_menciona(df, PATRON_A_REFORMAR, ANULA_A_REFORMAR),
        "ocupada": _descripcion_menciona(df, PATRON_OCUPADA, ANULA_OCUPADA),
    }, index=X.index)


def fuera_de_rango(X: pd.DataFrame, rangos_train: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Por variable, si el valor queda fuera del [mínimo, máximo] visto en train."""
    return pd.DataFrame({
        c: (X[c] < rangos_train[c]["min"]) | (X[c] > rangos_train[c]["max"])
        for c in X.columns if c in rangos_train
    })


# ------------------------------------------------------------------------------
# Predicción y traslado de precio
# ------------------------------------------------------------------------------
def predecir_precio_base(modelo, X: pd.DataFrame, smearing: float) -> np.ndarray:
    """Precio en euros al nivel de los datos de entrenamiento (2018), con la corrección de Duan."""
    return np.exp(modelo.predict(X)) * smearing


def ajustar_precio(precio_base, distrito_code, indices: pd.DataFrame) -> pd.DataFrame:
    """Precio en el año de destino (índice de venta del distrito) y renta mensual estimada."""
    tabla = indices.set_index("distrito_code")
    distrito = pd.Series(np.asarray(distrito_code, dtype=object))
    indice = distrito.map(tabla["indice_venta"]).to_numpy(dtype=float)
    factor = distrito.map(tabla["factor_renta_mensual"]).to_numpy(dtype=float)
    precio_destino = np.asarray(precio_base, dtype=float) * indice
    return pd.DataFrame({
        "indice_venta": indice,
        "precio_estimado_destino": precio_destino,
        "factor_renta_mensual": factor,
        "renta_mensual_estimada": precio_destino * factor,
    })


# ------------------------------------------------------------------------------
# API con caché
# ------------------------------------------------------------------------------
def cliente_api_desde_entorno():
    """
    Cliente autenticado con las credenciales del entorno. Antes lee el `.env` de la raíz del
    repositorio, sin pisar variables ya definidas. El secreto se acepta como
    `IDEALISTA_API_SECRET` o `IDEALISTA_SECRET`.
    """
    from src.api_extractor import IdeallistaAPIClient

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except ImportError:
        pass
    clave = os.environ.get("IDEALISTA_API_KEY")
    secreto = os.environ.get("IDEALISTA_API_SECRET") or os.environ.get("IDEALISTA_SECRET")
    if not clave or not secreto:
        raise RuntimeError("Define IDEALISTA_API_KEY e IDEALISTA_API_SECRET (en el entorno o en el .env de la raíz) "
                           "para llamar a la API.")
    cliente = IdeallistaAPIClient(clave, secreto)
    if not cliente.authenticate():
        raise RuntimeError("La autenticación con la API de idealista ha fallado.")
    return cliente


def buscar_con_cache(nombre: str, parametros: dict, dir_cache: Path, llamar_api: bool,
                     cliente=None) -> tuple[list[dict], dict]:
    """
    Devuelve los anuncios de una búsqueda desde `dir_cache/<nombre>.json` si existe. Si no
    existe, solo llama a la API cuando `llamar_api` es True, y guarda la respuesta.

    Con `paginas_aleatorias` (y opcionalmente `semilla`) en `parametros`, descarga ese número de
    páginas elegidas al azar entre todas las del resultado (`search_random_pages`) en vez de
    las primeras; las páginas elegidas se guardan en la caché.
    """
    ruta = Path(dir_cache) / f"{nombre}.json"
    if ruta.exists():
        guardado = json.loads(ruta.read_text(encoding="utf-8"))
        return guardado["anuncios"], guardado
    if not llamar_api:
        raise FileNotFoundError(f"No hay caché en {ruta}. Pon LLAMAR_API = True para descargarla "
                                "(consume peticiones de la API).")
    cliente = cliente or cliente_api_desde_entorno()
    busqueda = dict(parametros)
    n_aleatorias, semilla = busqueda.pop("paginas_aleatorias", None), busqueda.pop("semilla", 0)
    paginas = None
    if n_aleatorias:
        anuncios = cliente.search_random_pages(n_aleatorias, seed=semilla, **busqueda)
        paginas = getattr(cliente, "last_pages", None)
    else:
        anuncios = cliente.search_properties(**busqueda)
    if not anuncios:
        print(f"Aviso: '{nombre}' devuelve 0 anuncios. Si hubo un error de petición arriba, borra {ruta} y repite.")
    guardado = {
        "nombre": nombre,
        "descargado": datetime.now().isoformat(timespec="seconds"),
        "parametros": parametros,
        "paginas": paginas,
        "n_anuncios": len(anuncios),
        "anuncios": anuncios,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(guardado, ensure_ascii=False, indent=1), encoding="utf-8")
    return anuncios, guardado


