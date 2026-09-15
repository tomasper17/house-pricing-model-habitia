"""
Predictor de precio de venta para anuncios de Madrid ciudad, para usar fuera de los notebooks.

Dos pasos:

1. `exportar_paquete()` -- una vez, con `data/` completo. Congela en una carpeta todo lo que
   necesita la predicción: el modelo en formato nativo de XGBoost, los metadatos de
   entrenamiento, los polígonos de barrio, los puntos de interés, las variables de barrio y
   los índices de distrito. El pipeline que lo use no necesita los datos crudos (shapefile,
   Excel del SERPAVI, serie registral, incidencias, dataset de 2018).
2. `PredictorHabitia.cargar(carpeta).predecir(anuncios)` -- anuncios de la API de idealista
   (lista de dicts, respuesta con `elementList`, JSON de caché del notebook 08 o DataFrame)
   a una fila por anuncio con precio estimado, renta mensual y marcas de calidad.

Un anuncio no se predice (precio NaN, `valido` False y `motivo_no_valido`) si le falta
superficie, habitaciones, baños o coordenadas, si no es de venta, si cae fuera de Madrid
ciudad, si es un chalet o casa (el modelo no ve la parcela), si su superficie supera el
percentil 99 de train o si la descripción dice que está a reformar u ocupada/alquilada
(`produccion.motivos_fuera_de_dominio`). El resto se predice aunque tenga advertencias (`fuera_de_rango`, planta imputada,
ascensor desde la descripción...): quien consume decide qué filtrar.

Desde la raíz del repositorio:

    python -m src.predictor exportar
    python -m src.predictor predecir data/api_idealista/madrid_ciudad_venta.json -o predicciones.csv
    python -m src.predictor empaquetar

`empaquetar` genera `dist/habitia_predictor.zip` para compartir: el código mínimo (`src/`),
el paquete en su ruta por defecto, `requirements.txt` con las versiones exactas y `LEEME.md`.
Quien lo recibe lo descomprime y usa los mismos comandos desde esa carpeta; no necesita el
repositorio, los datos crudos ni credenciales de la API.
"""

import argparse
import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb

from src import produccion as prod
from src import rutas

RAIZ = rutas.RAIZ
RUTA_PAQUETE = rutas.DIR_PAQUETE_PRODUCCION
RUTA_ARTEFACTO = rutas.RUTA_MODELO_PRODUCCION
RUTA_ZIP = rutas.DIST / "habitia_predictor.zip"
MODULOS_PREDICCION = ["predictor.py", "produccion.py", "amenidades_descripcion.py", "rutas.py"]
PROCESADOS = rutas.PROCESADOS
VERSION_PAQUETE = 3

TIPOS_POI = ["City_Center", "Metro", "Castellana"]
COLUMNAS_API = ["propertyCode", "operation", "propertyType", "price", "size", "rooms", "bathrooms",
                "latitude", "longitude", "detailedType.subTypology"]
OBLIGATORIAS = {"CONSTRUCTEDAREA": "sin_superficie", "ROOMNUMBER": "sin_habitaciones", "BATHNUMBER": "sin_banos"}
COLUMNAS_SALIDA = ["propertyCode", "propertyType", "valido", "motivo_no_valido", "barrio_code", "distrito_code",
                   "precio_anunciado", "precio_estimado_base", "indice_venta", "precio_estimado",
                   "factor_renta_mensual", "renta_mensual_estimada", "anunciado_sobre_estimado", "fuera_de_rango",
                   "barrio_rescatado", "planta_imputada", "ascensor_desde_descripcion", "sin_descripcion",
                   "obra_nueva",
                   "ano_base", "ano_precio", "ano_renta", "modelo"]
METADATOS_ARTEFACTO = ["nombre", "familia", "dataset", "columnas", "params", "smearing", "metricas_test",
                       "mediana_planta_train", "rangos_train", "medias_train", "entrenado"]


# ------------------------------------------------------------------------------
# Exportación
# ------------------------------------------------------------------------------
def exportar_paquete(destino: Path = RUTA_PAQUETE, ruta_artefacto: Path = RUTA_ARTEFACTO,
                     ano_destino: int | None = None, ano_renta: int | None = None) -> Path:
    """
    Escribe el paquete de producción en `destino`. `ano_destino` (año del precio) por defecto
    es el último con serie de venta; `ano_renta` (factor precio→renta y alquiler de barrio), el
    último con venta y alquiler (ver `construir_indices`). Comprueba que el modelo nativo
    predice igual que el artefacto de `07` sobre el test antes de dar el paquete por bueno.
    """
    import joblib
    from src.indices_precio import construir_indices

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    artefacto = joblib.load(ruta_artefacto)

    ruta_modelo = destino / "modelo.json"
    artefacto["modelo"].get_booster().save_model(ruta_modelo)
    nativo = xgb.XGBRegressor()
    nativo.load_model(ruta_modelo)
    X_test = pd.read_parquet(PROCESADOS / f"X_test_{artefacto['dataset']}.parquet")[artefacto["columnas"]]
    diferencia = float(np.max(np.abs(nativo.predict(X_test) - artefacto["modelo"].predict(X_test))))
    if diferencia > 1e-5:
        raise RuntimeError(f"el modelo exportado no reproduce al original (máx |Δ log| = {diferencia:.2e})")
    area_max = float(pd.read_parquet(PROCESADOS / f"X_train_{artefacto['dataset']}.parquet")["CONSTRUCTEDAREA"]
                     .quantile(0.99))

    indices = construir_indices(ano_destino=ano_destino, ano_renta=ano_renta)
    tabla_barrio = prod.tabla_variables_barrio(ano_destino=indices.attrs["ano_renta"])
    indices.to_parquet(destino / "indices_distrito.parquet", index=False)
    tabla_barrio.reset_index().to_parquet(destino / "variables_barrio.parquet", index=False)
    prod.cargar_barrios().to_parquet(destino / "barrios.parquet")
    pois = prod.cargar_pois()
    (pd.concat([pois[t][["Lon", "Lat"]].assign(tipo=t) for t in TIPOS_POI], ignore_index=True)
     .to_parquet(destino / "pois.parquet", index=False))

    metadatos = {clave: artefacto[clave] for clave in METADATOS_ARTEFACTO}
    metadatos.update(
        version_paquete=VERSION_PAQUETE,
        exportado=datetime.now().isoformat(timespec="seconds"),
        ano_base=indices.attrs["ano_base"],
        ano_precio=indices.attrs["ano_destino"],
        ano_renta=indices.attrs["ano_renta"],
        ano_alquiler_barrio=tabla_barrio.attrs["ano_destino"],
        indice_alquiler_ciudad=tabla_barrio.attrs["indice_alquiler_ciudad"],
        calibracion_metodo=tabla_barrio.attrs["calibracion_metodo"],
        area_max_dominio=area_max,
        verificacion_modelo_nativo_max_abs_log=diferencia,
        versiones={"xgboost": xgb.__version__, "pandas": pd.__version__, "geopandas": gpd.__version__},
    )
    (destino / "metadatos.json").write_text(json.dumps(metadatos, ensure_ascii=False, indent=1, default=float),
                                            encoding="utf-8")
    return destino


def _leeme(m: dict) -> str:
    t = m["metricas_test"]
    return f"""# Predictor Habitia — precio de venta de viviendas en Madrid ciudad

Modelo `{m['nombre']}` ({m['familia']}, {len(m['columnas'])} variables), entrenado con anuncios de
idealista de {m['ano_base']} y trasladado a precios de {m['ano_precio']} con el índice registral de
venta de cada distrito. Paquete versión {m['version_paquete']}, exportado {m['exportado']}.

Error en test (datos de {m['ano_base']}): rmse_log {t['rmse_log']:.4f} | error mediano
{t['error_abs_mediano_eur']:,.0f} € ({t['error_pct_mediano']:.1f} %) | {t['pct_dentro_del_20pct']:.0f} % de viviendas a menos de ±20 %.

## Instalación

Python {platform.python_version()} (mínimo 3.10). Desde esta carpeta:

    pip install -r requirements.txt

## Uso

Desde esta carpeta, con un JSON de anuncios de la API de idealista (respuesta con
`elementList`, lista de anuncios o un anuncio):

    python -m src.predictor predecir anuncios.json -o predicciones.csv

o desde Python, con esta carpeta en el `PYTHONPATH`:

    from src.predictor import PredictorHabitia
    predictor = PredictorHabitia.cargar()      # una vez
    resultado = predictor.predecir(anuncios)   # lista de dicts, respuesta de la API o DataFrame

## Salida (una fila por anuncio, en el orden de entrada)

- `precio_estimado`: precio de venta estimado a nivel de {m['ano_precio']} (€).
- `renta_mensual_estimada`: precio estimado × relación renta/precio del distrito en {m['ano_renta']}.
- `anunciado_sobre_estimado`: precio anunciado / precio estimado.
- `valido` y `motivo_no_valido`: no se valora un anuncio sin superficie, habitaciones, baños
  o coordenadas, que no sea de venta, fuera de Madrid ciudad, chalet o casa, de más de
  {m['area_max_dominio']:.0f} m², o cuya descripción diga que está a reformar u ocupada/alquilada.
- Advertencias (el anuncio se valora igual): `fuera_de_rango`, `planta_imputada`,
  `ascensor_desde_descripcion`, `sin_descripcion`, `barrio_rescatado` y `obra_nueva` (una
  promoción anuncia cada vivienda por separado: sus errores no son independientes y conviene
  contar la promoción una vez en cualquier agregado).

## Límites

- El modelo no ve el estado de conservación, si es ático, las vistas ni la parcela. Una
  vivienda "a reformar" sale cara frente a su precio pedido; un ático, barato.
- Su error es de ±20 % en una de cada cinco viviendas: un anuncio aislado no se puede
  calificar de infravalorado o sobrevalorado.
- El índice de venta llega a {m['ano_precio']}; los anuncios posteriores llevan la subida de
  mercado posterior sin corregir.
- Instalaciones como terraza o aire acondicionado se leen de la descripción: lo que el
  anunciante no menciona cuenta como ausente.
"""


def empaquetar(destino: Path = RUTA_ZIP, paquete: Path = RUTA_PAQUETE) -> Path:
    """
    Zip autocontenido para compartir. Usa el paquete ya exportado, que tiene que corresponder
    a esta versión del código.
    """
    import pyarrow

    paquete = Path(paquete)
    metadatos = json.loads((paquete / "metadatos.json").read_text(encoding="utf-8"))
    if metadatos.get("version_paquete") != VERSION_PAQUETE:
        raise ValueError(f"paquete de versión {metadatos.get('version_paquete')} y código de versión "
                         f"{VERSION_PAQUETE}: ejecuta `python -m src.predictor exportar` antes")
    requisitos = [f"numpy=={np.__version__}", f"pandas=={pd.__version__}", f"pyarrow=={pyarrow.__version__}",
                  f"geopandas=={gpd.__version__}", f"xgboost=={xgb.__version__}"]
    ruta_paquete_zip = RUTA_PAQUETE.relative_to(RAIZ).as_posix()

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("src/__init__.py", "")
        for modulo in MODULOS_PREDICCION:
            z.write(RAIZ / "src" / modulo, f"src/{modulo}")
        for fichero in sorted(p for p in paquete.iterdir() if p.is_file()):
            z.write(fichero, f"{ruta_paquete_zip}/{fichero.name}")
        z.writestr("requirements.txt", "\n".join(requisitos) + "\n")
        z.writestr("LEEME.md", _leeme(metadatos))
    return destino


# ------------------------------------------------------------------------------
# Predicción
# ------------------------------------------------------------------------------
def leer_anuncios(anuncios) -> pd.DataFrame:
    """
    Acepta: ruta a un JSON, dict de caché del 08 (`anuncios`), respuesta cruda de la API
    (`elementList`), un único anuncio, lista de anuncios o DataFrame (anidado o ya aplanado).
    """
    if isinstance(anuncios, (str, Path)):
        anuncios = json.loads(Path(anuncios).read_text(encoding="utf-8"))
    if isinstance(anuncios, dict):
        anuncios = anuncios.get("anuncios", anuncios.get("elementList", [anuncios]))
    registros = anuncios.to_dict("records") if isinstance(anuncios, pd.DataFrame) else list(anuncios)
    df = pd.json_normalize(registros)
    for columna in COLUMNAS_API:
        if columna not in df.columns:
            df[columna] = np.nan
    return df.reset_index(drop=True)


class PredictorHabitia:
    """Carga el paquete una vez y predice lotes de anuncios."""

    def __init__(self, modelo, metadatos: dict, barrios: gpd.GeoDataFrame, pois: dict[str, pd.DataFrame],
                 tabla_barrio: pd.DataFrame, indices: pd.DataFrame):
        self.modelo = modelo
        self.metadatos = metadatos
        self.columnas = metadatos["columnas"]
        self.barrios = barrios
        self.pois = pois
        self.tabla_barrio = tabla_barrio
        self.indices = indices

    @classmethod
    def cargar(cls, carpeta: Path = RUTA_PAQUETE) -> "PredictorHabitia":
        carpeta = Path(carpeta)
        metadatos = json.loads((carpeta / "metadatos.json").read_text(encoding="utf-8"))
        if metadatos.get("version_paquete") != VERSION_PAQUETE:
            raise ValueError(f"paquete de versión {metadatos.get('version_paquete')}, se esperaba {VERSION_PAQUETE}")
        modelo = xgb.XGBRegressor()
        modelo.load_model(carpeta / "modelo.json")
        pois = pd.read_parquet(carpeta / "pois.parquet")
        return cls(
            modelo=modelo,
            metadatos=metadatos,
            barrios=gpd.read_parquet(carpeta / "barrios.parquet"),
            pois={t: g[["Lon", "Lat"]].reset_index(drop=True) for t, g in pois.groupby("tipo")},
            tabla_barrio=pd.read_parquet(carpeta / "variables_barrio.parquet").set_index("barrio_code"),
            indices=pd.read_parquet(carpeta / "indices_distrito.parquet"),
        )

    def preparar(self, anuncios) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Matriz del modelo, contexto de construcción y anuncios aplanados."""
        df = leer_anuncios(anuncios)
        X, contexto = prod.construir_variables(df, self.columnas, self.tabla_barrio, self.pois, self.barrios,
                                               self.metadatos["mediana_planta_train"])
        return X, contexto, df

    def predecir(self, anuncios, incluir_variables: bool = False) -> pd.DataFrame:
        X, contexto, df = self.preparar(anuncios)
        if X.empty:
            return pd.DataFrame(columns=COLUMNAS_SALIDA + ([f"x_{c}" for c in self.columnas] if incluir_variables else []))
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        sin_coordenadas = lat.isna() | lon.isna()

        problemas = pd.DataFrame({
            "operacion_no_venta": df["operation"].notna() & df["operation"].ne("sale"),
            "sin_coordenadas": sin_coordenadas,
            "fuera_de_madrid": contexto["sin_barrio"] & ~sin_coordenadas,
            **{motivo: X[columna].isna() | (X[columna] < 0) for columna, motivo in OBLIGATORIAS.items()},
        })
        problemas["sin_superficie"] |= X["CONSTRUCTEDAREA"].le(0)
        problemas = problemas.join(prod.motivos_fuera_de_dominio(df, X, self.metadatos["area_max_dominio"]))
        motivo = pd.Series([";".join(problemas.columns[fila]) for fila in problemas.to_numpy(dtype=bool)],
                           index=X.index, dtype=object)
        valido = motivo.eq("")

        fuera = prod.fuera_de_rango(X, self.metadatos["rangos_train"])
        base = prod.predecir_precio_base(self.modelo, X, self.metadatos["smearing"])
        ajuste = prod.ajustar_precio(base, contexto["distrito_code"], self.indices)

        salida = pd.DataFrame({
            "propertyCode": contexto["propertyCode"],
            "propertyType": df["propertyType"],
            "valido": valido,
            "motivo_no_valido": motivo.mask(valido),
            "barrio_code": contexto["barrio_code"],
            "distrito_code": contexto["distrito_code"],
            "precio_anunciado": contexto["precio_anunciado"],
            "precio_estimado_base": base,
            "indice_venta": ajuste["indice_venta"],
            "precio_estimado": ajuste["precio_estimado_destino"],
            "factor_renta_mensual": ajuste["factor_renta_mensual"],
            "renta_mensual_estimada": ajuste["renta_mensual_estimada"],
        })
        salida["anunciado_sobre_estimado"] = salida["precio_anunciado"] / salida["precio_estimado"]
        estimaciones = ["precio_estimado_base", "indice_venta", "precio_estimado", "factor_renta_mensual",
                        "renta_mensual_estimada", "anunciado_sobre_estimado"]
        salida.loc[~valido, estimaciones] = np.nan

        salida["fuera_de_rango"] = [";".join(fuera.columns[fila]) or None for fila in fuera.to_numpy(dtype=bool)]
        for marca in ["barrio_rescatado", "planta_imputada", "ascensor_desde_descripcion", "sin_descripcion",
                      "obra_nueva"]:
            salida[marca] = contexto[marca].to_numpy()
        salida["ano_base"] = self.metadatos["ano_base"]
        salida["ano_precio"] = self.metadatos["ano_precio"]
        salida["ano_renta"] = self.metadatos["ano_renta"]
        salida["modelo"] = self.metadatos["nombre"]
        if incluir_variables:
            salida = pd.concat([salida, X.add_prefix("x_")], axis=1)
        return salida


# ------------------------------------------------------------------------------
# Línea de comandos
# ------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.predictor", description=__doc__.split("\n\n")[0])
    ordenes = parser.add_subparsers(dest="orden", required=True)

    exportar = ordenes.add_parser("exportar", help="congela el paquete de producción")
    exportar.add_argument("--destino", type=Path, default=RUTA_PAQUETE)
    exportar.add_argument("--ano-destino", type=int, default=None, help="año del precio (def.: último con venta)")
    exportar.add_argument("--ano-renta", type=int, default=None, help="año del factor de renta (def.: último con alquiler)")

    empaq = ordenes.add_parser("empaquetar", help="zip autocontenido para compartir el predictor")
    empaq.add_argument("--destino", type=Path, default=RUTA_ZIP)
    empaq.add_argument("--paquete", type=Path, default=RUTA_PAQUETE)

    predecir = ordenes.add_parser("predecir", help="predice un JSON de anuncios")
    predecir.add_argument("entrada", type=Path)
    predecir.add_argument("-o", "--salida", type=Path, required=True, help=".csv o .parquet")
    predecir.add_argument("--paquete", type=Path, default=RUTA_PAQUETE)
    predecir.add_argument("--con-variables", action="store_true", help="añade la matriz del modelo (x_*)")

    args = parser.parse_args(argv)
    if args.orden == "exportar":
        ruta = exportar_paquete(args.destino, ano_destino=args.ano_destino, ano_renta=args.ano_renta)
        print(f"paquete -> {ruta}")
        return
    if args.orden == "empaquetar":
        ruta = empaquetar(args.destino, args.paquete)
        print(f"zip para compartir -> {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")
        return

    resultado = PredictorHabitia.cargar(args.paquete).predecir(args.entrada, incluir_variables=args.con_variables)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    if args.salida.suffix == ".parquet":
        resultado.to_parquet(args.salida, index=False)
    else:
        resultado.to_csv(args.salida, index=False)
    print(f"{len(resultado)} anuncios ({int(resultado['valido'].sum())} válidos) -> {args.salida}")


if __name__ == "__main__":
    main()
