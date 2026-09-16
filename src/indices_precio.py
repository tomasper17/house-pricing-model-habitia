"""
Índice de precio de venta por distrito para trasladar el modelo (entrenado con precios de
2018) a un año posterior, y factor para pasar de precio de venta a renta mensual.

Fuentes:
- Venta: `data/raw/madrid_compraventa/serie_compraventa_madrid.xlsx` -- precio medio declarado (€/m²) por
  distrito y barrio, Colegio de Registradores, 2007-2025.
- Alquiler: `data/raw/madrid_alquiler/alquiler_madrid_historico_distritoi.xlsx` -- renta mediana de
  vivienda colectiva (€/m²·mes) por distrito, dato oficial del SEIR (Ministerio de Vivienda y
  Agenda Urbana), 2024. Es distinto del fichero por sección censal que usa
  `alq_mediana_eur_m2_barrio` (`produccion.py`): aquí no hace falta agregar secciones, el
  distrito ya viene calculado por la fuente.

Todo se calcula por distrito. El exceso anual de un barrio sobre su distrito en la serie de
venta revierte al año siguiente (autocorrelación -0,41; exceso 2018-2021 frente a
2021-2024: -0,52): cerca de la mitad de la diferencia barrio-distrito en un índice a 6 años
es ruido de medida. El dato de distrito recoge además todas las compraventas, sin el umbral
de 15 casos que se exige al barrio.

La venta es una media y el alquiler una media de medianas, así que el factor es una
aproximación del nivel de renta, no un cociente entre dos medias del mismo tipo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import rutas

RUTA_VENTA = rutas.RUTA_COMPRAVENTA
RUTA_ALQUILER = rutas.RUTA_ALQUILER_DISTRITO

AÑO_BASE = 2018
AÑO_INICIO_TENDENCIA = 2018


def _numero_es(serie: pd.Series) -> pd.Series:
    """'3.746,27' -> 3746.27; '..' -> NaN. Las celdas ya numéricas se respetan."""
    def convertir(v):
        if isinstance(v, (int, float, np.number)):
            return float(v)
        try:
            return float(str(v).strip().replace(".", "").replace(",", "."))
        except ValueError:
            return np.nan
    return serie.map(convertir)



def cargar_serie_venta(ruta: Path = RUTA_VENTA) -> pd.DataFrame:
    """
    Formato largo: nivel (ciudad/distrito/barrio), codigo, nombre, ano, venta_eur_m2.
    `venta_eur_m2` es NaN donde el registro no publica dato ('..' o 0).
    """
    crudo = pd.read_excel(ruta, header=None)
    anos = pd.to_numeric(crudo.iloc[5, 2:], errors="coerce")
    col_ano = {col: int(a) for col, a in anos.items() if pd.notna(a)}

    etiqueta = crudo[1].astype(str).str.strip()
    partes = etiqueta.str.extract(r"^(\d{2,3})\.\s*(.+)$")
    es_ciudad = etiqueta.eq("Ciudad de Madrid")
    filas = crudo[partes[0].notna() | es_ciudad].copy()
    filas["codigo"] = partes.loc[filas.index, 0].fillna("ciudad")
    filas["nombre"] = partes.loc[filas.index, 1].fillna("Ciudad de Madrid")
    filas["nivel"] = filas["codigo"].map(
        lambda c: "ciudad" if c == "ciudad" else "distrito" if len(c) == 2 else "barrio")

    largo = filas.melt(id_vars=["nivel", "codigo", "nombre"], value_vars=list(col_ano),
                       var_name="columna", value_name="valor")
    largo["ano"] = largo["columna"].map(col_ano)
    largo["venta_eur_m2"] = _numero_es(largo["valor"]).where(lambda s: s > 0)
    return largo[["nivel", "codigo", "nombre", "ano", "venta_eur_m2"]].reset_index(drop=True)


def cargar_serie_alquiler(ruta: Path = RUTA_ALQUILER) -> pd.DataFrame:
    """
    Formato largo: nivel (ciudad/distrito), codigo, ano, alquiler_eur_m2_mes, n_viviendas.

    Una sola hoja con el dato oficial de distrito (renta mediana de vivienda colectiva,
    €/m²·mes) para el último año publicado; el año se lee de la propia hoja.
    """
    libro = pd.ExcelFile(ruta)
    crudo = pd.read_excel(libro, sheet_name=libro.sheet_names[0], header=None)
    ano = int(crudo.iloc[9, 1])

    etiqueta = crudo[1].astype(str).str.strip()
    partes = etiqueta.str.extract(r"^(\d{2})\s*-\s*(.+)$")
    es_ciudad = etiqueta.eq("Ciudad de Madrid")
    filas = crudo[partes[0].notna() | es_ciudad].copy()

    return pd.DataFrame({
        "nivel": np.where(es_ciudad.loc[filas.index], "ciudad", "distrito"),
        "codigo": partes.loc[filas.index, 0].fillna("ciudad"),
        "ano": ano,
        "alquiler_eur_m2_mes": pd.to_numeric(filas[8], errors="coerce"),
        "n_viviendas": pd.to_numeric(filas[3], errors="coerce"),
    }).reset_index(drop=True)


def serie_alquiler_secciones(anos, ruta: Path = rutas.RUTA_ALQUILER) -> pd.DataFrame:
    """
    Renta por distrito y ciudad (€/m²·mes) agregada desde el fichero por sección censal
    (media de medianas ponderada por nº de viviendas). Filas: `codigo` (distrito o 'ciudad');
    columnas: año. Solo sirve para medir la tendencia: su nivel difiere del dato oficial de
    distrito en 1-3 %, y el nivel que se usa es siempre el oficial.
    """
    from src.produccion import _alquiler_por_seccion

    filas = {}
    for ano in anos:
        secciones = _alquiler_por_seccion(ano, ruta)
        ponderada = (secciones.assign(p=secciones["renta"] * secciones["viviendas"], codigo=secciones["seccion"].str[:2])
                     .groupby("codigo")[["p", "viviendas"]].sum())
        filas[ano] = pd.concat([ponderada["p"] / ponderada["viviendas"],
                                pd.Series({"ciudad": np.average(secciones["renta"], weights=secciones["viviendas"])})])
    return pd.DataFrame(filas)


def crecimiento_tendencia(tabla: pd.DataFrame, ano_inicio: int, ano_final: int) -> pd.Series:
    """
    Tasa anual de la tendencia log-lineal de cada fila entre `ano_inicio` y `ano_final`
    (pendiente de MCO de log(valor) sobre el año, en tasa). Usa todos los años con dato de la
    ventana, así que suaviza los saltos de un año frente a una tasa entre extremos. Filas con
    menos de dos años con dato: NaN.
    """
    anos = np.array([a for a in tabla.columns if ano_inicio <= a <= ano_final])
    logaritmos = np.log(tabla[list(anos)].astype(float))

    def pendiente(fila: pd.Series) -> float:
        hay = fila.notna().to_numpy()
        return np.polyfit(anos[hay], fila.to_numpy()[hay], 1)[0] if hay.sum() >= 2 else np.nan

    return np.exp(logaritmos.apply(pendiente, axis=1)) - 1


def construir_indices(ano_base: int = AÑO_BASE, ano_destino: int | None = None, ano_renta: int | None = None,
                      venta: pd.DataFrame | None = None, alquiler: pd.DataFrame | None = None,
                      ano_inicio_tendencia: int = AÑO_INICIO_TENDENCIA) -> pd.DataFrame:
    """
    Una fila por distrito (código de 2 dígitos = los dos primeros de `barrio_code`):

    - `venta_<base>`, `venta_<destino>` (€/m²) e `indice_venta` (destino / base).
    - `alquiler_<renta>` (€/m²·mes) y `factor_renta_mensual`: renta mensual como fracción del
      precio de venta en `ano_renta` (renta mensual ≈ precio × factor). Se aplica al precio de
      `ano_destino`, lo que supone que la relación renta/precio no cambia entre los dos años.
      `anos_de_renta` y `rentabilidad_bruta_pct` son la misma relación expresada de otra forma.

    `ano_destino` por defecto es el último año con dato de venta; `ano_renta`, el último año
    con dato en ambas series que no pasa de `ano_destino` (la serie de alquiler acaba antes).

    **Proyección.** Un año posterior al último publicado se proyecta desde el último dato con
    la tendencia log-lineal de cada distrito desde `ano_inicio_tendencia`
    (`crecimiento_tendencia`): la venta con su serie registral (`crecimiento_venta_anual`) y
    el alquiler oficial con la tendencia del agregado por secciones
    (`serie_alquiler_secciones`, `crecimiento_alquiler_anual`), porque el fichero oficial de
    distrito solo trae un año. El factor sale del alquiler y la venta proyectados. `attrs`
    registra los últimos años observados.
    """
    venta = cargar_serie_venta() if venta is None else venta
    alquiler = cargar_serie_alquiler() if alquiler is None else alquiler
    venta_nc = (venta[venta["nivel"].isin(["distrito", "ciudad"])]
                .pivot(index="codigo", columns="ano", values="venta_eur_m2").dropna(axis=1, how="all"))
    alquiler_nc = (alquiler[alquiler["nivel"].isin(["distrito", "ciudad"])]
                   .pivot(index="codigo", columns="ano", values="alquiler_eur_m2_mes"))
    ultimo_venta, ultimo_alquiler = int(venta_nc.columns.max()), int(alquiler_nc.columns.max())
    if ano_destino is None:
        ano_destino = ultimo_venta
    if ano_renta is None:
        comunes = [a for a in set(venta_nc.columns) & set(alquiler_nc.columns) if a <= ano_destino]
        if not comunes:
            raise ValueError(f"ningún año con venta y alquiler hasta {ano_destino}")
        ano_renta = int(max(comunes))

    crecimiento_venta = crecimiento_alquiler = None
    if max(ano_destino, ano_renta) > ultimo_venta:
        crecimiento_venta = crecimiento_tendencia(venta_nc, ano_inicio_tendencia, ultimo_venta)
        for ano in range(ultimo_venta + 1, max(ano_destino, ano_renta) + 1):
            venta_nc[ano] = venta_nc[ultimo_venta] * (1 + crecimiento_venta) ** (ano - ultimo_venta)
    if ano_renta > ultimo_alquiler:
        secciones = serie_alquiler_secciones(range(ano_inicio_tendencia, ultimo_alquiler + 1))
        crecimiento_alquiler = crecimiento_tendencia(secciones, ano_inicio_tendencia, ultimo_alquiler)
        alquiler_nc[ano_renta] = (alquiler_nc[ultimo_alquiler]
                                  * (1 + crecimiento_alquiler.reindex(alquiler_nc.index)) ** (ano_renta - ultimo_alquiler))

    for nombre, tabla, anos in [("venta", venta_nc, {ano_base, ano_destino, ano_renta}),
                                ("alquiler", alquiler_nc, {ano_renta})]:
        faltan = anos - set(tabla.columns)
        if faltan:
            raise ValueError(f"serie de {nombre} sin datos de distrito para {sorted(faltan)}")
    venta_d, alquiler_d = venta_nc.drop(index="ciudad"), alquiler_nc.drop(index="ciudad", errors="ignore")

    nombres = venta[venta["nivel"].eq("distrito")].drop_duplicates("codigo").set_index("codigo")["nombre"]
    out = pd.DataFrame({
        "distrito": nombres,
        f"venta_{ano_base}": venta_d[ano_base],
        f"venta_{ano_destino}": venta_d[ano_destino],
    })
    out[f"venta_{ano_renta}"] = venta_d[ano_renta]
    out[f"alquiler_{ano_renta}"] = alquiler_d[ano_renta]
    out["indice_venta"] = out[f"venta_{ano_destino}"] / out[f"venta_{ano_base}"]
    out["factor_renta_mensual"] = out[f"alquiler_{ano_renta}"] / out[f"venta_{ano_renta}"]
    out["anos_de_renta"] = 1 / (12 * out["factor_renta_mensual"])
    out["rentabilidad_bruta_pct"] = 1200 * out["factor_renta_mensual"]
    if crecimiento_venta is not None:
        out["crecimiento_venta_anual"] = crecimiento_venta
    if crecimiento_alquiler is not None:
        out["crecimiento_alquiler_anual"] = crecimiento_alquiler
    out = out.rename_axis("distrito_code").reset_index()
    out.attrs.update(ano_base=ano_base, ano_destino=ano_destino, ano_renta=ano_renta,
                     ultimo_ano_venta=ultimo_venta, ultimo_ano_alquiler=ultimo_alquiler,
                     ano_inicio_tendencia=ano_inicio_tendencia,
                     indice_venta_ciudad=float(venta_nc.loc["ciudad", ano_destino] / venta_nc.loc["ciudad", ano_base]))
    return out
