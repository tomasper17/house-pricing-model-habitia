"""
Análisis de selección de variables: contribución única, descomposición por bloques
semánticos, selección forward y contraste de interacciones.

Este módulo implementa el código que respalda las cifras de
`docs/plan_inclusion_variables.md` §2-3 y de los notebooks `03_seleccion_variables`.
Hasta ahora esas cifras estaban documentadas pero no eran reproducibles; aquí lo son.

Criterio transversal (plan §1): con n = 75.803 la significación estadística está
saturada y no discrimina. Se decide por **tamaño del efecto**, medido como ΔR² único:
la caída de R² al retirar una variable (o un bloque) del modelo completo. Formalmente
es la correlación semiparcial al cuadrado de esa variable con log(PRICE) dado todo lo
demás.

Tres cautelas están incorporadas al módulo, no solo advertidas en prosa:

1. **Denominador.** Un ΔR² se lee como *cuota de la varianza residual* que la variable
   recupera, ΔR²/(1−R²_base), no como fracción de la varianza total. Con R²_base ≈
   0,912 queda solo un 8,8% de varianza por explicar: exigir un 0,5% del total equivale
   a exigir un ~5,7% de todo el error que queda, un listón desproporcionado para una
   sola variable. `cuota_residual` y `veredicto` trabajan ya en esa escala.

2. **Incertidumbre.** El ΔR² único es un estadístico muestral y ordenar variables por
   su cuarta cifra decimal sin un intervalo es exactamente el error que se le reprocha
   al p-valor, en espejo. `bootstrap_delta_r2` da intervalos por *bootstrap* de filas.

3. **Signo.** El R² dentro de muestra no puede bajar al añadir columnas, así que el
   ΔR² único es ≥ 0 por construcción y **nunca puede argumentar en contra** de una
   variable: solo produce "positivo pero pequeño", y dónde se corta "pequeño" es un
   parámetro libre. `delta_r2_cv` mide el mismo salto fuera de muestra, donde sí puede
   salir negativo y el cero deja de ser arbitrario.

Con variables redundantes la medida sigue siendo conservadora -- reparte a cero el
crédito de la señal compartida -- y por eso se lee junto a la descomposición por
bloques, a la curva forward y a `redundancia_respecto_a`, nunca sola.

Todas las funciones trabajan sobre una matriz ya transformada y dummificada, y sobre
un mapa `{nombre_variable: [columnas]}` que agrupa las dummies de una misma variable,
para que forward y ΔR² operen **por variable**, no por columna.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import transformaciones as t


# ==============================================================================
# Matriz del "modelo completo"
# ==============================================================================
def transformar_variables_completa(
    df: pd.DataFrame,
    params: t.ParametrosTransformacion,
) -> pd.DataFrame:
    """
    Vía lineal **sin descartar nada por ajuste**: mantiene las 4 orientaciones y las 4
    distancias a POI que `transformar_variables` retira por defecto.

    Es deliberado: el modelo completo tiene que incluir las variables que el plan
    acaba descartando, porque el argumento del descarte es precisamente que su
    aporte medido es despreciable. Descartarlas antes de medirlas sería petición de
    principio. Delega en `transformar_variables(..., retener_poi_orientacion=True)` -- es
    la misma matriz que consumirían un stepwise o un árbol "completos" (notebook 04),
    para que este análisis y esos datasets no puedan divergir en silencio.
    """
    X = t.dummificar(t.transformar_variables(df, "lineal", params, retener_poi_orientacion=True))

    # `_transformar_planta` crea `es_sotano_bajo` **y** las dummies de `planta_cat`,
    # cuyo nivel de reference es justamente "sotano/bajo". La identidad
    #
    #     es_sotano_bajo = 1 - (planta_cat_1-4 + planta_cat_5-8 + planta_cat_9+)
    #
    # se cumple fila a fila, así que la matriz sale con rango deficiente. Eso no es un
    # detalle numérico: es lo que hacía salir a `es_sotano_bajo` con p = 0,000000 y
    # ΔR² = 0,000000 a la vez -- una columna sin dirección propia, sobre la que ningún
    # contraste significa nada -- y lo que desestabilizaba cualquier cálculo que
    # invierta X'X (bootstrap, validación cruzada). Se retira aquí, donde la
    # información sigue estando completa en las dummies de planta.
    #
    # `COLS_INCLUIR` no la usa, así que los datasets del notebook 04 no cambian.
    if "es_sotano_bajo" in X.columns and any(c.startswith("planta_cat_") for c in X.columns):
        X = X.drop(columns=["es_sotano_bajo"])

    return X


# ==============================================================================
# Mapa variable -> columnas (para operar por variable, no por dummy)
# ==============================================================================
#: Variables categóricas cuyas dummies hay que agrupar bajo un único nombre.
_PREFIJOS_CATEGORICOS = [
    "habitaciones_cat",
    "epoca_construccion",
    "planta_cat",
    "alturaedif_cat",
    "delitos_q",
]


#: Bases de una misma variable continua que solo tienen sentido juntas. `log_area` y
#: `log_area_50` son los dos tramos de la *spline* de superficie: tratarlas como dos
#: "variables" hace que la forward las separe (una entra en el paso 1 y la otra en el
#: 5) y que el titular "dos variables dan el 93,6%" suene más parsimonioso de lo que
#: es -- en realidad son superficie (a medias) y barrio.
_GRUPOS_SPLINE = {"superficie": ["log_area", "log_area_50"]}


def mapa_variables(
    columnas: list[str],
    agrupar_splines: bool = True,
) -> dict[str, list[str]]:
    """
    Agrupa las columnas de la matriz bajo el nombre de la variable de la que salen:
    `{"epoca_construccion": ["epoca_construccion_<1900", ...], "log_area": ["log_area"]}`.

    Necesario para que la selección forward avance **por variable** (añadir una
    categórica significa añadir todas sus dummies a la vez) y para que el ΔR² único
    de una categórica mida la variable entera, no una categoría suelta.

    `agrupar_splines=True` extiende el mismo principio a las bases de una continua
    (`log_area` + `log_area_50` -> `superficie`): son una sola variable partida en dos
    columnas, igual que una categórica lo está en dummies. Con `False` se recupera el
    comportamiento anterior, útil solo para comparar con las cifras antiguas.
    """
    mapa: dict[str, list[str]] = {}
    for col in columnas:
        prefijo = next((p for p in _PREFIJOS_CATEGORICOS if col.startswith(p + "_")), None)
        mapa.setdefault(prefijo or col, []).append(col)

    if agrupar_splines:
        for nombre, bases in _GRUPOS_SPLINE.items():
            presentes = [c for c in bases if c in mapa]
            if len(presentes) > 1:
                for c in presentes:
                    mapa.pop(c)
                mapa[nombre] = presentes

    return mapa


# ==============================================================================
# R² y contribución única
# ==============================================================================
def r2_ols(X: pd.DataFrame, y: pd.Series) -> float:
    """R² de un OLS sobre las columnas dadas. Devuelve 0.0 si no hay columnas."""
    if X.shape[1] == 0:
        return 0.0
    Xf = X.astype("float64").to_numpy()
    return float(LinearRegression().fit(Xf, y).score(Xf, y))


def delta_r2_unico(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Contribución única de cada grupo: R²(modelo completo) - R²(modelo sin ese grupo).

    Es la métrica central del plan de inclusión. Dos lecturas importantes:

    - **Infravalora sistemáticamente a las variables redundantes**, porque si dos
      variables comparten la señal, quitar cualquiera de ellas casi no baja el R² y
      ambas salen con ΔR² pequeño pese a que juntas expliquen mucho. `redundancia_
      respecto_a` pone número a esa redundancia en vez de dejarla como advertencia.

    - **No es que la métrica sea "lineal" y por eso castigue las formas no lineales.**
      Es aditiva, no lineal: `planta_cat`, `epoca_construccion` y `habitaciones_cat`
      entran dummificadas y saturadas (`transformar_variables_completa` -> `dummificar`),
      de modo que cualquier forma no monótona, incluida una U, ya está plenamente
      representada. Si su ΔR² es bajo, la causa es redundancia con otras variables del
      modelo, no la forma funcional.

    Devuelve también `cuota_residual` = ΔR² / (1 - R²_completo): la fracción del error
    que **queda por explicar** que recupera esa variable. Es la escala en la que hay
    que fijar cualquier umbral (ver `UMBRAL_RESIDUAL`).
    """
    r2_completo = r2_ols(X, y)
    residual = 1.0 - r2_completo

    filas = []
    for nombre, cols in grupos.items():
        restantes = [c for c in X.columns if c not in cols]
        r2_sin = r2_ols(X[restantes], y)
        delta = r2_completo - r2_sin
        filas.append({
            "variable": nombre,
            "k_columnas": len(cols),
            "r2_sin": r2_sin,
            "delta_r2_unico": delta,
            "cuota_residual": delta / residual if residual > 0 else np.nan,
        })

    return (
        pd.DataFrame(filas)
        .sort_values("delta_r2_unico", ascending=False)
        .reset_index(drop=True)
    )


# ==============================================================================
# Descomposición por bloques semánticos
# ==============================================================================
def descomposicion_bloques(
    X: pd.DataFrame,
    y: pd.Series,
    bloques: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Para cada bloque semántico: R² del bloque en solitario y ΔR² único.

    El contraste entre ambas columnas es el resultado interesante. Un bloque puede
    tener un R² alto en solitario y un ΔR² único casi nulo: significa que mide algo
    real pero que ya está contenido en otros bloques (es *proxy*, no información
    nueva).
    """
    r2_completo = r2_ols(X, y)

    filas = []
    for nombre, cols in bloques.items():
        cols_presentes = [c for c in cols if c in X.columns]
        restantes = [c for c in X.columns if c not in cols_presentes]
        filas.append({
            "bloque": nombre,
            "k": len(cols_presentes),
            "r2_bloque_solo": r2_ols(X[cols_presentes], y),
            "delta_r2_unico": r2_completo - r2_ols(X[restantes], y),
        })

    return (
        pd.DataFrame(filas)
        .sort_values("delta_r2_unico", ascending=False)
        .reset_index(drop=True)
    )


def bloques_semanticos(columnas: list[str]) -> dict[str, list[str]]:
    """
    Asignación de cada columna a su bloque semántico (plan §2). La asignación es una
    decisión de dominio, no estadística: agrupa por *qué mide* la variable, para poder
    responder "¿cuánto aporta saber el tamaño?" en vez de "¿cuánto aporta log_area?".
    """
    def con_prefijo(*prefijos):
        return [c for c in columnas
                if any(c == p or c.startswith(p + "_") for p in prefijos)]

    bloques = {
        "TAMAÑO": con_prefijo(
            "log_area", "log_area_50", "BATHNUMBER", "habitaciones_cat",
            "BUILTTYPEID_1", "BUILTTYPEID_2",
        ),
        "LOCALIZACIÓN": con_prefijo(
            "alq_mediana_eur_m2", "indice_vulnerabilidad", "log_dist_centro",
            "log_dist_castellana", "log_dist_metro", "delitos_q",
        ),
        "EDIFICIO": con_prefijo(
            "CADASTRALQUALITYID", "epoca_construccion", "planta_cat",
            "alturaedif_cat", "sin_match_catastral", "log_viviendas_edif", "HASLIFT",
        ),
        "INSTALACIONES": con_prefijo(
            "HASAIRCONDITIONING", "HASSWIMMINGPOOL", "HASDOORMAN", "HASPARKINGSPACE",
            "HASBOXROOM", "HASWARDROBE", "HASTERRACE", "HASGARDEN",
            "ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR",
        ),
        "ORIENTACIÓN": con_prefijo(*t.ORIENTACIONES_DESCARTAR),
        "POI": con_prefijo(*[f"log_{c}" for c in t.POI_DESCARTAR]),
    }
    return {k: v for k, v in bloques.items() if v}


# ==============================================================================
# Selección forward
# ==============================================================================
def seleccion_forward(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: dict[str, list[str]],
    max_pasos: int | None = None,
) -> pd.DataFrame:
    """
    Selección forward **por variable**: en cada paso entra la variable (con todas sus
    dummies de golpe) que más sube el R².

    Responde a la pregunta que de verdad importa para el TFM: *¿cuántas variables
    hacen falta?* La columna `pct_r2_completo` expresa cada paso como porcentaje del
    R² del modelo completo, que es la forma legible del resultado.
    """
    r2_completo = r2_ols(X, y)
    max_pasos = max_pasos or len(grupos)

    seleccionadas: list[str] = []
    disponibles = dict(grupos)
    filas = []

    for paso in range(1, max_pasos + 1):
        if not disponibles:
            break

        mejor_nombre, mejor_r2 = None, -np.inf
        for nombre, cols in disponibles.items():
            r2 = r2_ols(X[seleccionadas + cols], y)
            if r2 > mejor_r2:
                mejor_nombre, mejor_r2 = nombre, r2

        seleccionadas += disponibles.pop(mejor_nombre)
        filas.append({
            "paso": paso,
            "variable": mejor_nombre,
            "r2": mejor_r2,
            "pct_r2_completo": 100 * mejor_r2 / r2_completo,
        })

    return pd.DataFrame(filas)


# ==============================================================================
# Contraste de una variable candidata / de una interacción
# ==============================================================================
def contraste_incremental(
    df: pd.DataFrame,
    y: pd.Series,
    base: list[str],
    candidatas: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Aporte de cada candidata sobre un mismo modelo base, y aporte del conjunto.

    Pensada para dos usos: contrastar una variable derivada frente a la original de
    la que sale (¿aporta `es_bajo` algo que `FLOORCLEAN` no tenga ya?) y contrastar
    términos de interacción sobre sus efectos principales.
    """
    r2_base = r2_ols(df[base], y)

    filas = [{"modelo": "base", "k": len(base), "r2": r2_base, "delta_r2": np.nan}]
    for nombre, cols in candidatas.items():
        r2 = r2_ols(df[base + cols], y)
        filas.append({
            "modelo": f"+ {nombre}",
            "k": len(base) + len(cols),
            "r2": r2,
            "delta_r2": r2 - r2_base,
        })

    if len(candidatas) > 1:
        todas = [c for cols in candidatas.values() for c in cols]
        r2 = r2_ols(df[base + todas], y)
        filas.append({
            "modelo": "+ todas",
            "k": len(base) + len(todas),
            "r2": r2,
            "delta_r2": r2 - r2_base,
        })

    tabla = pd.DataFrame(filas)
    tabla["delta_r2_%"] = (tabla["delta_r2"] * 100).round(3)
    return tabla


def terminos_interaccion(
    df: pd.DataFrame,
    variable: str,
    columnas_grupo: list[str],
    prefijo: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Construye los productos de `variable` por cada columna de `columnas_grupo`.

    Devuelve el df con las columnas nuevas y la lista de nombres creados. Con una
    categórica dummificada esto genera un término por categoría, que es la forma
    correcta de dejar que el efecto de `variable` cambie de pendiente en cada nivel.
    """
    df = df.copy()
    prefijo = prefijo or f"{variable}_x"

    nuevas = []
    for col in columnas_grupo:
        nombre = f"{prefijo}_{col}"
        df[nombre] = df[variable].astype("float64") * df[col].astype("float64")
        nuevas.append(nombre)

    return df, nuevas


# ==============================================================================
# Comparación de formas funcionales (respaldo de `plan_transformaciones.md`)
# ==============================================================================
def comparar_formas_funcionales(
    df: pd.DataFrame,
    y: pd.Series,
    variables: list[str],
    base: list[str] | None = None,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Para cada variable, compara tres formas de meterla en el modelo:

    - `r2_x`     cruda
    - `r2_logx`  en `log1p`
    - `r2_bin`   binnada en `n_bins` cuantiles (techo no paramétrico: cuánta señal
                 hay si se permite *cualquier* forma, incluida una U)

    Y, si se pasa `base`, el **R² parcial**: cuánto añade la versión binnada sobre un
    modelo que ya contiene los drivers principales. Esa última columna es la que
    separa señal propia de confusión espacial.

    Regla de lectura (plan_transformaciones.md §1):
      `r2_logx >> r2_x`                    -> log
      `r2_bin >> max(r2_x, r2_logx)`       -> binning (relación no monótona o con saltos)
      los tres parecidos                   -> tal cual
      r2 parcial ~ 0 pese a r2 marginal    -> descartar (era confusión)
    """
    r2_base = r2_ols(df[base], y) if base else None

    filas = []
    for var in variables:
        x = pd.to_numeric(df[var], errors="coerce").astype("float64")

        cruda = x.to_frame(var)
        logx = np.log1p(x.clip(lower=0)).to_frame(var)
        binned = pd.get_dummies(
            pd.qcut(x, n_bins, duplicates="drop"), prefix=var, drop_first=True
        ).astype(int)

        fila = {
            "variable": var,
            "skew": float(x.skew()),
            "skew_log1p": float(np.log1p(x.clip(lower=0)).skew()),
            "r2_x": r2_ols(cruda, y),
            "r2_logx": r2_ols(logx, y),
            "r2_bin": r2_ols(binned, y),
        }

        if base:
            cols_base = [c for c in base if c != var]
            X_con = pd.concat([df[cols_base].reset_index(drop=True),
                               binned.reset_index(drop=True)], axis=1)
            fila["r2_parcial_bin"] = r2_ols(X_con, y) - r2_base

        filas.append(fila)

    tabla = pd.DataFrame(filas)
    tabla["forma_sugerida"] = [
        _sugerir_forma(f["r2_x"], f["r2_logx"], f["r2_bin"]) for f in filas
    ]
    return tabla


#: Margen relativo que debe superar el binning para preferirlo. Es exigente a
#: propósito: binnar en 20 cuantiles gasta 19 grados de libertad frente a 1, así que
#: una mejora pequeña no compensa la pérdida de parsimonia e interpretabilidad.
MARGEN_BINNING = 1.20
#: Margen relativo para preferir log sobre la variable cruda (ambas gastan 1 g.l.,
#: así que basta una mejora modesta).
MARGEN_LOG = 1.05


def _sugerir_forma(r2_x: float, r2_logx: float, r2_bin: float) -> str:
    """
    Aplica la regla de `plan_transformaciones.md` §1 con márgenes explícitos.

    Es una **sugerencia**, no la decisión final: no incorpora parsimonia más allá del
    margen, ni criterios de interpretabilidad o de despliegue. Donde la decisión del
    plan se aparta de esta sugerencia, el notebook 02 lo razona caso a caso.
    """
    mejor_parametrica = max(r2_x, r2_logx)
    if mejor_parametrica > 0 and r2_bin / mejor_parametrica >= MARGEN_BINNING:
        return "binning"
    if r2_x > 0 and r2_logx / r2_x >= MARGEN_LOG:
        return "log"
    return "tal cual"


# ==============================================================================
# Umbral: en cuota de varianza residual, no de varianza total
# ==============================================================================
UMBRAL_RESIDUAL = 0.01
"""
Umbral de inclusión por ajuste: **1% de la varianza residual** del modelo contra el
que se mide, es decir ΔR² / (1 − R²_base) >= 0,01.

Sustituye al antiguo umbral de 0,005 *de R² total*, que estaba expresado en el
denominador equivocado. Con R²_base ≈ 0,912 solo queda un ~8,8% de varianza por
explicar, así que aquel 0,5% del total exigía en realidad un **~5,7% de todo el error
restante** a una única variable -- un listón tan alto que solo lo superaban 3 de las 39
variables de entonces, y por eso hubo que saltárselo en 15 de las 18 inclusiones. Un
criterio que se incumple en la mayoría de los casos en los que se aplica no es el
criterio real.

En cuota residual el 1% deja pasar las variables que de verdad mueven el ajuste y deja
fuera el resto sin necesidad de excepciones narrativas. Sigue siendo una convención:
lo que la justifica no es el número, sino que por debajo de él los intervalos
*bootstrap* de `bootstrap_delta_r2` empiezan a solaparse entre sí y con el cero, de
modo que el orden deja de ser distinguible del ruido muestral.
"""


def cuota_residual(delta_r2: float, r2_base: float) -> float:
    """ΔR² expresado como fracción de la varianza que el modelo base aún no explica."""
    residual = 1.0 - r2_base
    return delta_r2 / residual if residual > 0 else float("nan")


def veredicto(delta_r2: float, r2_base: float, umbral: float = UMBRAL_RESIDUAL) -> str:
    """
    Veredicto **de ajuste**, en tres niveles y no en dos.

    El binario INCLUIR/descartar anterior forzaba a que el número decidiera siempre,
    cuando en la mayoría de las variables no decide: el ajuste solo separa con
    claridad en la cola alta. Aquí, por debajo del umbral, el veredicto explícito es
    que *el ajuste no decide* -- y la decisión pasa a los criterios estructurales
    (§4: resolución espacial, origen declarativo, disponibilidad en despliegue,
    categoría no cerrada), que es lo que de hecho ocurría.
    """
    cuota = cuota_residual(delta_r2, r2_base)
    if cuota >= umbral:
        return "incluir (ajuste)"
    if cuota >= umbral / 4:
        return "el ajuste no decide"
    return "descartar (ajuste)"


# ==============================================================================
# Incertidumbre y validación fuera de muestra del ΔR² único
# ==============================================================================
# Las tres funciones de esta sección comparten un atajo algebraico. Para un OLS con
# matriz de diseño X (con intercepto), la suma de cuadrados que aporta un grupo de
# columnas g **entrando en último lugar** es
#
#     SSR_g = b_g' (A_gg)^-1 b_g,     con A = (X'X)^-1  y  b = A X'y
#
# -- el numerador del contraste F para H0: beta_g = 0. Es decir: basta una única
# factorización de X'X para obtener el ΔR² de *todas* las variables, en vez de
# reajustar un modelo por variable. Eso es lo que hace viable repetir el cálculo
# completo cientos de veces en un bootstrap: el coste pasa de 40 ajustes por réplica
# a uno.
def _gram(X: pd.DataFrame, y: pd.Series):
    """Devuelve (X'X, X'y, TSS, índice columna->posición) con intercepto en la posición 0."""
    Xf = X.astype("float64").to_numpy()
    Xa = np.column_stack([np.ones(len(Xf)), Xf])
    yv = np.asarray(y, dtype="float64")
    tss = float(((yv - yv.mean()) ** 2).sum())
    return Xa.T @ Xa, Xa.T @ yv, tss, {c: i + 1 for i, c in enumerate(X.columns)}


#: Corte relativo de valores singulares por debajo del cual una dirección se
#: considera nula. Sin él, una columna casi dependiente produce coeficientes
#: astronómicos que se cancelan dentro de muestra y estallan fuera de ella.
_RCOND = 1e-10


def _resolver(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve A x = b truncando las direcciones de rango deficiente.

    Se usa `lstsq` con `rcond` explícito en lugar de `solve` a propósito: las matrices
    de este módulo pueden ser casi singulares (dummies + su nivel de referencia,
    *splines* de la misma variable), y ahí `solve` no falla -- devuelve una solución
    numéricamente basura que solo se nota al predecir fuera de muestra.
    """
    return np.linalg.lstsq(A, b, rcond=_RCOND)[0]



def _deltas_desde_gram(
    XtX: np.ndarray, Xty: np.ndarray, tss: float,
    posicion: dict[str, int], grupos: dict[str, list[str]],
) -> dict[str, float]:
    beta = _resolver(XtX, Xty)
    A = np.linalg.pinv(XtX)

    deltas = {}
    for nombre, cols in grupos.items():
        j = [posicion[c] for c in cols if c in posicion]
        if not j:
            deltas[nombre] = 0.0
            continue
        # pinv y no _resolver: si el bloque es casi singular, la pseudoinversa anula
        # esa dirección y el ΔR² sale infravalorado -- conservador, no explosivo.
        ssr = float(beta[j] @ np.linalg.pinv(A[np.ix_(j, j)], rcond=1e-10) @ beta[j])
        deltas[nombre] = max(ssr / tss, 0.0)
    return deltas


def delta_r2_rapido(
    X: pd.DataFrame, y: pd.Series, grupos: dict[str, list[str]],
) -> pd.Series:
    """
    Mismo ΔR² único que `delta_r2_unico`, por la vía algebraica de arriba.

    Existe para el bootstrap, no para sustituir a la función legible. El notebook 03
    comprueba que ambas coinciden antes de usarla: un atajo que no se verifica es un
    sitio estupendo donde esconder un error.
    """
    XtX, Xty, tss, posicion = _gram(X, y)
    return pd.Series(_deltas_desde_gram(XtX, Xty, tss, posicion, grupos))


def bootstrap_delta_r2(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: dict[str, list[str]],
    n_boot: int = 300,
    semilla: int = 42,
    nivel: float = 0.95,
) -> pd.DataFrame:
    """
    Intervalo percentil del ΔR² único por *bootstrap* de filas.

    Responde a la objeción que el criterio de tamaño del efecto tenía abierta: se
    rechaza el p-valor por saturado, pero luego se ordenan variables comparando la
    cuarta cifra decimal de un R² sin ningún intervalo. Rechazar el contraste de
    significación no elimina el error muestral -- solo retira el único instrumento
    que lo estimaba.

    La columna `separado_de_cero` es deliberadamente conservadora: marca solo las
    variables cuyo percentil inferior queda por encima de una milésima de la varianza
    residual. Por debajo de eso, "0,00009 frente a 0,00025" no es un orden: es ruido.
    """
    Xf = X.astype("float64").to_numpy()
    yv = np.asarray(y, dtype="float64")
    posicion = {c: i + 1 for i, c in enumerate(X.columns)}
    n = len(Xf)
    rng = np.random.default_rng(semilla)

    muestras = []
    for _ in range(n_boot):
        filas = rng.integers(0, n, n)
        Xb = np.column_stack([np.ones(n), Xf[filas]])
        yb = yv[filas]
        tss = float(((yb - yb.mean()) ** 2).sum())
        muestras.append(_deltas_desde_gram(Xb.T @ Xb, Xb.T @ yb, tss, posicion, grupos))

    dist = pd.DataFrame(muestras)
    alfa = (1 - nivel) / 2
    residual = 1.0 - r2_ols(X, y)

    tabla = pd.DataFrame({
        "variable": dist.columns,
        "delta_r2_medio": dist.mean().to_numpy(),
        "ic_bajo": dist.quantile(alfa).to_numpy(),
        "ic_alto": dist.quantile(1 - alfa).to_numpy(),
    })
    tabla["cuota_residual_ic_bajo"] = tabla["ic_bajo"] / residual
    tabla["separado_de_cero"] = tabla["ic_bajo"] > 0.001 * residual
    return tabla.sort_values("delta_r2_medio", ascending=False).reset_index(drop=True)


def delta_r2_cv(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: dict[str, list[str]],
    n_splits: int = 5,
    semilla: int = 42,
) -> pd.DataFrame:
    """
    ΔR² único medido **fuera de muestra**: R²_cv(completo) - R²_cv(sin la variable),
    con predicciones *out-of-fold* sobre una K-fold barajada.

    Corrige el defecto más serio de la versión dentro de muestra: allí el R² no puede
    bajar al añadir columnas, así que el ΔR² es >= 0 **por construcción** y la métrica
    solo sabe decir "positivo pero pequeño" -- nunca puede aportar evidencia *en
    contra* de una variable, y dónde se corta queda como parámetro libre. Fuera de
    muestra un ΔR² negativo significa que la variable empeora la predicción, y el cero
    deja de ser una convención.

    Coste: la matriz de Gram se calcula una vez por *fold* y cada modelo reducido sale
    de un submenor suyo, así que la K-fold entera cuesta aproximadamente lo mismo que
    un puñado de ajustes.
    """
    from sklearn.model_selection import KFold

    Xf = X.astype("float64").to_numpy()
    yv = np.asarray(y, dtype="float64")
    n, p = Xf.shape
    posicion = {c: i + 1 for i, c in enumerate(X.columns)}
    todas = list(range(p + 1))

    nombres = ["__completo__"] + list(grupos)
    pred = {nombre: np.empty(n) for nombre in nombres}

    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=semilla).split(Xf):
        Xtr = np.column_stack([np.ones(len(tr)), Xf[tr]])
        Xte = np.column_stack([np.ones(len(te)), Xf[te]])
        XtX, Xty = Xtr.T @ Xtr, Xtr.T @ yv[tr]

        for nombre in nombres:
            if nombre == "__completo__":
                j = todas
            else:
                fuera = {posicion[c] for c in grupos[nombre] if c in posicion}
                j = [k for k in todas if k not in fuera]
            beta = _resolver(XtX[np.ix_(j, j)], Xty[j])
            pred[nombre][te] = Xte[:, j] @ beta

    tss = float(((yv - yv.mean()) ** 2).sum())
    r2_cv = {nombre: 1 - float(((yv - v) ** 2).sum()) / tss for nombre, v in pred.items()}
    r2_completo_cv = r2_cv.pop("__completo__")
    residual = 1.0 - r2_completo_cv

    tabla = pd.DataFrame({
        "variable": list(r2_cv),
        "r2_cv_sin": list(r2_cv.values()),
    })
    tabla["delta_r2_cv"] = r2_completo_cv - tabla["r2_cv_sin"]
    tabla["cuota_residual_cv"] = tabla["delta_r2_cv"] / residual
    tabla["empeora_fuera_de_muestra"] = tabla["delta_r2_cv"] < 0
    tabla.attrs["r2_cv_completo"] = r2_completo_cv
    return tabla.sort_values("delta_r2_cv", ascending=False).reset_index(drop=True)


# ==============================================================================
# Redundancia: qué parte de una variable ya está contenida en las incluidas
# ==============================================================================
def redundancia_respecto_a(
    X: pd.DataFrame,
    grupos: dict[str, list[str]],
    columnas_referencia: list[str],
) -> pd.DataFrame:
    """
    Para cada grupo, R² de predecir **sus propias columnas** a partir del conjunto de
    referencia (las incluidas): cuánto de esa variable ya está contenido en lo que el
    modelo se queda.

    Pone número a la única advertencia del plan que se declaraba y luego no se
    operacionalizaba. Decir "`HASDOORMAN` es *proxy* de `HASLIFT`" y descartarla por
    ΔR² bajo es circular mientras la redundancia no se mida: un ΔR² pequeño **con**
    redundancia alta significa "esto ya lo tengo"; un ΔR² pequeño **sin** redundancia
    significa "esto no aporta". Son dos descartes distintos y piden justificaciones
    distintas en la memoria.
    """
    ref = [c for c in columnas_referencia if c in X.columns]
    Xr = np.column_stack([np.ones(len(X)), X[ref].astype("float64").to_numpy()])
    XtX = Xr.T @ Xr

    filas = []
    for nombre, cols in grupos.items():
        objetivo = [c for c in cols if c in X.columns and c not in ref]
        if not objetivo:
            filas.append({"variable": nombre, "k_columnas": len(cols),
                          "r2_explicada_por_referencia": np.nan})
            continue

        r2s = []
        for c in objetivo:
            z = X[c].astype("float64").to_numpy()
            beta = _resolver(XtX, Xr.T @ z)
            sst = float(((z - z.mean()) ** 2).sum())
            sse = float(((z - Xr @ beta) ** 2).sum())
            r2s.append(1 - sse / sst if sst > 0 else np.nan)

        filas.append({"variable": nombre, "k_columnas": len(cols),
                      "r2_explicada_por_referencia": float(np.nanmax(r2s))})

    return (
        pd.DataFrame(filas)
        .sort_values("r2_explicada_por_referencia", ascending=False)
        .reset_index(drop=True)
    )


# ==============================================================================
# Stepwise (AIC / BIC): un criterio de parada real, no impuesto a mano
# ==============================================================================
def seleccion_stepwise(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: dict[str, list[str]],
    criterio: str = "bic",
    max_pasos: int | None = None,
) -> pd.DataFrame:
    """
    Selección forward **por variable**, con parada por AIC o BIC en vez de por un
    umbral de ΔR² elegido a mano.

    Es la comprobación que faltaba: `seleccion_forward` (§3 del notebook) ordena
    TODAS las variables por cuánto suben el R², sin ningún criterio de parada -- no
    puede tener uno, porque el R² dentro de muestra nunca baja. El umbral de 1% de
    varianza residual que decide el resto del análisis es una convención razonada,
    pero sigue siendo una convención elegida por el analista. AIC/BIC penalizan la
    verosimilitud por el número de parámetros y dan una regla de parada que sale del
    propio criterio: el número final de variables es una salida del algoritmo, no una
    elección.

    Con AIC (penalización fija de 2 por parámetro) y n ≈ 60.000, el resultado
    esperado es que retenga casi todo -- 2 es una penalización trivial frente a la
    mejora de verosimilitud de cualquier variable con señal real a este tamaño
    muestral, así que reproduce el mismo fenómeno de saturación que ya se documentó
    con p < 0,001 (§1), solo que con otro nombre. BIC penaliza k·ln(n) ≈ 11·k con
    este n, un filtro bastante más exigente, y es la comparación que aporta algo
    nuevo de verdad frente al umbral de cuota residual.

    Implementación: entra en cada paso la variable que más *reduce* el criterio; para
    cuando ninguna variable disponible lo reduce más. AIC/BIC se calculan a partir de
    la RSS (no hace falta reajustar con statsmodels para tener la verosimilitud):
    con errores gaussianos, AIC = n·ln(RSS/n) + 2(k+1), BIC = n·ln(RSS/n) + (k+1)·ln(n),
    contando el intercepto en k+1.
    """
    if criterio not in {"aic", "bic"}:
        raise ValueError(f"criterio debe ser 'aic' o 'bic', recibido: {criterio!r}")

    n = len(y)
    yv = np.asarray(y, dtype="float64")
    tss = float(((yv - yv.mean()) ** 2).sum())
    max_pasos = max_pasos or len(grupos)

    def valor_criterio(k: int, r2: float) -> float:
        rss = max(tss * (1 - r2), 1e-12)
        base = n * np.log(rss / n)
        return base + 2 * (k + 1) if criterio == "aic" else base + (k + 1) * np.log(n)

    seleccionadas: list[str] = []
    disponibles = dict(grupos)
    valor_actual = valor_criterio(0, 0.0)  # modelo con solo el intercepto

    filas = [{"paso": 0, "variable": None, "k": 0, "r2": 0.0, criterio: valor_actual}]

    for paso in range(1, max_pasos + 1):
        if not disponibles:
            break

        mejor_nombre, mejor_valor, mejor_r2 = None, valor_actual, None
        cols_actuales = [c for v in seleccionadas for c in grupos[v]]
        for nombre, cols in disponibles.items():
            r2 = r2_ols(X[cols_actuales + cols], y)
            valor = valor_criterio(len(cols_actuales) + len(cols), r2)
            if valor < mejor_valor:
                mejor_nombre, mejor_valor, mejor_r2 = nombre, valor, r2

        if mejor_nombre is None:
            break  # ninguna variable disponible mejora el criterio: aquí para

        seleccionadas.append(mejor_nombre)
        disponibles.pop(mejor_nombre)
        valor_actual = mejor_valor
        filas.append({"paso": paso, "variable": mejor_nombre,
                      "k": len(cols_actuales) + len(grupos[mejor_nombre]),
                      "r2": mejor_r2, criterio: mejor_valor})

    tabla = pd.DataFrame(filas)
    tabla.attrs["seleccionadas"] = seleccionadas
    tabla.attrs["no_seleccionadas"] = [v for v in grupos if v not in seleccionadas]
    return tabla
