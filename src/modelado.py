"""
Utilidades compartidas por los notebooks de modelado (05-07).
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.utils import check_random_state

from src import rutas
from src.analisis_seleccion import mapa_variables, seleccion_stepwise

RUTA_REGISTRO = rutas.RESULTADOS / "resultados_experimentos.csv"

SEMILLA = 42


class RegresionStepwiseBIC(BaseEstimator, RegressorMixin):
    """
    OLS sobre el subconjunto de variables que selecciona forward-stepwise con
    parada por BIC (notebook 03 §8) -- pero reajustado dentro de cada llamada a
    `fit`, no sobre la lista de 27 variables que ya fijó ese notebook.

    Por qué hace falta reajustar y no basta con esa lista: se decidió una vez,
    sobre TODO train. Usarla tal cual dentro de una validación cruzada
    reutilizaría en la selección de variables la misma porción de datos que
    luego actúa como validación de cada fold -- el mismo problema de fuga por
    el que notebook 04 descarta partir `stepwise` de `incluido`. Reajustar el
    stepwise en cada `fit` (una vez por fold) es lo que convierte esto en una
    comparación válida frente a `incluido`/`desplegable`.

    A cambio, el conjunto de variables seleccionado puede -- y en la práctica
    lo hace -- variar de un fold a otro. Eso no es un fallo de la
    implementación: es la varianza real del método, y es precisamente lo que
    una comparación honesta tiene que dejar ver. `columnas_seleccionadas_` y
    `variables_seleccionadas_` quedan expuestas tras `fit` para poder
    inspeccionar esa variación entre folds.

    Pensado para `X_train_lineal_completo` (todas las variables candidatas,
    incluidas las 4 distancias a POI y las 4 orientaciones que `incluido`
    descarta por defecto) -- partir de `incluido` sería seleccionar dos veces
    con el mismo criterio.
    """

    def __init__(self, criterio: str = "bic", agrupar_splines: bool = True):
        self.criterio = criterio
        self.agrupar_splines = agrupar_splines

    def fit(self, X: pd.DataFrame, y):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = pd.Series(np.asarray(y, dtype="float64"), index=X.index)

        mapa = mapa_variables(X.columns.tolist(), agrupar_splines=self.agrupar_splines)
        tabla = seleccion_stepwise(X, y, mapa, criterio=self.criterio)

        self.variables_seleccionadas_ = list(tabla.attrs["seleccionadas"])
        self.columnas_seleccionadas_ = [
            c for v in self.variables_seleccionadas_ for c in mapa[v]
        ]

        if not self.columnas_seleccionadas_:
            # BIC no mejora sobre el intercepto solo: modelo trivial.
            self.modelo_ = None
            self.media_y_ = float(y.mean())
        else:
            self.modelo_ = LinearRegression()
            self.modelo_.fit(X[self.columnas_seleccionadas_], y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.modelo_ is None:
            return np.full(len(X), self.media_y_)
        return self.modelo_.predict(X[self.columnas_seleccionadas_])


# ==============================================================================
# Validación cruzada: KFold aleatorio -- mismo protocolo en 05 y 06, sin
# agrupar por geografía (no hay una unidad de agrupación natural por encima de
# la fila individual en este dataset: cada activo es su propia observación).
# ==============================================================================
def cv_aleatorio(n_splits: int = 5, semilla: int = SEMILLA) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=semilla)


# ==============================================================================
# Espacios de búsqueda: "no usar la palanca" siempre es una opción.
# ==============================================================================
class ConApagado:
    """
    Distribución para `RandomizedSearchCV` que, con probabilidad `p_apagado`,
    devuelve `valor_apagado` (el valor que desactiva el hiperparámetro -- p. ej.
    `min_impurity_decrease=0`, `gamma=0`, `subsample=1.0`) y, si no, muestrea
    de `distribucion`.

    Hace falta porque las distribuciones continuas de `scipy` (`loguniform`,
    `uniform`) nunca devuelven exactamente ese valor: un espacio construido
    solo con ellas obliga a que *todas* las configuraciones usen la palanca, y
    la búsqueda no puede descubrir que lo mejor era no usarla.
    """

    def __init__(self, distribucion, valor_apagado=0.0, p_apagado: float = 0.25):
        self.distribucion = distribucion
        self.valor_apagado = valor_apagado
        self.p_apagado = p_apagado

    def rvs(self, random_state=None):
        rng = check_random_state(random_state)
        if rng.uniform() < self.p_apagado:
            return self.valor_apagado
        return self.distribucion.rvs(random_state=rng)

    def __repr__(self) -> str:
        return f"ConApagado({self.distribucion!r}, valor_apagado={self.valor_apagado}, p_apagado={self.p_apagado})"


# ==============================================================================
# Refinamiento de espacios de búsqueda: construir el "espacio 2" de una
# búsqueda en dos etapas centrado en el ganador de la etapa 1, sin perder la
# opción de apagar la palanca. Compartido por Random Forest y XGBoost en
# `06_fine_tuning.ipynb` -- el ganador de la primera etapa es lo que decide el
# centro; estas funciones solo deciden cuánto de ancho es "cerca de él".
# ==============================================================================
def rango_log(centro: float, factor: float = 5, lo: float = 1e-8, hi: float = 100.0):
    """loguniform entre centro/factor y centro*factor, recortado a [lo, hi]."""
    return loguniform(max(lo, centro / factor), min(hi, centro * factor))


def rango_uniforme(centro: float, ancho: float, lo: float = 0.0, hi: float = 1.0):
    """uniform en [centro-ancho, centro+ancho], recortado a [lo, hi]."""
    a = max(lo, centro - ancho)
    b = min(hi, centro + ancho)
    return uniform(a, b - a)


def rango_entero(centro: int, radio: int, lo: int = 1, hi: int | None = None):
    """randint en [centro-radio, centro+radio], recortado a [lo, hi]."""
    a = max(lo, centro - radio)
    b = centro + radio if hi is None else min(hi, centro + radio)
    return randint(a, b + 1)


def vecinos_lista(valores: list, centro, radio: int = 1) -> list:
    """
    Subconjunto de `valores` centrado en el elemento `centro`: él mismo, más
    `radio` vecinos a cada lado según el orden de la propia lista original.

    Para hiperparámetros que no son un rango numérico -- `max_features`
    mezcla `"sqrt"` con floats, `max_depth`/`max_samples` incluyen `None` --
    no se puede construir un `loguniform`/`uniform` alrededor del ganador.
    Tomar sus vecinos en la lista ya declarada (§3) hace lo mismo con las
    mismas piezas: nunca inventa un valor que la búsqueda amplia no
    contemplaba, y sigue centrando el refinamiento en lo que de verdad ganó,
    no en el *default* a ciegas.
    """
    if centro not in valores:
        return valores  # el ganador no vino de esta lista -- no se puede acotar, se deja tal cual
    i = valores.index(centro)
    vecinos = valores[max(0, i - radio):min(len(valores), i + radio + 1)]
    return vecinos if len(vecinos) > 1 else valores


def refinar_con_apagado(original: ConApagado, centro, rango_refinado) -> ConApagado | list:
    """
    Estrecha un `ConApagado` alrededor de `centro` (el valor ganador de la
    etapa 1) -- salvo que la etapa 1 ya haya decidido apagar la palanca.

    Si `centro` es el propio valor de apagado, la etapa 1 ya respondió la
    pregunta "¿usar esta palanca?" con un no. La etapa 2 no vuelve a
    plantearla: se fija al valor de apagado (`[original.valor_apagado]`,
    una lista de un solo elemento -- `RandomizedSearchCV` la trata como
    constante, nunca la muestrea de otra forma). Reabrir la pregunta en el
    refinamiento gastaría presupuesto de búsqueda en volver a probar algo
    que la etapa amplia, con más margen para explorar, ya descartó.

    Si no, se aplica `rango_refinado(centro)` -- una llamada a `rango_log`,
    `rango_uniforme` o `rango_entero` -- para estrechar alrededor del
    ganador, conservando la probabilidad de apagado y el valor de apagado de
    la etapa 1: la palanca sigue pudiendo apagarse en la etapa 2, solo que
    ahora alrededor de un centro distinto de cero.
    """
    if centro == original.valor_apagado:
        return [original.valor_apagado]
    return ConApagado(rango_refinado(centro), original.valor_apagado, p_apagado=original.p_apagado)


# ==============================================================================
# Evaluación y registro de experimentos: una fila por combinación
# (familia, variante de dataset, modelo) -- la tabla que compara 05 y 06.
# ==============================================================================
def evaluar_cv(
    modelo,
    X: pd.DataFrame,
    y: pd.Series,
    nombre: str,
    familia: str,
    n_splits: int = 5,
    **params_extra,
) -> dict:
    """
    CV con `KFold` aleatorio. Devuelve una fila lista para
    `registrar_experimento`: media y desviación de `rmse_log` entre folds
    (nunca un único número -- el notebook 03 ya advirtió contra ordenar por la
    cuarta cifra decimal sin banda de incertidumbre) más el tiempo total.
    """
    cv = cv_aleatorio(n_splits)
    t0 = perf_counter()
    resultado = cross_validate(
        modelo, X, y, cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    tiempo = perf_counter() - t0
    rmse_folds = -resultado["test_score"]

    return {
        "nombre": nombre,
        "familia": familia,
        "n_columnas": X.shape[1],
        "rmse_log_media": float(rmse_folds.mean()),
        "rmse_log_std": float(rmse_folds.std()),
        "rmse_log_folds": [float(round(v, 4)) for v in rmse_folds],
        "tiempo_s": round(tiempo, 1),
        **params_extra,
    }


def registrar_experimento(fila: dict, ruta: str | Path = RUTA_REGISTRO) -> None:
    """Añade una fila al registro de experimentos (crea el fichero si no existe)."""
    df = pd.DataFrame([fila])
    ruta = Path(ruta)
    if ruta.exists():
        pd.concat([pd.read_csv(ruta), df], ignore_index=True).to_csv(ruta, index=False)
    else:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False)
