"""
Funciones auxiliares de EDA para el dataset de vivienda de Madrid (Habitia 2018).

Adaptadas de trabajos previos del máster y generalizadas para funcionar con los dtypes
enteros anulables (`Int32`) del dataset de origen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ==============================================================================
# Descriptivos
# ==============================================================================
def descriptivos_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """Describe + skew + kurtosis + rango + % de nulos para las columnas numéricas de df."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    desc = df[numeric_cols].describe().T
    desc["skew"] = df[numeric_cols].skew(numeric_only=True)
    desc["kurtosis"] = df[numeric_cols].kurtosis(numeric_only=True)
    desc["range"] = df[numeric_cols].max() - df[numeric_cols].min()
    desc["missing_pct"] = df[numeric_cols].isna().mean() * 100
    return desc


def descriptivos_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """Frecuencias por variable categórica (object/bool/category); imprime y devuelve el resumen."""
    cat_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    rows = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False, normalize=True)
        rows.append({
            "var": c,
            "n_categories": vc.shape[0],
            "top_category": vc.idxmax(),
            "top_pct": vc.max() / vc.sum() * 100,
        })
        print(f"Frecuencias para variable categórica '{c}':\n{vc}\n")
    resumen = pd.DataFrame(rows)
    print(resumen)
    return resumen


def histogramas_numericos(df, grid: tuple = (3, 3), figsize: tuple = (27, 21)) -> None:
    """
    Histogramas descriptivos para variables numéricas: bins Freedman-Diaconis,
    media/mediana marcadas, skew y % de nulos en el título.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    cols = df.select_dtypes(include="number").columns.tolist()

    rows, cols_grid = grid
    max_plots = rows * cols_grid
    cols = cols[:max_plots]

    fig, axes = plt.subplots(rows, cols_grid, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna().astype("float64")

        if s.empty:
            ax.axis("off")
            continue

        q75, q25 = np.percentile(s, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            bin_width = 2 * iqr * (len(s) ** (-1 / 3))
            bins = int((s.max() - s.min()) / bin_width) if bin_width > 0 else 10
            bins = max(10, min(bins, 40))
        else:
            bins = 10

        mean = s.mean()
        median = s.median()
        skew = s.skew()
        missing_pct = df[col].isna().mean() * 100

        ax.hist(s, bins=bins, edgecolor="black", alpha=0.8)
        ax.axvline(mean, linestyle="--", linewidth=2, label="Media")
        ax.axvline(median, linestyle=":", linewidth=2, label="Mediana")
        ax.set_title(f"{col}\nSkew={skew:.2f} | Miss={missing_pct:.1f}% | n={len(s)}", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(fontsize=9)

    for ax in axes[len(cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de nulos (n y %) por columna, ordenada de mayor a menor."""
    return pd.DataFrame({
        "missing_n": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "dtype": df.dtypes.astype(str),
    }).sort_values("missing_pct", ascending=False)


# ==============================================================================
# Correlaciones y colinealidad numéricas
# ==============================================================================
def correlaciones_numericas(
    df: pd.DataFrame,
    target: str,
    metodo: str = "spearman",
    min_abs_corr: float = 0.0,
) -> pd.DataFrame:
    """Correlación de target contra el resto de variables numéricas, ordenada por |corr|."""
    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = numeric_cols.drop(target)

    corr = (
        df[numeric_cols]
        .corrwith(df[target], method=metodo)
        .dropna()
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )

    corr_df = corr.to_frame(name=f"corr_{metodo}")
    corr_df["abs_corr"] = corr_df[f"corr_{metodo}"].abs()

    return corr_df[corr_df["abs_corr"] >= min_abs_corr]


def plot_corr_matrix(df: pd.DataFrame, num_cols: list[str], method: str = "pearson") -> None:
    """Heatmap de la matriz de correlaciones de num_cols."""
    corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr(method=method)

    plt.figure(figsize=(15, 10))
    plt.title(f"Matriz de correlaciones ({method})")
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.show()


def vif_table(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """
    Variance Inflation Factor de num_cols (statsmodels). Descarta filas con nulos y
    columnas degeneradas (varianza nula), reportándolas en vez de fallar.
    """
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna().astype("float64")

    valid_cols = [c for c in X.columns if X[c].std(ddof=0) > 0]
    dropped = sorted(set(num_cols) - set(valid_cols))
    if dropped:
        print(f"vif_table: columnas excluidas por varianza nula/degeneradas: {dropped}")

    X = add_constant(X[valid_cols])

    rows = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        rows.append({"variable": col, "VIF": vif})

    vif_df = pd.DataFrame(rows)
    vif_df = vif_df[vif_df["variable"] != "const"].sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


# ==============================================================================
# V de Cramer / redundancia categórica
# ==============================================================================
def _preparar_para_cramer(s: pd.Series, max_categorias_discretas: int = 10) -> pd.Series:
    """Discretiza en quantiles las variables numéricas con muchos valores distintos
    (continuas); el resto se trata tal cual. Los nulos pasan a la categoría 'NA'."""
    s = s.copy()
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > max_categorias_discretas:
        s = pd.to_numeric(s, errors="coerce")
        cortes = sorted(set(s.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).dropna()))
        if len(cortes) >= 2:
            s = pd.cut(s, bins=cortes, include_lowest=True)
    s = s.astype("object")
    return s.where(pd.notna(s), "NA")


def Vcramer(v: pd.Series, target: pd.Series) -> float:
    """
    Coeficiente V de Cramer entre dos variables. Las variables numéricas continuas
    (>10 valores distintos) se discretizan en quantiles; el resto se usa tal cual.
    """
    v = _preparar_para_cramer(v).reset_index(drop=True)
    target = _preparar_para_cramer(target).reset_index(drop=True)

    tabla_cruzada = pd.crosstab(v, target)

    grados_libertad = min(tabla_cruzada.shape) - 1
    n = tabla_cruzada.sum().sum()
    if grados_libertad <= 0 or n == 0:
        return np.nan

    chi2 = chi2_contingency(tabla_cruzada)[0]
    return float(np.sqrt(chi2 / (n * grados_libertad)))


def cramers_v_matrix(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Matriz simétrica de V de Cramer para un conjunto de variables."""
    v_matrix = pd.DataFrame(
        np.zeros((len(cat_cols), len(cat_cols))),
        index=cat_cols,
        columns=cat_cols,
    )

    for i, col_i in enumerate(cat_cols):
        for j, col_j in enumerate(cat_cols):
            if i <= j:
                v = Vcramer(df[col_i], df[col_j])
                v_matrix.loc[col_i, col_j] = v
                v_matrix.loc[col_j, col_i] = v

    return v_matrix


def graficoVcramer(matriz: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Barras horizontales con la V de Cramer de cada columna de matriz frente a target."""
    salida = {x: Vcramer(matriz[x], target) for x in matriz.columns}
    sorted_data = dict(sorted(salida.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(10, max(4, 0.35 * len(sorted_data))))
    plt.barh(list(sorted_data.keys()), list(sorted_data.values()), color="skyblue")
    plt.xlabel("V de Cramer")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return pd.Series(sorted_data, name="cramers_v")


def detectar_redundancia_categorica(
    df: pd.DataFrame,
    cat_cols: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Pares de variables con V de Cramer por encima de threshold (candidatas a redundantes)."""
    resultados = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            col_i, col_j = cat_cols[i], cat_cols[j]
            v = Vcramer(df[col_i], df[col_j])
            if v >= threshold:
                resultados.append({"var_1": col_i, "var_2": col_j, "cramers_v": round(v, 3)})

    resultado_df = pd.DataFrame(resultados, columns=["var_1", "var_2", "cramers_v"])
    return resultado_df.sort_values("cramers_v", ascending=False)


# ==============================================================================
# Separación frente al target / outliers
# ==============================================================================
def evaluar_separacion_categorica(df: pd.DataFrame, target: str, cat_cols: list[str]) -> None:
    """Media/std/n de target por nivel de cada variable categórica, y diferencia vs la media global."""
    for col in cat_cols:
        resumen = df.groupby(col, observed=True)[target].agg(["mean", "std", "count"]).dropna()
        resumen["mean_diff_vs_global"] = resumen["mean"] - df[target].mean()
        print(f"--- {col} ---")
        print(resumen.sort_values("mean"), "\n")


def graficar_distribuciones_num_vs_cat(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_col: str,
    tipo_grafico: str = "boxplot",
) -> None:
    """Cuadrícula de boxplots/violinplots de cada variable numérica separada por cat_col."""
    columnas_plot = [c for c in num_cols if c in df.columns]

    n_vars = len(columnas_plot)
    n_cols_grid = min(4, n_vars) or 1
    n_rows = int(np.ceil(n_vars / n_cols_grid))

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(5 * n_cols_grid, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, col_num in enumerate(columnas_plot):
        ax = axes[i]
        if tipo_grafico == "boxplot":
            sns.boxplot(x=cat_col, y=col_num, data=df, ax=ax, hue=cat_col, palette="Set2", legend=False)
        else:
            sns.violinplot(x=cat_col, y=col_num, data=df, ax=ax, hue=cat_col, palette="Set2", legend=False)
        ax.set_title(f"{col_num} según {cat_col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for j in range(len(columnas_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def outliers_iqr_flags(s: pd.Series, k: float = 1.5) -> tuple[pd.Series, pd.Series, pd.Series]:
    s = pd.to_numeric(s, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi), (s < lo), (s > hi)


def outliers_zscore_flags(s: pd.Series, z: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=s.index)
    return ((s - mu) / sd).abs() > z


def summarize_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n = int(s.shape[0])
        iqr_flags, iqr_neg, iqr_pos = outliers_iqr_flags(s)
        rows.append({
            "var": c,
            "iqr_outliers_%": round(int(iqr_flags.sum()) / n * 100, 2),
            "iqr_outliers_%_neg": round(int(iqr_neg.sum()) / n * 100, 2),
            "iqr_outliers_%_pos": round(int(iqr_pos.sum()) / n * 100, 2),
            "z_outliers_%": round(int(outliers_zscore_flags(s).sum()) / n * 100, 2),
            "total_n": n,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# Media del objetivo por grupo (para justificar binning/no-linealidad)
# ==============================================================================
def mean_por_grupo(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    order: list | None = None,
    figsize: tuple = (8, 4),
    color: str = "steelblue",
    ax=None,
):
    """
    Barras de la media de `target` por nivel de `group_col`, con el tamaño de cada
    grupo anotado encima. Pensada para enseñar de un vistazo si una relación es
    monótona, en U, o tiene saltos -- el argumento de fondo de casi todas las
    decisiones de binning del plan de transformaciones.
    """
    resumen = df.groupby(group_col, observed=True)[target].agg(["mean", "count"])
    if order is not None:
        resumen = resumen.reindex(order)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.bar(resumen.index.astype(str), resumen["mean"], color=color, edgecolor="black")
    for i, (m, n) in enumerate(zip(resumen["mean"], resumen["count"])):
        if pd.notna(m):
            ax.text(i, m, f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"{target} medio")
    ax.set_xlabel(group_col)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    return resumen
