# ==========================================================
# toolbox_ML.py
# Team Challenge - Toolbox ML
# Grupo # 4
# Módulo de funciones para análisis y selección de features
# Enfoque principal: regresión (target numérico)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, f_oneway, kruskal


def describe_df(df):
    """
    Genera un resumen del DataFrame mostrando, para cada variable:
    - Tipo de dato
    - Porcentaje de valores nulos
    - Número de valores únicos
    - Porcentaje de cardinalidad

    Argumentos:
    df (pd.DataFrame): DataFrame a analizar.

    Retorna:
    pd.DataFrame: DataFrame con una columna por cada variable y filas:
        - DATA_TYPE
        - MISSING(%)
        - UNIQUE_VALUES
        - CARDIN (%)
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df no es un DataFrame de pandas.")
        return None

    resumen = pd.DataFrame({
        "DATA_TYPE": df.dtypes,
        "MISSING(%)": df.isnull().mean() * 100,
        "UNIQUE_VALUES": df.nunique(dropna=True),
        "CARDIN (%)": (df.nunique(dropna=True) / len(df)) * 100
    }).T

    return resumen


def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Sugiere el tipo de cada variable en función de su cardinalidad y porcentaje de cardinalidad.

    Reglas:
    - Si cardinalidad == 2: "Binaria"
    - Si cardinalidad < umbral_categoria: "Categórica"
    - Si cardinalidad >= umbral_categoria:
        - Si % cardinalidad >= umbral_continua: "Numerica Continua"
        - En caso contrario: "Numerica Discreta"

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    umbral_categoria (int): Umbral para considerar variable categórica.
    umbral_continua (float): Umbral de % cardinalidad para considerar continua.
                             Se interpreta como porcentaje (0-100).

    Retorna:
    pd.DataFrame: DataFrame con columnas:
        - nombre_variable
        - tipo_sugerido
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df no es un DataFrame de pandas.")
        return None

    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        print("Error: umbral_categoria debe ser un entero positivo.")
        return None

    if not isinstance(umbral_continua, (int, float)) or not (0 < float(umbral_continua) <= 100):
        print("Error: umbral_continua debe ser un número entre 0 y 100.")
        return None

    umbral_continua = float(umbral_continua)
    n_filas = len(df)

    resultados = []

    for col in df.columns:
        cardinalidad = df[col].nunique(dropna=True)
        pct_card = (cardinalidad / n_filas) * 100

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        else:
            if pct_card >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"

        resultados.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultados)


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve una lista de variables numéricas cuya correlación con target_col sea superior
    (en valor absoluto) a umbral_corr.

    Si pvalue no es None, además filtra por significación estadística usando Pearson:
    p_val <= pvalue.

    IMPORTANTE:
    - Pensado para regresión (target numérico).
    - Se excluyen variables booleanas (bool) de la selección numérica.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la variable objetivo (numérica).
    umbral_corr (float): Umbral de correlación entre 0 y 1.
    pvalue (float o None): Nivel de significación. Si None, no se filtra por p-value.

    Retorna:
    list: Lista de columnas numéricas seleccionadas.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df no es un DataFrame de pandas.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns:
        print("Error: target_col no existe en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser una variable numérica.")
        return None

    if not isinstance(umbral_corr, (int, float)) or not (0 <= float(umbral_corr) <= 1):
        print("Error: umbral_corr debe ser un número entre 0 y 1.")
        return None

    umbral_corr = float(umbral_corr)

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) < 1):
            print("Error: pvalue debe ser None o un número entre 0 y 1.")
            return None
        pvalue = float(pvalue)

    # Seleccionamos solo numéricas int/float (excluye bool)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    selected = []

    for col in numeric_cols:
        data = df[[col, target_col]].dropna()

        if len(data) < 2:
            continue

        corr, p_val = pearsonr(data[col], data[target_col])

        if abs(corr) >= umbral_corr:
            if pvalue is None or p_val <= pvalue:
                selected.append(col)

    return selected


def plot_features_num_regression(df, target_col="", columns=None, umbral_corr=0, pvalue=None):
    """
    Pinta pairplots entre target_col y variables numéricas seleccionadas por correlación.

    - Si columns es None o lista vacía, usa todas las variables numéricas (int/float) excepto target.
    - Filtra por abs(corr) >= umbral_corr.
    - Si pvalue no es None, filtra además por p_val <= pvalue.

    EXTRA:
    - Si hay muchas columnas, divide en bloques de máximo 4 variables + target.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Columna objetivo numérica.
    columns (list[str] o None): Lista de columnas candidatas.
    umbral_corr (float): Umbral de correlación (0-1).
    pvalue (float o None): Nivel de significación.

    Retorna:
    list: Lista de columnas finalmente representadas.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df no es un DataFrame de pandas.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns:
        print("Error: target_col no válido o no existe.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser numérica.")
        return None

    if not isinstance(umbral_corr, (int, float)) or not (0 <= float(umbral_corr) <= 1):
        print("Error: umbral_corr debe ser un número entre 0 y 1.")
        return None

    umbral_corr = float(umbral_corr)

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) < 1):
            print("Error: pvalue debe ser None o un número entre 0 y 1.")
            return None
        pvalue = float(pvalue)

    if columns is None or len(columns) == 0:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    valid_cols = []

    for col in columns:
        if col not in df.columns:
            continue

        # Solo int/float
        if col not in df.select_dtypes(include=["int64", "float64"]).columns:
            continue

        data = df[[col, target_col]].dropna()
        if len(data) < 2:
            continue

        corr, p_val = pearsonr(data[col], data[target_col])

        if abs(corr) >= umbral_corr:
            if pvalue is None or p_val <= pvalue:
                valid_cols.append(col)

    for i in range(0, len(valid_cols), 4):
        subset = valid_cols[i:i + 4]
        sns.pairplot(df[[target_col] + subset].dropna())
        plt.show()

    return valid_cols


def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve una lista de variables categóricas cuya relación con target_col
    sea estadísticamente significativa.

    Se consideran categóricas:
    - object
    - category
    - bool (binarias)

    Test aplicado:
    - ANOVA si nº categorías <= 10
    - Kruskal-Wallis si nº categorías > 10

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Variable objetivo numérica.
    pvalue (float): Nivel de significación (por defecto 0.05).

    Retorna:
    list: Lista de variables categóricas significativas.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df no es un DataFrame de pandas.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns:
        print("Error: target_col no existe en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser una variable numérica.")
        return None

    if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) < 1):
        print("Error: pvalue debe ser un número entre 0 y 1.")
        return None

    pvalue = float(pvalue)

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    selected = []

    for col in cat_cols:
        unique_vals = df[col].dropna().unique()
        groups = [df[df[col] == val][target_col].dropna() for val in unique_vals]

        if len(groups) < 2:
            continue

        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue

        if len(groups) <= 10:
            stat, p = f_oneway(*groups)
        else:
            stat, p = kruskal(*groups)

        if p <= pvalue:
            selected.append(col)

    return selected


def plot_features_cat_regression(df, target_col="", columns=None, pvalue=0.05, with_individual_plot=False):
    """
    Visualiza la relación entre variables categóricas y un target numérico.

    - Si columns es None o lista vacía, usa todas las categóricas (object/category/bool).
    - Filtra por significación estadística (pvalue).
    - Si with_individual_plot=True, muestra un gráfico por variable categórica.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Variable objetivo numérica.
    columns (list[str] o None): Variables categóricas candidatas.
    pvalue (float): Nivel de significación.
    with_individual_plot (bool): Si True, muestra gráficos individuales.

    Retorna:
    list: Lista de variables categóricas significativas.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df no es un DataFrame de pandas.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns:
        print("Error: target_col no válido o no existe.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser numérica.")
        return None

    if not isinstance(pvalue, (int, float)) or not (0 < float(pvalue) < 1):
        print("Error: pvalue debe ser un número entre 0 y 1.")
        return None

    pvalue = float(pvalue)

    if columns is None or len(columns) == 0:
        columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    valid_cols = []

    for col in columns:
        if col not in df.columns:
            continue

        unique_vals = df[col].dropna().unique()
        groups = [df[df[col] == val][target_col].dropna() for val in unique_vals]

        if len(groups) < 2:
            continue

        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue

        if len(groups) <= 10:
            stat, p = f_oneway(*groups)
        else:
            stat, p = kruskal(*groups)

        if p <= pvalue:
            valid_cols.append(col)

            if with_individual_plot:
                sns.histplot(data=df, x=target_col, hue=col, kde=True)
                plt.title(f"{target_col} por {col}")
                plt.show()

    return valid_cols
