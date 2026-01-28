# TOOLBOX ML

## Team Challenge 3 – Grupo 4

Este repositorio contiene un módulo en Python (toolbox_ML.py) con un conjunto de funciones diseñadas para el análisis exploratorio de datos (EDA) y la selección de variables (features) en problemas de regresión, donde la variable objetivo es numérica.

El proyecto combina:

- Selección de variables basada en correlación y tests estadísticos

- Visualización de relaciones entre variables

- Un ejemplo práctico completo usando el dataset Titanic

## 📁 Estructura del proyecto

toolbox_ML.py   # Módulo con las funciones del challenge
README.md       # Documentación del proyecto

## ⚙️ Dependencias

El módulo utiliza las siguientes librerías:

- pandas

- numpy

- matplotlib

- seaborn

- scipy

### Instalación:

pip install pandas numpy matplotlib seaborn scipy

## 🧠 Funciones incluidas
### 1️⃣ describe_df(df)

Genera un resumen del DataFrame que incluye:

- Tipo de dato

- Porcentaje de valores nulos

- Número de valores únicos

- Porcentaje de cardinalidad

Útil para una primera inspección rápida del dataset.

### 2️⃣ tipifica_variables(df, umbral_categoria, umbral_continua)

Sugiere automáticamente el tipo de cada variable según su cardinalidad:

- Binaria

- Categórica

- Numérica discreta

- Numérica continua

Facilita decidir qué análisis aplicar a cada variable.

### 3️⃣ get_features_num_regression(df, target_col, umbral_corr, pvalue=None)

Selecciona variables numéricas cuya correlación (Pearson) con el target:

- Supere un umbral mínimo (umbral_corr)

- Y opcionalmente sea estadísticamente significativa (pvalue)

Excluye variables booleanas e incluye validaciones completas de entrada

### 4️⃣ plot_features_num_regression(df, target_col, columns=None, umbral_corr=0, pvalue=None)

- Visualiza la relación entre el target y variables numéricas mediante pairplots.

- Si no se especifican columnas, se usan todas las numéricas

- Filtra por correlación y significación estadística


### 5️⃣ get_features_cat_regression(df, target_col, pvalue=0.05)

- Selecciona variables categóricas relacionadas significativamente con un target numérico.

- Test estadístico aplicado automáticamente:

- ANOVA → si el número de categorías ≤ 10

- Kruskal-Wallis → si el número de categorías > 10


### 6️⃣ plot_features_cat_regression(df, target_col, columns=None, pvalue=0.05, with_individual_plot=False)

- Visualiza la distribución del target numérico agrupado por variables categóricas.

- Puede generar un gráfico por variable (with_individual_plot=True)

- Filtra previamente por significación estadística

## 🧪 Ejemplo práctico: Titanic

Este apartado demuestra el uso del módulo toolbox_ML.py en un caso real.

### 📊 Dataset

Dataset: Titanic (Seaborn)

Variable objetivo (regresión): fare

El objetivo es identificar y visualizar variables relevantes para explicar el precio del billete.

### 1️⃣ Resumen y tipificación de variables

Se emplean describe_df y tipifica_variables para:

- Analizar tipos de datos, valores nulos y cardinalidad

- Clasificar variables según su naturaleza (categórica, numérica, etc.)

Esto permite preparar correctamente el análisis posterior.

### 2️⃣ Selección de variables numéricas

Se seleccionan variables numéricas relevantes y se visualizan usando:

plot_features_num_regression(
    df,
    target_col="fare",
    columns=num_features,
    umbral_corr=0.2,
    pvalue=0.05
)


Resultado:

['parch']

#### 📈 Interpretación del pairplot (fare vs parch)

fare presenta una distribución asimétrica a la derecha, con la mayoría de billetes baratos y algunos valores extremos elevados.

parch es una variable discreta, siendo 0 el valor más frecuente.

El scatterplot muestra una tendencia positiva moderada: valores más altos de parch tienden a asociarse con precios de billete más elevados, aunque con bastante dispersión.


### 3️⃣ Selección de variables categóricas

Se seleccionan variables categóricas relacionadas significativamente con fare:

cat_features = get_features_cat_regression(
    df,
    target_col="fare",
    pvalue=0.05
)

cat_features


Resultado:

['sex', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone']

### 4️⃣ Visualización de variables categóricas

Se representan las distribuciones de fare para cada variable categórica significativa:

plot_features_cat_regression(
    df,
    target_col="fare",
    columns=cat_features,
    pvalue=0.05,
    with_individual_plot=True
)


Estas visualizaciones permiten comparar cómo cambia el precio del billete según el perfil del pasajero.

### ✅ Validaciones y robustez

Todas las funciones:

- Verifican tipos y valores de los argumentos

- Comprueban que target_col sea numérica

- Evitan errores por datos insuficientes

- Devuelven None e informan por pantalla cuando los parámetros no son válidos

### Conclusión

El módulo toolbox_ML.py proporciona una solución modular, robusta y reutilizable para:

- Analizar datasets de regresión

- Seleccionar variables relevantes

- Visualizar relaciones estadísticas clave

El ejemplo con Titanic valida su correcto funcionamiento y utilidad práctica.

## ✨ Autores

Daniel Mascarilla
Jorge Martínez Delgado
Kelly Escalante
Rebeca Prior
