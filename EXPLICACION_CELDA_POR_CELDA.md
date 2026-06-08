# EXPLICACIÓN CELDA POR CELDA — Los 3 Notebooks

## NOTEBOOK 1: CLASIFICACIÓN (Fraude)

### CELDA 1 — Markdown: Portada
```markdown
# Detección de Fraude en Transacciones de Tarjeta de Crédito
## Caso de Uso de Clasificación — CRISP-DM
**Universidad Popular del Cesar**
Docente: Aimer Rivera Centeno
```
**Qué es:** Título del notebook. Identifica el caso, tipo de problema y la universidad.

---

### CELDA 2 — Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import joblib
import os
```

**Explicación línea por línea:**

| Línea | Qué hace | Por qué |
|---|---|---|
| `import pandas as pd` | Librería para manipular datos tabulares (DataFrames) | Para cargar, filtrar, agrupar el CSV |
| `import numpy as np` | Operaciones numéricas y arreglos | Para máscaras booleanas, operaciones matemáticas |
| `import matplotlib.pyplot as plt` | Visualizaciones estáticas | Para histogramas, heatmaps |
| `import seaborn as sns` | Visualizaciones estadísticas sobre matplotlib | Para el mapa de correlación |
| `import plotly.express as px` | Visualizaciones interactivas (API simple) | Para pie charts, scatter plots interactivos |
| `import plotly.graph_objects as go` | Visualizaciones interactivas (API avanzada) | Para curvas ROC custom, matrices de confusión |
| `import warnings` | Módulo para controlar advertencias | — |
| `warnings.filterwarnings('ignore')` | Silencia todos los warnings | Para que el notebook se vea limpio |
| `from sklearn.model_selection import train_test_split` | Divide datos en train y test | Para evaluar el modelo con datos no vistos |
| `cross_val_score` | Validación cruzada | Para evaluar estabilidad del modelo (importado pero no usado directamente) |
| `StratifiedKFold` | Validación cruzada estratificada | Preserva proporción de clases en cada fold (importado pero no usado) |
| `from sklearn.preprocessing import StandardScaler` | Normaliza features a media=0, std=1 | Para escalar Time y Amount |
| `from sklearn.linear_model import LogisticRegression` | Modelo de regresión logística | Baseline lineal para clasificación |
| `from sklearn.ensemble import RandomForestClassifier` | Ensemble de árboles de decisión | Modelo principal, robusto |
| `GradientBoostingClassifier` | Boosting secuencial de árboles | Modelo alternativo de boosting |
| `from sklearn.metrics import classification_report` | Reporte completo de métricas | Para ver precision, recall, f1 por clase |
| `confusion_matrix` | Matriz de confusión | Para ver TP, FP, TN, FN |
| `roc_auc_score` | Calcula el área bajo la curva ROC | Métrica principal de poder discriminativo |
| `roc_curve` | Genera puntos para la curva ROC | Para graficar la curva |
| `precision_recall_curve` | Genera puntos para curva PR | Para análisis adicional (importado pero no graficado) |
| `f1_score` | Media armónica de precision y recall | Métrica de balance |
| `precision_score` | TP / (TP + FP) | Qué tan confiable es cuando dice "fraude" |
| `recall_score` | TP / (TP + FN) | Qué porcentaje de fraudes reales detecta |
| `from imblearn.over_sampling import SMOTE` | Técnica de sobremuestreo sintético | Para balancear el dataset |
| `from imblearn.pipeline import Pipeline as ImbPipeline` | Pipeline que incluye SMOTE | Para evitar data leakage (importado pero no usado) |
| `import xgboost as xgb` | Gradient boosting optimizado | Modelo state-of-the-art |
| `import joblib` | Serialización de modelos | Para guardar/cargar modelos .pkl |
| `import os` | Operaciones del sistema operativo | Para manejar rutas de archivos |

---

### CELDA 3 — Markdown: Fase 1 CRISP-DM
```markdown
## Fase 1: Comprensión del Negocio
Objetivo del negocio: Minimizar pérdidas financieras...
Métrica principal: Recall de la clase fraude...
```
**Qué es:** Documenta el objetivo de negocio. Explica por qué Recall es la métrica principal (un falso negativo = dinero perdido).

---

### CELDA 4 — Carga de datos y EDA básico
```python
df = pd.read_csv('../data/creditcard.csv')
print("Shape:", df.shape)
print("\nPrimeras filas:")
display(df.head())
print("\nInformación del dataset:")
df.info()
print("\nEstadísticas descriptivas:")
display(df.describe())
print("\nValores nulos:")
print(df.isnull().sum())
```

| Línea | Qué hace | Resultado |
|---|---|---|
| `df = pd.read_csv('../data/creditcard.csv')` | Carga el CSV desde la carpeta data | DataFrame con 284,807 filas × 31 columnas |
| `print("Shape:", df.shape)` | Imprime dimensiones | (284807, 31) |
| `display(df.head())` | Muestra las primeras 5 filas | Para inspección visual |
| `df.info()` | Muestra tipos de datos y nulos | 30 float64 + 1 int64, 0 nulos |
| `display(df.describe())` | Estadísticas descriptivas | Media, std, min, cuartiles, max |
| `print(df.isnull().sum())` | Cuenta nulos por columna | Todos 0 → no hay imputación necesaria |

---

### CELDA 5 — Distribución de clases
```python
clase_counts = df['Class'].value_counts()
fig = px.pie(values=clase_counts.values,
             names=['Legítima (0)', 'Fraude (1)'],
             title='Distribución de Clases — Dataset Extremadamente Desbalanceado',
             color_discrete_sequence=['#2196F3', '#F44336'])
fig.show()
print(f"\nTransacciones legítimas: {clase_counts[0]:,} ({clase_counts[0]/len(df)*100:.2f}%)")
print(f"Transacciones fraudulentas: {clase_counts[1]:,} ({clase_counts[1]/len(df)*100:.4f}%)")
```

| Línea | Qué hace |
|---|---|
| `df['Class'].value_counts()` | Cuenta cuántos 0 y cuántos 1 hay |
| `px.pie(...)` | Crea un gráfico de pastel interactivo |
| `values=clase_counts.values` | Los tamaños de cada porción |
| `names=['Legítima (0)', 'Fraude (1)']` | Etiquetas de las porciones |
| `color_discrete_sequence=['#2196F3', '#F44336']` | Azul para legítima, rojo para fraude |
| `fig.show()` | Renderiza el gráfico |
| `print(...)` | Imprime números exactos: 284,315 legítimas (99.83%), 492 fraudes (0.17%) |

---

### CELDA 6 — Distribución de Amount y Time por clase
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df[df['Class']==0]['Amount'].plot(kind='hist', bins=50, ax=axes[0],
    alpha=0.7, color='blue', label='Legítima')
df[df['Class']==1]['Amount'].plot(kind='hist', bins=50, ax=axes[0],
    alpha=0.7, color='red', label='Fraude')
axes[0].set_title('Distribución del Monto por Clase')
axes[0].set_xlabel('Amount (€)')
axes[0].legend()
axes[0].set_yscale('log')

df[df['Class']==0]['Time'].plot(kind='hist', bins=50, ax=axes[1],
    alpha=0.7, color='blue', label='Legítima')
df[df['Class']==1]['Time'].plot(kind='hist', bins=50, ax=axes[1],
    alpha=0.7, color='red', label='Fraude')
axes[1].set_title('Distribución del Tiempo por Clase')
axes[1].set_xlabel('Time (segundos)')
axes[1].legend()
plt.tight_layout()
plt.show()
```

| Línea | Qué hace |
|---|---|
| `plt.subplots(1, 2, figsize=(14, 5))` | Crea 1 fila × 2 columnas de gráficos |
| `df[df['Class']==0]['Amount']` | Filtra solo transacciones legítimas, columna Amount |
| `.plot(kind='hist', bins=50, ax=axes[0])` | Histograma de 50 barras en el primer eje |
| `alpha=0.7` | Transparencia para superponer histogramas |
| `axes[0].set_yscale('log')` | Escala logarítmica porque hay mucha diferencia de frecuencias |
| `plt.tight_layout()` | Ajusta espaciado entre gráficos |

---

### CELDA 7 — Mapa de correlación
```python
plt.figure(figsize=(16, 12))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
            annot=False, linewidths=0.5, fmt='.2f')
plt.title('Mapa de Correlación — Variables V1 a V28, Time, Amount, Class')
plt.tight_layout()
plt.show()

print("\nTop 10 correlaciones con 'Class':")
print(corr['Class'].abs().sort_values(ascending=False).head(11))
```

| Línea | Qué hace |
|---|---|
| `df.corr()` | Calcula correlación de Pearson entre TODAS las columnas |
| `np.triu(np.ones_like(corr, dtype=bool))` | Crea máscara triangular superior para no duplicar |
| `sns.heatmap(...)` | Heatmap de correlación |
| `mask=mask` | Oculta el triángulo superior |
| `cmap='coolwarm'` | Rojo = positiva, Azul = negativa |
| `center=0` | El blanco es correlación 0 |
| `corr['Class'].abs().sort_values(ascending=False)` | Ordena por correlación absoluta con Class |
| `.head(11)` | Top 11 (incluye Class consigo misma = 1.0) |

**Resultado:** V17 (0.326), V14 (0.303), V12 (0.261) son las más correlacionadas con fraude.

---

### CELDA 8 — Preparación de datos (Scaling + SMOTE + Split)
```python
scaler = StandardScaler()
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

df_clean = df.drop(['Time', 'Amount'], axis=1)

X = df_clean.drop('Class', axis=1)
y = df_clean['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Fraudes en train: {y_train.sum()} | Fraudes en test: {y_test.sum()}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"\nDespués de SMOTE — Train shape: {X_train_res.shape}")
print(f"Distribución post-SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")
```

| Línea | Qué hace | Por qué |
|---|---|---|
| `StandardScaler()` | Crea el scaler | Normaliza a media=0, std=1 |
| `scaler.fit_transform(df[['Time']])` | Ajusta y transforma Time | Time va de 0 a 172,792 — necesita escalado |
| `scaler.fit_transform(df[['Amount']])` | Ajusta y transforma Amount | Amount va de 0 a $25,691 — necesita escalado |
| `df.drop(['Time', 'Amount'], axis=1)` | Elimina columnas originales | Ya tenemos las versiones escaladas |
| `X = df_clean.drop('Class', axis=1)` | Separa features (30 columnas) | Input del modelo |
| `y = df_clean['Class']` | Separa variable objetivo | Target del modelo |
| `train_test_split(...)` | Divide 80% train, 20% test | Para evaluar con datos no vistos |
| `test_size=0.2` | 20% para test | Estándar de la industria |
| `random_state=42` | Semilla fija | Reproducibilidad |
| `stratify=y` | Mantiene proporción de clases | Esencial con datos desbalanceados |
| `SMOTE(random_state=42)` | Crea el oversampler sintético | Para balancear el training set |
| `smote.fit_resample(X_train, y_train)` | Aplica SMOTE SOLO al train | Evita data leakage |

**Resultado:** Train: 227,845 → 454,902 filas (50/50 después de SMOTE). Test: 56,962 filas (sin balancear — mundo real).

---

### CELDA 9 — Entrenamiento de modelos
```python
modelos = {
    'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

resultados = {}
modelos_entrenados = {}

for nombre, modelo in modelos.items():
    print(f"\nEntrenando: {nombre}...")
    modelo.fit(X_train_res, y_train_res)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    resultados[nombre] = {
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }
    modelos_entrenados[nombre] = {'modelo': modelo, 'y_pred': y_pred, 'y_prob': y_prob}
    print(f"  ROC-AUC: {resultados[nombre]['ROC-AUC']:.4f} | F1: {resultados[nombre]['F1-Score']:.4f}")
```

| Línea | Qué hace | Por qué |
|---|---|---|
| `LogisticRegression(max_iter=1000)` | Regresión logística con 1000 iteraciones | max_iter=1000 para que converja |
| `RandomForestClassifier(n_estimators=100)` | 100 árboles de decisión | Más árboles = más estable |
| `n_jobs=-1` | Usa todos los cores del CPU | Más rápido |
| `xgb.XGBClassifier(eval_metric='logloss')` | XGBoost con métrica logloss | logloss es apropiado para binaria |
| `GradientBoostingClassifier(n_estimators=100)` | 100 árboles de boosting secuencial | Cada árbol corrige errores del anterior |
| `modelo.fit(X_train_res, y_train_res)` | Entrena con datos balanceados por SMOTE | Los datos res tienen 50/50 |
| `modelo.predict(X_test)` | Predice clases (0 o 1) en el test | Para precision, recall, f1 |
| `modelo.predict_proba(X_test)[:, 1]` | Predice probabilidad de fraude | Para ROC-AUC (necesita probabilidades, no clases) |
| `roc_auc_score(y_test, y_prob)` | Calcula área bajo curva ROC | Poder discriminativo |
| `f1_score(y_test, y_pred)` | Media armónica precision/recall | Balance entre ambas |
| `precision_score(y_test, y_pred)` | TP/(TP+FP) | Cuándo dice fraude, ¿cuántas veces acierta? |
| `recall_score(y_test, y_pred)` | TP/(TP+FN) | ¿Qué % de fraudes reales detectó? |

---

### CELDA 10 — Evaluación
```python
print("=== COMPARACIÓN DE MODELOS ===")
df_resultados = pd.DataFrame(resultados).T.round(4)
display(df_resultados.sort_values('ROC-AUC', ascending=False))

fig = go.Figure()
for nombre, data in modelos_entrenados.items():
    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
    auc = resultados[nombre]['ROC-AUC']
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{nombre} (AUC={auc:.3f})"))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash', color='gray')))
fig.update_layout(title='Curvas ROC — Comparación de Modelos',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate')
fig.show()

mejor_nombre = df_resultados['ROC-AUC'].idxmax()
mejor_pred = modelos_entrenados[mejor_nombre]['y_pred']
cm = confusion_matrix(y_test, mejor_pred)
fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                   labels=dict(x="Predicho", y="Real"),
                   x=['Legítima','Fraude'], y=['Legítima','Fraude'],
                   title=f'Matriz de Confusión — {mejor_nombre}')
fig_cm.show()

print(f"\nMejor modelo: {mejor_nombre}")
print(classification_report(y_test, mejor_pred, target_names=['Legítima', 'Fraude']))
```

| Línea | Qué hace |
|---|---|
| `pd.DataFrame(resultados).T` | Convierte diccionario a DataFrame y transpone |
| `.sort_values('ROC-AUC', ascending=False)` | Ordena de mayor a menor ROC-AUC |
| `roc_curve(y_test, data['y_prob'])` | Genera puntos (FPR, TPR) para la curva ROC |
| `go.Scatter(x=fpr, y=tpr, ...)` | Agrega cada curva al gráfico |
| `x=[0,1], y=[0,1]` | Línea diagonal = modelo aleatorio (AUC=0.5) |
| `df_resultados['ROC-AUC'].idxmax()` | Nombre del modelo con mayor ROC-AUC |
| `confusion_matrix(y_test, mejor_pred)` | Matriz 2×2: TP, FP, TN, FN |
| `px.imshow(cm, text_auto=True)` | Heatmap de la matriz de confusión |
| `classification_report(...)` | Tabla completa: precision, recall, f1 por clase |

---

### CELDA 11 — Guardar modelo
```python
import os
os.makedirs('../../../app/clasificacion_app/model', exist_ok=True)
joblib.dump(modelos_entrenados[mejor_nombre]['modelo'],
            '../../../app/clasificacion_app/model/modelo_fraude.pkl')
print(f"Modelo '{mejor_nombre}' guardado exitosamente.")
```

| Línea | Qué hace |
|---|---|
| `os.makedirs(..., exist_ok=True)` | Crea la carpeta si no existe |
| `joblib.dump(modelo, 'ruta.pkl')` | Serializa el modelo en archivo .pkl |
| `modelos_entrenados[mejor_nombre]['modelo']` | El modelo Random Forest ganador |

---

## NOTEBOOK 2: REGRESIÓN (Seguro Médico)

### CELDA 1 — Markdown: Portada
```markdown
# Predicción de Costos de Seguro Médico
## Caso de Uso de Regresión — CRISP-DM
```

---

### CELDA 2 — Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os

print("✓ Librerías cargadas exitosamente")
```

**Diferencias con el notebook de clasificación:**

| Import | Qué es | Por qué aquí y no en clasificación |
|---|---|---|
| `LinearRegression` | Regresión lineal simple | Baseline para regresión (no existe en clasificación) |
| `Ridge` | Regresión lineal con regularización L2 | Previene overfitting penalizando coeficientes grandes |
| `Lasso` | Regresión lineal con regularización L1 | Puede eliminar features (coeficiente = 0) |
| `RandomForestRegressor` | Random Forest para regresión | Versión regresión (predice valor continuo) |
| `GradientBoostingRegressor` | Gradient Boosting para regresión | Versión regresión |
| `SVR` | Support Vector Regression | **Importado pero NO usado**. Se descartó por ser lento y difícil de interpretar |
| `LabelEncoder` | Codifica categóricas como números | Para convertir sex, smoker a 0/1 |
| `mean_absolute_error` | MAE: error promedio en dólares | Métrica principal de regresión |
| `mean_squared_error` | MSE: error cuadrático promedio | Para calcular RMSE |
| `r2_score` | R²: bondad de ajuste | Qué % de varianza explica el modelo |

**¿Qué es SVR y por qué NO lo usamos?**
SVR (Support Vector Regression) es una variante de SVM para regresión. Busca un hiperplano que minimice el error dentro de un margen epsilon. **Lo importamos pero NO lo usamos** porque:
1. Es muy lento con tuning de hiperparámetros
2. Es sensible al escalado de datos
3. Es difícil de interpretar (no da importancia de features)
4. Los modelos basados en árboles (Gradient Boosting, XGBoost) son superiores para este tipo de datos tabulares

---

### CELDA 3 — Carga y EDA
```python
df = pd.read_csv('../data/insurance.csv')
print("=" * 60)
print("RESUMEN DEL DATASET")
print("=" * 60)
print(f"Shape (filas, columnas): {df.shape}")
print("\nPrimeras 10 filas:")
display(df.head(10))
print("\nInformación del dataset:")
df.info()
print(f"\nValores nulos totales: {df.isnull().sum().sum()}")
print("\nEstadísticas descriptivas:")
display(df.describe())
```

**Resultado:** 1,338 filas × 7 columnas. 0 nulos. Columnas: age(int), sex(str), bmi(float), children(int), smoker(str), region(str), charges(float).

---

### CELDA 4 — Distribución de charges
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['charges'], bins=50, color='#2196F3', edgecolor='white')
axes[0].set_title('Distribución de Charges (original)')
axes[0].set_xlabel('Charges (USD)')
axes[0].set_ylabel('Frecuencia')

axes[1].hist(np.log1p(df['charges']), bins=50, color='#4CAF50', edgecolor='white')
axes[1].set_title('Distribución de log(Charges) — normalizada')
axes[1].set_xlabel('log(Charges)')
axes[1].set_ylabel('Frecuencia')
plt.tight_layout()
plt.show()

print(f"Media: ${df['charges'].mean():,.2f}")
print(f"Mediana: ${df['charges'].median():,.2f}")
print(f"Skewness (asimetría): {df['charges'].skew():.3f}")
print(f"\nConclusión: La media > mediana y skewness positivo confirman "
      f"una distribución con cola derecha larga.")
```

| Línea | Qué hace |
|---|---|
| `np.log1p(df['charges'])` | Calcula ln(1 + x) para evitar log(0) | Transformación para normalizar la distribución |
| `df['charges'].skew()` | Calcula asimetría | > 1 = fuertemente sesgada a la derecha |

**Resultado:** Media $13,270 > Mediana $9,382, Skewness = 1.516 → cola larga a la derecha.

---

### CELDA 5 — Impacto del tabaquismo
```python
fig = px.box(df, x='smoker', y='charges', color='smoker',
             title='Distribución de Charges por Estatus de Fumador',
             color_discrete_map={'yes': '#F44336', 'no': '#2196F3'},
             labels={'smoker': 'Fumador', 'charges': 'Charges (USD)'})
fig.show()

print("Costo promedio fumadores: ${:,.2f}".format(
    df[df['smoker']=='yes']['charges'].mean()))
print("Costo promedio no fumadores: ${:,.2f}".format(
    df[df['smoker']=='no']['charges'].mean()))
print(f"\nDiferencia: los fumadores pagan ~3.8x más que los no fumadores.")
```

| Línea | Qué hace |
|---|---|
| `px.box(...)` | Box plot: muestra mediana, cuartiles, outliers |
| `color_discrete_map={'yes': '#F44336'}` | Rojo para fumadores, azul para no fumadores |
| `df[df['smoker']=='yes']['charges'].mean()` | Promedio de charges solo para fumadores |

**Resultado:** Fumadores $32,050 vs No fumadores $8,434 → 3.8× más.

---

### CELDA 6 — Scatter matrix y correlaciones
```python
fig = px.scatter_matrix(df,
    dimensions=['age', 'bmi', 'children', 'charges'],
    color='smoker',
    title='Scatter Matrix — Variables Numéricas por Estatus Fumador',
    color_discrete_map={'yes': '#F44336', 'no': '#2196F3'})
fig.show()

corr = df.select_dtypes(include=np.number).corr()
fig_h = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                  title='Correlación entre Variables Numéricas')
fig_h.show()
```

| Línea | Qué hace |
|---|---|
| `px.scatter_matrix(...)` | Matriz de scatter plots: cada par de variables |
| `dimensions=['age', 'bmi', 'children', 'charges']` | Solo variables numéricas |
| `color='smoker'` | Colorea por fumador para ver patrones |
| `df.select_dtypes(include=np.number)` | Solo columnas numéricas (excluye sex, smoker, region) |
| `.corr()` | Matriz de correlación de Pearson |

---

### CELDA 7 — Charges por región y sexo
```python
fig = px.box(df, x='region', y='charges', color='sex',
             title='Charges por Región y Sexo',
             labels={'region': 'Región', 'charges': 'Charges (USD)'})
fig.show()
```
**Qué hace:** Box plot de charges por región, coloreado por sexo. Permite ver si hay diferencias significativas entre regiones o sexos.

---

### CELDA 8 — Preparación de datos
```python
df_model = df.copy()

df_model['smoker_enc'] = (df_model['smoker'] == 'yes').astype(int)
df_model['sex_enc'] = (df_model['sex'] == 'male').astype(int)
df_model = pd.get_dummies(df_model, columns=['region'], prefix='region', drop_first=True)
df_model.drop(['sex', 'smoker'], axis=1, inplace=True)

print("Columnas después del encoding:")
print(df_model.columns.tolist())

df_model['bmi_smoker'] = df_model['bmi'] * df_model['smoker_enc']
df_model['age_sq'] = df_model['age'] ** 2

X = df_model.drop('charges', axis=1)
y = df_model['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
```

| Línea | Qué hace | Por qué |
|---|---|---|
| `df.copy()` | Copia el DataFrame | Para no modificar el original |
| `(df_model['smoker'] == 'yes').astype(int)` | Convierte yes→1, no→0 | Encoding binario (solo 2 valores) |
| `(df_model['sex'] == 'male').astype(int)` | Convierte male→1, female→0 | Encoding binario |
| `pd.get_dummies(..., drop_first=True)` | One-hot encoding para region | drop_first=True evita multicolinealidad (4 regiones → 3 dummies) |
| `drop(['sex', 'smoker'], axis=1)` | Elimina columnas originales | Ya tenemos las versiones encoded |
| `df_model['bmi_smoker'] = bmi * smoker_enc` | **Feature engineering: interacción** | Captura efecto sinérgico BMI-tabaquismo |
| `df_model['age_sq'] = age ** 2` | **Feature engineering: término cuadrático** | Captura no-linealidad de edad |
| `train_test_split(...)` | Divide 80/20 | Sin stratify (es regresión, no clasificación) |
| `StandardScaler()` | Normaliza todas las features | Necesario para Ridge, Lasso, SVR |
| `scaler.fit_transform(X_train)` | Ajusta y transforma train | fit en train |
| `scaler.transform(X_test)` | Solo transforma test | NO fit en test (evita data leakage) |

**Columnas finales:** age, bmi, children, smoker_enc, sex_enc, region_northwest, region_southeast, region_southwest, bmi_smoker, age_sq = **10 features**

---

### CELDA 9 — Modelado
```python
modelos = {
    'Regresión Lineal': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
}

resultados = {}
modelos_entrenados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_sc, y_train)
    y_pred = modelo.predict(X_test_sc)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    resultados[nombre] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    modelos_entrenados[nombre] = {'modelo': modelo, 'y_pred': y_pred}
    print(f"{nombre:25s} | MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | R²: {r2:.4f}")
```

| Línea | Qué hace |
|---|---|
| `LinearRegression()` | Regresión lineal simple (sin regularización) |
| `Ridge(alpha=1.0)` | Regresión lineal con penalización L2 (alpha=1.0) |
| `Lasso(alpha=1.0)` | Regresión lineal con penalización L1 (alpha=1.0) |
| `RandomForestRegressor(n_estimators=200)` | 200 árboles para regresión |
| `xgb.XGBRegressor(n_estimators=200)` | XGBoost para regresión |
| `GradientBoostingRegressor(n_estimators=200)` | Gradient Boosting para regresión |
| `modelo.fit(X_train_sc, y_train)` | Entrena con datos escalados |
| `mean_absolute_error(y_test, y_pred)` | MAE: error promedio en dólares |
| `np.sqrt(mean_squared_error(...))` | RMSE: raíz del error cuadrático |
| `r2_score(y_test, y_pred)` | R²: % de varianza explicada |

---

### CELDA 10 — Evaluación
```python
df_res = pd.DataFrame(resultados).T.round(4)
display(df_res.sort_values('R²', ascending=False))

mejor_nombre = df_res['R²'].idxmax()
mejor_pred = modelos_entrenados[mejor_nombre]['y_pred']

fig = px.scatter(x=y_test, y=mejor_pred,
                 labels={'x': 'Charges Real (USD)', 'y': 'Charges Predicho (USD)'},
                 title=f'Real vs Predicho — {mejor_nombre}',
                 opacity=0.6)
fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
              x1=y_test.max(), y1=y_test.max(),
              line=dict(color='red', dash='dash'))
fig.show()

residuales = y_test - mejor_pred
fig_res = px.histogram(residuales, nbins=50,
                        title=f'Distribución de Residuales — {mejor_nombre}',
                        labels={'value': 'Error (USD)'})
fig_res.show()

if hasattr(modelos_entrenados[mejor_nombre]['modelo'], 'feature_importances_'):
    importances = modelos_entrenados[mejor_nombre]['modelo'].feature_importances_
    fi_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(10)
    fig_fi = px.bar(fi_df, x='importance', y='feature', orientation='h',
                    title=f'Importancia de Features — {mejor_nombre}',
                    color='importance', color_continuous_scale='Viridis')
    fig_fi.show()
```

| Línea | Qué hace |
|---|---|
| `px.scatter(x=y_test, y=mejor_pred)` | Scatter: real vs predicho |
| `fig.add_shape(type='line', ...)` | Línea diagonal roja = predicción perfecta |
| `residuales = y_test - mejor_pred` | Error de cada predicción |
| `px.histogram(residuales)` | Distribución de errores (debería ser normal centrada en 0) |
| `hasattr(..., 'feature_importances_')` | Verifica si el modelo da importancia de features |
| `modelo.feature_importances_` | Importancia de cada variable (solo árboles) |
| `fi_df.sort_values('importance', ascending=False).head(10)` | Top 10 features más importantes |

---

### CELDA 11 — Guardar modelo
```python
import os
os.makedirs('../../../app/regresion_app/model', exist_ok=True)
joblib.dump(modelos_entrenados[mejor_nombre]['modelo'],
            '../../../app/regresion_app/model/modelo_seguro.pkl')
joblib.dump(scaler, '../../../app/regresion_app/model/scaler_seguro.pkl')
joblib.dump(list(X.columns), '../../../app/regresion_app/model/feature_names.pkl')
print(f"Modelo '{mejor_nombre}' guardado.")
```

| Línea | Qué hace |
|---|---|
| `joblib.dump(modelo, 'modelo_seguro.pkl')` | Guarda el modelo Gradient Boosting |
| `joblib.dump(scaler, 'scaler_seguro.pkl')` | Guarda el scaler (necesario para inferencia) |
| `joblib.dump(list(X.columns), 'feature_names.pkl')` | Guarda nombres de features (para orden correcto en inferencia) |

---

## NOTEBOOK 3: AGRUPAMIENTO (Spotify)

### CELDA 1 — Markdown: Portada
```markdown
# Agrupamiento de Canciones Spotify por Características de Audio
## Caso de Uso de Clustering — CRISP-DM
```

---

### CELDA 2 — Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

import joblib
import os

print('✓ Todas las librerías cargadas exitosamente')
```

| Import | Qué es | Por qué |
|---|---|---|
| `KMeans` | Algoritmo principal de clustering | Agrupa por distancia euclidiana |
| `DBSCAN` | Clustering basado en densidad | **Importado pero NO usado**. Sensible a parámetros en 10D |
| `AgglomerativeClustering` | Clustering jerárquico | **Importado pero NO usado**. O(n²), muy lento con 114K filas |
| `PCA` | Análisis de Componentes Principales | Para reducir de 10D a 2D y visualizar |
| `silhouette_score` | Mide qué tan bien separados están los clusters | Cohesión interna + separación externa |
| `davies_bouldin_score` | Ratio de similitud entre clusters | Menor es mejor |

**¿Por qué NO usamos DBSCAN?**
DBSCAN agrupa por densidad: puntos cercanos forman clusters, puntos aislados son ruido. Problemas:
1. Muy sensible a los parámetros eps (distancia) y min_samples
2. No funciona bien en alta dimensionalidad (10 features) — la "maldición de la dimensionalidad" hace que todas las distancias sean similares
3. No garantiza que todos los puntos pertenezcan a un cluster (puede marcar muchos como ruido)

**¿Por qué NO usamos Agglomerative Clustering?**
Es clustering jerárquico: empieza con cada punto como cluster y va fusionando los más cercanos. Problema:
1. Complejidad O(n²) o O(n³) — con 114,000 canciones tardaría HORAS
2. Consume mucha memoria

---

### CELDA 3 — Carga y EDA
```python
df = pd.read_csv('../data/dataset.csv')
print('Shape del dataset:', df.shape)
print('\nPrimeras 5 filas:')
display(df.head())
print('\nInformación del dataset:')
df.info()
print('\nValores nulos por columna:')
print(df.isnull().sum())
print('\nGéneros únicos:', df['track_genre'].nunique())
```

**Resultado:** 114,000 filas × 21 columnas. 114 géneros únicos. artists, album_name, track_name tienen 1 null cada uno (irrelevantes para clustering).

---

### CELDA 4 — Distribuciones de features
```python
features_audio = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, feat in enumerate(features_audio):
    axes[i].hist(df[feat].dropna(), bins=50, color='#1DB954', edgecolor='white', alpha=0.8)
    axes[i].set_title(f'Distribución: {feat}', fontsize=12)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Frecuencia')

plt.suptitle('Distribuciones de Características de Audio — Spotify', fontsize=16)
plt.tight_layout()
plt.show()
```

| Línea | Qué hace |
|---|---|
| `features_audio = [...]` | Lista de 9 features de audio (sin popularity) |
| `plt.subplots(3, 3, ...)` | Grid 3×3 = 9 histogramas |
| `axes.flatten()` | Convierte matriz 3×3 en lista plana de 9 ejes |
| `for i, feat in enumerate(features_audio)` | Itera sobre cada feature |
| `df[feat].dropna()` | Elimina nulos antes de graficar |
| `color='#1DB954'` | Verde Spotify |

---

### CELDA 5 — Correlaciones
```python
corr_features = df[features_audio + ['popularity']].corr()

fig = px.imshow(corr_features,
                title='Matriz de Correlación — Features de Audio',
                color_continuous_scale='RdBu_r',
                text_auto='.2f',
                width=800, height=700)
fig.show()

print('\nObservaciones:')
print('- Energy y loudness tienen correlación positiva fuerte (r > 0.7)')
print('- Energy y acousticness tienen correlación negativa fuerte (r < -0.7)')
```

| Línea | Qué hace |
|---|---|
| `features_audio + ['popularity']` | 9 features + popularity = 10 columnas |
| `.corr()` | Matriz de correlación 10×10 |
| `px.imshow(...)` | Heatmap interactivo |
| `color_continuous_scale='RdBu_r'` | Rojo = positiva, Azul = negativa (invertido) |
| `text_auto='.2f'` | Muestra valores con 2 decimales |

---

### CELDA 6 — Scatter danceability vs energy por género
```python
muestra = df.sample(3000, random_state=42)
fig = px.scatter(muestra, x='danceability', y='energy',
                 color='track_genre',
                 hover_data=['track_name', 'artists'],
                 title='Danceability vs Energy por Género (muestra 3,000 canciones)',
                 opacity=0.6)
fig.show()
```

| Línea | Qué hace | Por qué |
|---|---|---|
| `df.sample(3000, random_state=42)` | Muestra aleatoria de 3,000 canciones | 114K es demasiado para graficar todo |
| `color='track_genre'` | Colorea por género | Para ver si géneros se agrupan naturalmente |
| `hover_data=['track_name', 'artists']` | Muestra nombre y artista al pasar el mouse | Interactividad |

---

### CELDA 7 — Preparación de datos
```python
features_cluster = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness',
                    'valence', 'tempo', 'popularity']

df_cluster = df[features_cluster].dropna()
print(f"Registros para clustering: {df_cluster.shape[0]:,}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)
print("Datos normalizados. Shape:", X_scaled.shape)
```

| Línea | Qué hace | Por qué |
|---|---|---|
| `features_cluster = [...]` | 10 features para clustering | Incluye popularity |
| `df[features_cluster].dropna()` | Solo las 10 features, sin nulos | Limpieza |
| `StandardScaler()` | Normaliza a media=0, std=1 | **OBLIGATORIO** en K-Means |
| `scaler.fit_transform(df_cluster)` | Escala todas las features | Sin escalar, tempo (50-200) dominaría la distancia |

---

### CELDA 8 — Método del codo
```python
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, kmeans.labels_, sample_size=5000, random_state=42)
    silhouettes.append(sil)
    print(f"K={k} | Inertia: {kmeans.inertia_:.0f} | Silhouette: {sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('Método del Codo')
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inercia')

axes[1].plot(K_range, silhouettes, 'rs-')
axes[1].set_title('Silhouette Score por K')
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
plt.tight_layout()
plt.show()
```

| Línea | Qué hace | Por qué |
|---|---|---|
| `K_range = range(2, 11)` | Prueba K de 2 a 10 | Rango razonable |
| `KMeans(n_clusters=k, n_init=10)` | K-Means con k clusters, 10 inicializaciones | n_init=10 para evitar óptimos locales |
| `kmeans.inertia_` | Suma de distancias al cuadrado al centroide | Menor = clusters más compactos |
| `silhouette_score(..., sample_size=5000)` | Silhouette con muestra de 5,000 | 114K es demasiado para calcular exacto |
| `axes[0].plot(K_range, inertias, 'bo-')` | Gráfico de inercia (azul, círculos, línea) | Buscar el "codo" |
| `axes[1].plot(K_range, silhouettes, 'rs-')` | Gráfico de silhouette (rojo, cuadrados, línea) | Buscar el máximo |

---

### CELDA 9 — K-Means con K=6
```python
K_OPTIMO = 6

kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_cluster = df_cluster.copy()
df_cluster['cluster'] = clusters
df.loc[df_cluster.index, 'cluster'] = clusters

sil_final = silhouette_score(X_scaled, clusters, sample_size=5000, random_state=42)
db_final = davies_bouldin_score(X_scaled, clusters)
print(f"K-Means K={K_OPTIMO}")
print(f"Silhouette Score: {sil_final:.4f}")
print(f"Davies-Bouldin Index: {db_final:.4f}")
print(f"\nDistribución de clusters:\n{pd.Series(clusters).value_counts().sort_index()}")
```

| Línea | Qué hace |
|---|---|
| `K_OPTIMO = 6` | Elegimos K=6 (balance entre granularidad e interpretabilidad) |
| `kmeans.fit_predict(X_scaled)` | Entrena y predice clusters en una sola llamada |
| `df_cluster['cluster'] = clusters` | Agrega columna de cluster al DataFrame |
| `silhouette_score(...)` | Silhouette final: 0.1652 (moderado) |
| `davies_bouldin_score(...)` | Davies-Bouldin: 1.6242 (menor es mejor) |
| `pd.Series(clusters).value_counts()` | Cuántas canciones en cada cluster |

---

### CELDA 10 — Visualización PCA
```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Varianza explicada por 2 componentes: {pca.explained_variance_ratio_.sum()*100:.1f}%")

df_plot = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': clusters.astype(str)
})

fig = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster',
                 title=f'Clusters K-Means (K={K_OPTIMO}) — Proyección PCA 2D',
                 opacity=0.5, color_discrete_sequence=px.colors.qualitative.Set1)
fig.show()
```

| Línea | Qué hace |
|---|---|
| `PCA(n_components=2)` | Reduce de 10 dimensiones a 2 |
| `pca.fit_transform(X_scaled)` | Ajusta y transforma los datos |
| `pca.explained_variance_ratio_.sum()` | % de varianza total explicada por 2 componentes |
| `X_pca[:, 0]` | Primera componente principal (eje X) |
| `X_pca[:, 1]` | Segunda componente principal (eje Y) |
| `px.scatter(..., color='Cluster')` | Scatter plot coloreado por cluster |

**Resultado:** 2 componentes explican ~43% de la varianza. La visualización es orientativa, no perfecta.

---

### CELDA 11 — Perfil de clusters
```python
perfil = df_cluster.groupby('cluster')[features_cluster].mean().round(3)
print("Perfil promedio por cluster:")
display(perfil)

fig = go.Figure()
features_radar = ['danceability', 'energy', 'speechiness',
                  'acousticness', 'liveness', 'valence']
for cluster_id in range(K_OPTIMO):
    valores = perfil.loc[cluster_id, features_radar].tolist()
    valores += [valores[0]]
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=features_radar + [features_radar[0]],
        fill='toself',
        name=f'Cluster {cluster_id}'
    ))
fig.update_layout(title='Perfil de Audio por Cluster (Radar Chart)',
                  polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
fig.show()

nombres_clusters = {
    0: 'Energía y Baile',
    1: 'Acústico y Relajado',
    2: 'Instrumental',
    3: 'Hablado',
    4: 'En Vivo',
    5: 'Melódico Positivo'
}
print("\nNombres sugeridos para clusters:", nombres_clusters)
```

| Línea | Qué hace |
|---|---|
| `df_cluster.groupby('cluster')[features].mean()` | Promedio de cada feature por cluster |
| `go.Scatterpolar(...)` | Gráfico radar (polígono) por cluster |
| `valores += [valores[0]]` | Cierra el polígono repitiendo el primer valor |
| `fill='toself'` | Rellena el polígono |
| `polar=dict(radialaxis=dict(range=[0, 1]))` | Eje radial de 0 a 1 (todas las features están en ese rango) |

---

### CELDA 12 — Guardar modelos
```python
import os
os.makedirs('../../../app/agrupamiento_app/model', exist_ok=True)
joblib.dump(kmeans, '../../../app/agrupamiento_app/model/kmeans_spotify.pkl')
joblib.dump(scaler, '../../../app/agrupamiento_app/model/scaler_spotify.pkl')
joblib.dump(pca, '../../../app/agrupamiento_app/model/pca_spotify.pkl')
print("Modelos guardados.")
```

| Línea | Qué hace |
|---|---|
| `joblib.dump(kmeans, ...)` | Guarda el modelo K-Means |
| `joblib.dump(scaler, ...)` | Guarda el scaler (necesario para escalar nuevas canciones) |
| `joblib.dump(pca, ...)` | Guarda el PCA (para visualización en la app) |

---

## RESUMEN DE IMPORTS DESCARTADOS

| Notebook | Import | ¿Se usó? | ¿Por qué se descartó? |
|---|---|---|---|
| **Clasificación** | `cross_val_score` | ❌ No | No se hizo validación cruzada explícita |
| **Clasificación** | `StratifiedKFold` | ❌ No | No se hizo validación cruzada con folds |
| **Clasificación** | `precision_recall_curve` | ❌ No | No se graficó la curva Precision-Recall |
| **Clasificación** | `ImbPipeline` | ❌ No | Se aplicó SMOTE manualmente, no en pipeline |
| **Regresión** | `cross_val_score` | ❌ No | No se hizo validación cruzada explícita |
| **Regresión** | `LabelEncoder` | ❌ No | Se usó encoding manual con booleanos |
| **Regresión** | `SVR` | ❌ No | Lento, sensible a escalado, difícil de interpretar |
| **Agrupamiento** | `DBSCAN` | ❌ No | Sensible a parámetros en 10D, no garantiza clusters |
| **Agrupamiento** | `AgglomerativeClustering` | ❌ No | O(n²), muy lento con 114K filas |
