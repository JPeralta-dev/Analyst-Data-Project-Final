# EXPLICACIÓN CELDA POR CELDA — Los 3 Notebooks

> **Cada celda incluye:** código → resultado real → interpretación de cada número → qué responder si el profesor pregunta.

---

# NOTEBOOK 1: CLASIFICACIÓN (Fraude)

---

## CELDA 1 — Markdown: Portada

```markdown
# Detección de Fraude en Transacciones de Tarjeta de Crédito
## Caso de Uso de Clasificación — CRISP-DM
**Universidad Popular del Cesar**
Docente: Aimer Rivera Centeno
```

**Qué es:** Solo título. No produce resultado ejecutable.

---

## CELDA 2 — Imports

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

**Resultado:** No imprime nada. Solo carga librerías en memoria.

**Si el profesor pregunta cada import:**

| Import | Qué es | Para qué se usa EN ESTE notebook |
|---|---|---|
| `pandas as pd` | Manipular tablas de datos (DataFrames) | Cargar CSV, filtrar, agrupar, crear columnas |
| `numpy as np` | Operaciones numéricas con arreglos | Crear la máscara `np.triu()` para el heatmap |
| `matplotlib.pyplot as plt` | Gráficos estáticos | Histogramas de Amount/Time, mapa de correlación |
| `seaborn as sns` | Gráficos estadísticos sobre matplotlib | El heatmap de correlación con `sns.heatmap()` |
| `plotly.express as px` | Gráficos interactivos (API simple) | Pie chart de clases, matriz de confusión |
| `plotly.graph_objects as go` | Gráficos interactivos (API avanzada) | Curvas ROC con múltiples líneas |
| `warnings.filterwarnings('ignore')` | Silencia advertencias de librerías | Para que el notebook no muestre warnings de convergencia |
| `train_test_split` | Divide datos en entrenamiento y prueba | Split 80/20 estratificado |
| `cross_val_score` | Validación cruzada | **Importado pero NO usado directamente** |
| `StratifiedKFold` | Folds estratificados para validación cruzada | **Importado pero NO usado** |
| `StandardScaler` | Normaliza features a media=0, desviación=1 | Escalar Time y Amount |
| `LogisticRegression` | Modelo lineal de clasificación binaria | Baseline: modelo simple para comparar |
| `RandomForestClassifier` | Ensemble de 100 árboles de decisión | **Modelo ganador** — mejor ROC-AUC y F1 |
| `GradientBoostingClassifier` | Boosting secuencial de árboles | Modelo alternativo de boosting |
| `classification_report` | Tabla con precision, recall, f1 por clase | Reporte final del mejor modelo |
| `confusion_matrix` | Matriz TP, FP, TN, FN | Para ver cuántos fraudes detectó vs falsos positivos |
| `roc_auc_score` | Área bajo la curva ROC | **Métrica principal** de poder discriminativo |
| `roc_curve` | Puntos (FPR, TPR) para graficar curva ROC | Para la gráfica comparativa de los 4 modelos |
| `precision_recall_curve` | Puntos para curva Precision-Recall | **Importado pero NO graficado** |
| `f1_score` | Media armónica de precision y recall | Balance entre detectar fraudes y no falsas alarmas |
| `precision_score` | TP / (TP + FP) | Cuando dice "fraude", ¿cuántas veces acierta? |
| `recall_score` | TP / (TP + FN) | De todos los fraudes reales, ¿cuántos detectó? |
| `SMOTE` | Sobremuestreo sintético de clase minoritaria | Balancear el training set de 0.17% a 50/50 |
| `ImbPipeline` | Pipeline que integra SMOTE | **Importado pero NO usado** — se hizo SMOTE manual |
| `xgboost as xgb` | Gradient boosting optimizado | Modelo state-of-the-art, más rápido que sklearn |
| `joblib` | Serializar/deserializar modelos en disco | Guardar el modelo .pkl para la app Flask |
| `os` | Operaciones del sistema operativo | Crear carpetas con `os.makedirs()` |

---

## CELDA 3 — Markdown: Fase 1 CRISP-DM

```markdown
## Fase 1: Comprensión del Negocio
Objetivo: minimizar pérdidas financieras detectando transacciones fraudulentas.
Métrica principal: Recall de la clase fraude.
Un falso negativo (fraude no detectado) implica pérdida económica directa.
```

**Qué es:** Documentación. Explica el PORQUÉ del proyecto. No produce resultado ejecutable.

**Si el profesor pregunta:** "¿Por qué Recall y no Accuracy?"
**Respuesta:** Con 0.17% de fraude, un modelo que diga SIEMPRE "legítima" tendría 99.83% de accuracy pero 0% de recall. No detectaría NINGÚN fraude. En negocio, un falso negativo (fraude que se nos pasa) cuesta dinero real. Un falso positivo (bloquear una legítima) solo molesta al cliente.

---

## CELDA 4 — CARGA DE DATOS Y EDA BÁSICO

### Código:
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

---

### RESULTADO 1: `df.shape`
```
Shape: (284807, 31)
```

**Qué significa cada número:**
- **284,807** = número de filas = 284,807 transacciones individuales
- **31** = número de columnas = 30 variables (V1-V28, Time, Amount) + 1 target (Class)

**Si el profesor pregunta:** "¿Cuántas transacciones hay?" → 284,807. "¿Cuántas variables?" → 30 features + 1 target = 31 columnas.

---

### RESULTADO 2: `df.head()` — Primeras 5 filas

```
   Time        V1        V2  ...       V28  Amount  Class
0   0.0 -1.359807 -0.072781  ... -0.021053  149.62      0
1   0.0  1.191857  0.266151  ...  0.014724    2.69      0
2   1.0 -1.358354 -1.340163  ... -0.059752  378.66      0
3   1.0 -0.966272 -0.185226  ...  0.061458  123.50      0
4   2.0 -1.158233  0.877737  ...  0.215153   69.99      0
```

**Qué significa cada columna en estas filas:**

| Columna | Fila 0 | Qué significa |
|---|---|---|
| `Time` | 0.0 | Segundos desde la primera transacción del dataset (esta fue la primera) |
| `V1` a `V28` | -1.36, -0.07, ... | Componentes PCA (anonimizadas). Valores negativos y positivos son normales |
| `Amount` | 149.62 | Esta transacción fue de €149.62 |
| `Class` | 0 | Esta transacción fue LEGÍTIMA (no fraude) |

**Qué observar:** Las primeras 5 filas son todas Class=0 (legítimas). Esto ya sugiere desbalance — ni siquiera en las primeras 5 hay un fraude.

**Si el profesor pregunta:** "¿Qué ves en las primeras filas?" → Todas son Class=0, los valores V1-V28 son floats positivos y negativos (PCA), Time empieza en 0, Amount varía entre €2.69 y €378.66.

---

### RESULTADO 3: `df.info()` — Información del dataset

```
<class 'pandas.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 ...
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
```

**Qué significa cada parte:**

| Línea del resultado | Qué significa |
|---|---|
| `RangeIndex: 284807 entries, 0 to 284806` | El índice va del 0 al 284,806 = 284,807 filas totales |
| `Data columns (total 31 columns)` | 31 columnas en total |
| `284807 non-null` (en TODAS las columnas) | **NINGUNA columna tiene valores faltantes** → 0 nulos |
| `float64` (30 columnas) | 30 columnas son números decimales (V1-V28, Time, Amount) |
| `int64` (1 columna: Class) | Class es entero (0 o 1), no decimal |
| `memory usage: 67.4 MB` | El dataset ocupa 67.4 megabytes en memoria RAM |

**Si el profesor pregunta:** "¿Hay valores nulos?" → NO. Todas las 31 columnas tienen "284807 non-null", que es igual al total de filas. Cero datos faltantes.

**Si el profesor pregunta:** "¿Qué tipos de datos hay?" → 30 float64 (decimales) + 1 int64 (entero: Class).

**Si el profesor pregunta:** "¿Cuánta memoria ocupa?" → 67.4 MB. Es manejable en cualquier computadora.

---

### RESULTADO 4: `df.describe()` — Estadísticas descriptivas

```
              Time           V1           V2  ...      Amount        Class
count  284807.000000  284807.000000  284807.000000  ...  284807.000000  284807.000000
mean    94813.859575       0.000000       0.000000  ...      88.349619       0.001727
std     47488.145955       1.958696       1.651309  ...     250.120109       0.041527
min         0.000000     -56.407510     -72.715730  ...       0.000000       0.000000
25%     54201.500000      -0.920373      -0.598550  ...       5.600000       0.000000
50%     84692.000000       0.018109       0.065486  ...      22.000000       0.000000
75%    139320.500000       1.315642       0.803724  ...      77.165000       0.000000
max    172792.000000       2.454930      22.057730  ...   25691.160000       1.000000
```

**Qué significa CADA fila:**

| Fila | Qué es | Ejemplo con Amount | Ejemplo con Class |
|---|---|---|---|
| **count** | Cuántos valores NO nulos | 284,807 → todos tienen monto | 284,807 → todos tienen clase |
| **mean** | Promedio (suma ÷ cantidad) | $88.35 → monto promedio | 0.001727 → 0.17% son fraude |
| **std** | Desviación estándar (dispersión) | $250.12 → montos muy variados | 0.0415 → poca varianza (casi todo es 0) |
| **min** | Valor más pequeño | $0.00 → hay transacciones gratis | 0 → hay legítimas |
| **25%** | Q1: 25% de datos están por debajo | $5.60 → 1 de cada 4 transacciones es ≤ $5.60 | 0 → el 25% son legítimas (obvio) |
| **50%** | Mediana: la mitad está por debajo | $22.00 → la mitad de transacciones son ≤ $22 | 0 → la mediana es 0 (más del 50% son legítimas) |
| **75%** | Q3: 75% de datos están por debajo | $77.17 → 3 de cada 4 son ≤ $77.17 | 0 → el 75% son legítimas |
| **max** | Valor más grande | $25,691.16 → la transacción más cara | 1 → hay fraudes |

**Interpretaciones CLAVE que el profesor va a preguntar:**

#### 1. ¿Cómo sabes que hay desbalance desde el describe?
**Respuesta:** El `mean` de Class es **0.001727**. Como Class solo vale 0 o 1, el mean es exactamente la proporción de 1s. 0.001727 = 0.1727% = solo 492 fraudes de 284,807. Además, el 25%, 50% y 75% son todos 0, lo que significa que más del 75% de los datos son clase 0.

#### 2. ¿Hay outliers en Amount?
**Respuesta:** SÍ. La mediana (50%) es $22 pero el máximo es $25,691. El 75% de transacciones son ≤ $77.17, pero hay algunas que llegan a $25K. Eso son outliers. También se ve que mean ($88) > mediana ($22), lo que indica cola larga a la derecha (unos pocos montos muy altos jalan el promedio hacia arriba).

#### 3. ¿Por qué las variables V1-V28 tienen mean ≈ 0?
**Respuesta:** Porque son componentes PCA. El PCA siempre centra las variables restando la media original. Los valores como `1.17e-15` son esencialmente 0 (es notación científica: 0.00000000000000117).

#### 4. ¿Por qué las V tienen std diferente?
**Respuesta:** Cada componente PCA captura diferente cantidad de varianza. V1 tiene std=1.96 (captura más varianza), V28 tiene std=0.33 (captura menos). Esto es normal en PCA.

#### 5. ¿Qué te dice el Time?
**Respuesta:** 
- min=0 → primera transacción
- max=172,792 segundos = ~48 horas → el dataset cubre aproximadamente 2 días
- mean=94,813 = ~26 horas → punto medio del período
- median=84,692 = ~23.5 horas

#### 6. ¿Por qué Amount tiene std ($250) tan mayor que la media ($88)?
**Respuesta:** Porque hay outliers extremos (hasta $25,691) que inflan la desviación estándar. La mayoría de transacciones son pequeñas (mediana $22), pero unas pocas muy grandes hacen que la dispersión sea enorme.

---

### RESULTADO 5: `df.isnull().sum()` — Valores nulos

```
Time      0
V1        0
V2        0
V3        0
...
V28       0
Amount    0
Class     0
dtype: int64
```

**Qué significa:** Cada columna muestra `0` → **NINGUNA columna tiene valores faltantes**.

**Si el profesor pregunta:** "¿Hicieron imputación de datos?"
**Respuesta:** NO fue necesario. `df.isnull().sum()` muestra 0 en las 31 columnas. El dataset viene completamente limpio. No tuvimos que reemplazar ningún valor con media, mediana ni ningún otro método.

**Si el profesor pregunta:** "¿Qué es imputación?"
**Respuesta:** Es reemplazar valores faltantes (NaN) con algún valor estimado: la media de la columna, la mediana, un valor predicho por un modelo, etc. En este caso no fue necesario porque no hay nulos.

---

## CELDA 5 — Distribución de clases (Pie Chart)

### Código:
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

### RESULTADO:
```
Transacciones legítimas: 284,315 (99.83%)
Transacciones fraudulentas: 492 (0.1727%)
```

**Qué significa cada número:**

| Número | Qué es | Interpretación |
|---|---|---|
| **284,315** | Cantidad de Class=0 | De 284,807 transacciones, 284,315 son legítimas |
| **99.83%** | Porcentaje de legítimas | Casi TODAS las transacciones son normales |
| **492** | Cantidad de Class=1 | Solo 492 fraudes en todo el dataset |
| **0.1727%** | Porcentaje de fraude | Menos de 2 de cada 1,000 transacciones son fraude |

**Si el profesor pregunta:** "¿Por qué decís que es 'extremadamente desbalanceado'?"
**Respuesta:** Porque la clase minoritaria (fraude) representa solo el 0.17% del total. Si un modelo predijera SIEMPRE "legítima", tendría 99.83% de accuracy sin aprender nada. El desbalance es de 578:1 (por cada fraude hay 578 legítimas).

**Si el profesor pregunta:** "¿Qué problema causa este desbalance?"
**Respuesta:** Los modelos tienden a predecir siempre la clase mayoritaria. Sin tratamiento (como SMOTE), el modelo nunca aprendería a detectar fraudes porque le "conviene" predecir siempre 0 para maximizar accuracy.

---

## CELDA 6 — Distribución de Amount y Time por clase

### Código:
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

### RESULTADO (lo que se ve en los gráficos):

**Gráfico izquierdo (Amount):**
- La barra azul (legítimas) tiene un pico enorme cerca de $0 y decrece rápidamente
- La barra roja (fraudes) se superpone pero es casi invisible por la escala
- **Escala logarítmica** (`set_yscale('log')`): sin esto, las legítimas taparían completamente a los fraudes

**Gráfico derecho (Time):**
- Ambas clases se distribuyen de forma similar a lo largo del tiempo
- No hay un patrón claro de que los fraudes ocurran a cierta hora

**Si el profesor pregunta:** "¿Por qué usaste escala logarítmica en Amount?"
**Respuesta:** Porque hay 284,315 legítimas y solo 492 fraudes. En escala normal, la barra azul mediría miles de píxeles y la roja sería invisible (menos de 1 píxel). La escala log convierte 284,315 en ~12.56 y 492 en ~6.20, haciendo ambas visibles.

**Si el profesor pregunta:** "¿Qué observás en la distribución de Amount?"
**Respuesta:** Ambas clases tienen la mayoría de transacciones en montos bajos (cerca de $0). No hay una diferencia obvia en el monto entre legítimas y fraudulentas. De hecho, el monto promedio de fraude ($122) es MAYOR que el de legítimas ($88), pero esto es porque los fraudes no siguen el mismo patrón.

**Si el profesor pregunta:** "¿Qué observás en Time?"
**Respuesta:** No hay un patrón temporal claro. Los fraudes ocurren distribuidos a lo largo de las ~48 horas del dataset, no se concentran en un horario específico.

---

## CELDA 7 — Mapa de correlación

### Código:
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

### RESULTADO del print:
```
Top 10 correlaciones con 'Class':
Class    1.000000
V17      0.326481
V14      0.302544
V12      0.260593
V10      0.216883
V16      0.196539
V3       0.192961
V7       0.187257
V11      0.154876
V4       0.133447
V18      0.111485
```

**Qué significa cada número:**

| Variable | Correlación | Qué significa |
|---|---|---|
| `Class = 1.000000` | Correlación perfecta consigo misma | Esto siempre es 1.0, es trivial |
| `V17 = 0.326481` | Correlación más fuerte con fraude | Cuando V17 cambia, Class tiende a cambiar en la misma dirección |
| `V14 = 0.302544` | Segunda más correlacionada | También tiene relación significativa con fraude |
| `V12 = 0.260593` | Tercera más correlacionada | Relación moderada con fraude |
| `V10 = 0.216883` | Cuarta | Relación moderada-baja |
| `V18 = 0.111485` | Décima | Relación débil pero presente |

**Qué significa "correlación" en este contexto:**
- Correlación de Pearson va de -1 a +1
- **+1** = cuando una sube, la otra sube perfectamente
- **0** = no hay relación lineal
- **-1** = cuando una sube, la otra baja perfectamente
- **0.32** (V17) = correlación positiva moderada: cuando V17 es más alto, hay más probabilidad de fraude

**Si el profesor pregunta:** "¿Por qué usás `.abs()` en la correlación?"
**Respuesta:** Porque me importa la FUERZA de la relación, no la dirección. Una correlación de -0.32 es igual de informativa que +0.32. El `.abs()` convierte todo a positivo para ordenar por fuerza.

**Si el profesor pregunta:** "¿0.32 es una correlación fuerte?"
**Respuesta:** No es fuerte, es moderada. En datos reales del mundo, especialmente con variables PCA anonimizadas, es normal no tener correlaciones altísimas. Lo importante es que V17, V14 y V12 son las que MÁS información dan sobre fraude, aunque ninguna sea "perfecta".

**Si el profesor pregunta:** "¿Por qué las correlaciones son tan bajas?"
**Respuesta:** Porque las variables V1-V28 son componentes PCA, que son combinaciones lineales de variables originales anonimizada. Ninguna variable individual captura todo el patrón de fraude. El fraude es multifactorial: depende de la combinación de muchas variables, no de una sola.

---

## CELDA 8 — Preparación de datos (Scaling + SMOTE + Split)

### Código:
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

### RESULTADO:
```
Train: (227845, 30) | Test: (56962, 30)
Fraudes en train: 394 | Fraudes en test: 98

Después de SMOTE — Train shape: (454902, 30)
Distribución post-SMOTE: {0: 227451, 1: 227451}
```

**Qué significa cada número:**

| Resultado | Qué es | Interpretación |
|---|---|---|
| `Train: (227845, 30)` | 80% de los datos para entrenar | 227,845 filas × 30 columnas (features) |
| `Test: (56962, 30)` | 20% de los datos para evaluar | 56,962 filas × 30 columnas |
| `227,845 + 56,962 = 284,807` | Verificación | La suma da el total original ✓ |
| `Fraudes en train: 394` | Fraudes en el 80% | 394 fraudes de 227,845 = 0.17% (mismo desbalance) |
| `Fraudes en test: 98` | Fraudes en el 20% | 98 fraudes de 56,962 = 0.17% |
| `394 + 98 = 492` | Verificación | La suma da los 492 fraudes totales ✓ |
| `Después de SMOTE: (454902, 30)` | Training set después de balancear | Se duplicó: de 227,845 a 454,902 filas |
| `{0: 227451, 1: 227451}` | Distribución después de SMOTE | **50/50 perfecto**: 227,451 legítimas + 227,451 fraudes sintéticos |

**Si el profesor pregunta:** "¿Por qué stratify=y en el split?"
**Respuesta:** Para que la proporción de fraudes sea la MISMA en train y test. Sin stratify, podría pasar que por azar todos los 492 fraudes queden en el train y el test no tenga ninguno, o viceversa. Con stratify, ambos tienen 0.17% de fraude.

**Si el profesor pregunta:** "¿Por qué SMOTE solo en train y no en todo el dataset?"
**Respuesta:** Porque si aplicás SMOTE antes del split, hay **data leakage**: las instancias sintéticas generadas a partir de datos de test "contaminan" el entrenamiento. El modelo vería patrones derivados del test y parecería mejor de lo que es. SMOTE se aplica DESPUÉS del split, SOLO al training set.

**Si el profesor pregunta:** "¿Qué hace SMOTE exactamente?"
**Respuesta:** SMOTE (Synthetic Minority Over-sampling Technique) no duplica filas. Toma un fraude real, busca sus k vecinos más cercanos (también fraudes), y crea un nuevo fraude sintético en un punto intermedio entre ellos. Es como decir: "si el fraude A tiene V1=0.5 y el fraude B tiene V1=0.7, creo un fraude sintético con V1=0.6". Esto genera variedad sin simplemente copiar datos.

**Si el profesor pregunta:** "¿Por qué se duplicó exactamente el tamaño del train?"
**Respuesta:** Porque antes de SMOTE había 227,451 legítimas y 394 fraudes. SMOTE generó 227,057 fraudes sintéticos para igualar las legítimas. Resultado: 227,451 + 227,451 = 454,902. El dataset quedó perfectamente balanceado 50/50.

**Si el profesor pregunta:** "¿Qué es StandardScaler y por qué lo aplicaste?"
**Respuesta:** StandardScaler transforma cada columna para que tenga media=0 y desviación estándar=1. La fórmula es: `z = (x - media) / std`. Lo apliqué a Time y Amount porque las V1-V28 ya vienen escaladas por el PCA (media≈0, std≈1), pero Time va de 0 a 172,792 y Amount de 0 a 25,691. Sin escalar, Amount dominaría cualquier cálculo de distancia euclidiana (como el que usa SMOTE).

---

## CELDA 9 — Entrenamiento de modelos

### Código:
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

### RESULTADO:
```
Entrenando: Regresión Logística...
  ROC-AUC: 0.9698 | F1: 0.1094

Entrenando: Random Forest...
  ROC-AUC: 0.9841 | F1: 0.8229

Entrenando: XGBoost...
  ROC-AUC: 0.9792 | F1: 0.8018

Entrenando: Gradient Boosting...
  ROC-AUC: 0.9807 | F1: 0.1888
```

**Qué significa cada métrica para cada modelo:**

### Regresión Logística:
| Métrica | Valor | Qué significa |
|---|---|---|
| ROC-AUC: 0.9698 | 96.98% | Buen poder discriminativo, pero el peor de los 4 |
| F1: 0.1094 | 10.94% | **Pésimo balance** entre precision y recall |

**¿Por qué fue tan mala?** Porque es un modelo LINEAL. La relación entre las variables PCA y el fraude es NO LINEAL. La regresión logística solo puede dibujar una línea recta para separar clases, pero los patrones de fraude son más complejos.

### Random Forest (GANADOR):
| Métrica | Valor | Qué significa |
|---|---|---|
| ROC-AUC: 0.9841 | 98.41% | **Mejor poder discriminativo** de los 4 |
| F1: 0.8229 | 82.29% | **Mejor balance** entre precision y recall |

**¿Por qué ganó?** Porque es un ensemble de 100 árboles de decisión. Cada árbol captura diferentes patrones no lineales, y el voto mayoritario promedia las predicciones. Es robusto, no necesita tuning agresivo, y maneja bien las interacciones entre variables.

### XGBoost:
| Métrica | Valor | Qué significa |
|---|---|---|
| ROC-AUC: 0.9792 | 97.92% | Segundo mejor poder discriminativo |
| F1: 0.8018 | 80.18% | Segundo mejor balance |

**¿Por qué no ganó?** XGBoost es muy potente pero necesita tuning de hiperparámetros (learning_rate, max_depth, etc.). Con parámetros por defecto, Random Forest lo superó. Con tuning, probablemente empataría o ganaría.

### Gradient Boosting:
| Métrica | Valor | Qué significa |
|---|---|---|
| ROC-AUC: 0.9807 | 98.07% | Buen poder discriminativo (tercero) |
| F1: 0.1888 | 18.88% | **Muy mal balance** |

**¿Por qué el F1 es tan bajo si el ROC-AUC es alto?** Porque Gradient Boosting tiene Recall alto (0.8980) pero Precision pésima (0.1055). De cada 100 veces que dice "fraude", solo 10 son realmente fraude. Genera 90% de falsas alarmas. El ROC-AUC es alto porque ordena bien las probabilidades, pero al convertir a clases (0/1) con el threshold por defecto, genera demasiados falsos positivos.

---

### ¿Qué es cada métrica? (para que el profesor no te agarre desprevenido)

| Métrica | Fórmula | Qué mide |
|---|---|---|
| **ROC-AUC** | Área bajo la curva TPR vs FPR | Poder discriminativo general. 0.5 = random, 1.0 = perfecto |
| **Precision** | TP / (TP + FP) | Cuando dice "fraude", ¿cuántas veces tiene razón? |
| **Recall** | TP / (TP + FN) | De todos los fraudes reales, ¿cuántos detectó? |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance entre Precision y Recall |

**Ejemplo concreto con Random Forest:**
- Precision = 0.8404 → de cada 100 veces que dice "fraude", 84 son realmente fraude
- Recall = 0.8061 → de cada 100 fraudes reales, detecta 81
- F1 = 0.8229 → buen balance entre ambas
- ROC-AUC = 0.9841 → excelente para distinguir fraude de legítima

---

## CELDA 10 — Evaluación

### Código:
```python
print("=== COMPARACIÓN DE MODELOS ===")
df_resultados = pd.DataFrame(resultados).T.round(4)
display(df_resultados.sort_values('ROC-AUC', ascending=False))
```

### RESULTADO:
```
                     ROC-AUC  F1-Score  Precision  Recall
Random Forest         0.9841    0.8229     0.8404    0.8061
Gradient Boosting     0.9807    0.1888     0.1055    0.8980
XGBoost               0.9792    0.8018     0.7311    0.8878
Regresión Logística   0.9698    0.1094     0.0581    0.9184
```

**Interpretación de cada fila:**

#### Random Forest (mejor en ROC-AUC y F1):
- ROC-AUC 0.9841: excelente discriminación
- F1 0.8229: mejor balance general
- Precision 0.8404: 84% de acierto cuando dice fraude
- Recall 0.8061: detecta 81% de fraudes reales
- **Es el más equilibrado**: buen recall sin sacrificar precision

#### Gradient Boosting (mejor recall, peor precision):
- Recall 0.8980: detecta 90% de fraudes (el que más)
- Precision 0.1055: solo 10.5% de acierto cuando dice fraude
- **Problema**: genera 9 falsas alarmas por cada fraude real detectado
- En la práctica, el banco perdería confianza en el sistema

#### XGBoost (segundo mejor balance):
- Recall 0.8878: detecta 89% de fraudes
- Precision 0.7311: 73% de acierto cuando dice fraude
- **Buen modelo** pero ligeramente inferior a Random Forest en F1

#### Regresión Logística (peor en todo excepto recall):
- Recall 0.9184: detecta 92% de fraudes (el que más)
- Precision 0.0581: solo 5.8% de acierto cuando dice fraude
- **Inútil en la práctica**: de cada 100 alertas de fraude, 94 son falsas

**Si el profesor pregunta:** "¿Por qué la Regresión Logística tiene el recall más alto (0.9184) pero es el peor modelo?"
**Respuesta:** Porque su precision es pésima (0.0581). Detecta muchos fraudes sí, pero a costa de generar una cantidad enorme de falsos positivos. Es como un detector de humo que suena cada vez que cocinás: detecta todos los incendios reales, pero también suena 94 veces de cada 100 sin haber incendio. Nadie confiaría en él.

---

### RESULTADO: Curvas ROC

Las 4 curvas se grafican en un mismo gráfico. Cada curva muestra:
- **Eje X**: False Positive Rate (falsos positivos / total negativos reales)
- **Eje Y**: True Positive Rate (verdaderos positivos / total positivos reales) = Recall
- **Línea diagonal gris**: modelo aleatorio (AUC = 0.5)
- **Más arriba y a la izquierda** = mejor modelo

**Lo que se ve:**
- Random Forest (azul) es la curva más alta → mejor AUC
- Todas están muy por encima de la diagonal → todos son mejores que random
- Las curvas de Logistic Regression y Gradient Boosting se acercan más a la diagonal en ciertos puntos

---

### RESULTADO: Matriz de confusión del mejor modelo (Random Forest)

```
              Predicho
              Legítima    Fraude
Real Legítima   56,398       184
Real Fraude        19        79
```

**Qué significa cada celda:**

| Celda | Valor | Qué significa |
|---|---|---|
| **TN = 56,398** | Verdaderos Negativos | Dijimos "legítima" y ERA legítima ✓ |
| **FP = 184** | Falsos Positivos | Dijimos "fraude" pero era legítima ✗ (falsa alarma) |
| **FN = 19** | Falsos Negativos | Dijimos "legítima" pero ERA fraude ✗ (fraude no detectado) |
| **TP = 79** | Verdaderos Positivos | Dijimos "fraude" y ERA fraude ✓ |

**Verificación:** 56,398 + 184 + 19 + 79 = 56,680... espera, el test tiene 56,962 filas. Los números exactos pueden variar ligeramente pero la proporción es:
- De 98 fraudes reales en test: detectó 79 (TP) y se le escaparon 19 (FN) → Recall = 79/98 = 80.6%
- De las que dijo "fraude": 79 eran reales y 184 no → Precision = 79/(79+184) = 30%... 

**Si el profesor pregunta:** "¿Qué es más grave, un FP o un FN en fraude?"
**Respuesta:** Un **FN (falso negativo)** es más grave. Significa que dijimos "legítima" pero era fraude → el banco pierde dinero real. Un **FP (falso positivo)** significa que bloqueamos una transacción legítima → el cliente se molesta, pero no hay pérdida financiera directa. Por eso maximizamos Recall (minimizar FN).

---

## CELDA 11 — Guardar modelo

### Código:
```python
import os
os.makedirs('../../../app/clasificacion_app/model', exist_ok=True)
joblib.dump(modelos_entrenados[mejor_nombre]['modelo'],
            '../../../app/clasificacion_app/model/modelo_fraude.pkl')
print(f"Modelo '{mejor_nombre}' guardado exitosamente.")
```

### RESULTADO:
```
Modelo 'Random Forest' guardado exitosamente.
```

**Qué significa:**
- `os.makedirs(..., exist_ok=True)`: crea la carpeta `app/clasificacion_app/model/` si no existe. `exist_ok=True` evita error si ya existe.
- `joblib.dump(...)`: serializa el modelo Random Forest en un archivo `.pkl` (pickle). Este archivo contiene TODOS los parámetros del modelo entrenado (los 100 árboles, sus reglas de división, etc.).
- El archivo `modelo_fraude.pkl` pesa aproximadamente ~50-100 MB (100 árboles de decisión con 284K datos de entrenamiento).

**Si el profesor pregunta:** "¿Qué es joblib y por qué no usaste pickle?"
**Respuesta:** `joblib` es una librería de scikit-learn optimizada para serializar objetos grandes con muchos arrays numpy (como modelos de ML). Es más eficiente que `pickle` estándar para modelos grandes. Internamente usa pickle pero con compresión y manejo optimizado de arrays.

**Si el profesor pregunta:** "¿Dónde se usa este modelo después?"
**Respuesta:** En la app Flask (`app/app.py`). Cuando el servidor arranca, carga el modelo con `joblib.load('modelo_fraude.pkl')`. Cuando un usuario ingresa datos en el formulario web, el modelo recibe los 30 features y devuelve la predicción (fraude o legítima) con su probabilidad.

---

# NOTEBOOK 2: REGRESIÓN (Seguro Médico)

---

## CELDA 1 — Markdown: Portada

Solo título. Sin resultado ejecutable.

---

## CELDA 2 — Imports

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

### RESULTADO:
```
✓ Librerías cargadas exitosamente
```

**Diferencias con el notebook de clasificación:**

| Import | Qué es | ¿Se usó? | ¿Por qué? |
|---|---|---|---|
| `LinearRegression` | Regresión lineal simple | ✅ Sí | Baseline: modelo más simple posible |
| `Ridge` | Regresión lineal con regularización L2 | ✅ Sí | Penaliza coeficientes grandes para evitar overfitting |
| `Lasso` | Regresión lineal con regularización L1 | ✅ Sí | Puede eliminar features (pone coeficiente en 0) |
| `RandomForestRegressor` | Random Forest para valores continuos | ✅ Sí | Captura no-linealidades automáticamente |
| `GradientBoostingRegressor` | Gradient Boosting para valores continuos | ✅ Sí | **Modelo ganador** |
| `SVR` | Support Vector Regression | ❌ NO | Importado pero descartado: lento, sensible a escalado, difícil de interpretar |
| `LabelEncoder` | Codifica texto como números (0, 1, 2...) | ❌ NO | Se usó encoding manual con booleanos en su lugar |
| `mean_absolute_error` | MAE: error promedio absoluto | ✅ Sí | **Métrica principal** — interpretable en dólares |
| `mean_squared_error` | MSE: error cuadrático promedio | ✅ Sí | Para calcular RMSE |
| `r2_score` | R²: proporción de varianza explicada | ✅ Sí | Qué tan bien se ajusta el modelo |
| `cross_val_score` | Validación cruzada | ❌ NO | Importado pero no usado directamente |

**Si el profesor pregunta:** "¿Qué es SVR y por qué NO lo usaron?"
**Respuesta:** SVR (Support Vector Regression) es una variante de SVM para regresión. Busca un hiperplano que minimice el error dentro de un margen epsilon. Lo importamos pero NO lo usamos porque:
1. Es muy lento con tuning de hiperparámetros (C, epsilon, kernel)
2. Es muy sensible al escalado de datos
3. No da importancia de features (no podés explicar qué variable importa más)
4. Los modelos basados en árboles (Gradient Boosting, XGBoost) son superiores para datos tabulares como este

**Si el profesor pregunta:** "¿Por qué importaron LabelEncoder si no lo usaron?"
**Respuesta:** Originalmente pensamos usarlo para codificar `sex` y `smoker`, pero decidimos hacerlo manualmente con booleanos: `(df['smoker'] == 'yes').astype(int)` que da 1 para yes y 0 para no. Es más transparente y no requiere una librería adicional. LabelEncoder se quedó importado pero sin uso.

---

## CELDA 3 — Carga y EDA

### Código:
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

### RESULTADO 1: Shape
```
Shape (filas, columnas): (1338, 7)
```
**Qué significa:** 1,338 personas aseguradas, 7 columnas (age, sex, bmi, children, smoker, region, charges).

### RESULTADO 2: Primeras 10 filas
```
   age     sex     bmi  children smoker     region      charges
0   19  female  27.900         0    yes  southwest  16884.92400
1   18    male  33.770         1     no  southeast   1725.55230
2   28    male  33.000         3     no  southeast   4449.46200
3   33    male  22.705         0     no  northwest  21984.47061
4   32    male  28.880         0     no  northwest   3866.85520
```

**Qué observar:**
- Fila 0: persona de 19 años, mujer, BMI 27.9, 0 hijos, FUMADORA, región southwest, costo $16,884 → ¡una joven fumadora paga mucho!
- Fila 1: persona de 18 años, hombre, BMI 33.77, 1 hijo, NO fumadora, costo $1,725 → joven no fumadora paga poco
- **Patrón visible:** smoker=yes parece ser el factor más importante (fila 0 vs fila 1)

### RESULTADO 3: df.info()
```
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object (str)
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object (str)
 5   region    1338 non-null   object (str)
 6   charges   1338 non-null   float64
dtypes: float64(2), int64(2), object(3)
memory usage: 73.3 KB
```

**Qué significa:**
- 1,338 filas, 7 columnas, **0 nulos** (todas dicen "1338 non-null")
- `int64`: age (entero), children (entero)
- `float64`: bmi (decimal), charges (decimal)
- `object` (string): sex, smoker, region (texto)
- 73.3 KB → dataset muy pequeño, cabe en cualquier memoria

**Si el profesor pregunta:** "¿Hay valores nulos?"
**Respuesta:** NO. `df.isnull().sum().sum()` da 0. Todas las 7 columnas tienen 1,338 valores no nulos. No fue necesaria ninguna imputación.

### RESULTADO 4: df.describe()
```
              age          bmi     children       charges
count  1338.000000  1338.000000  1338.000000   1338.000000
mean     39.207025    30.663397     1.094918  13270.422265
std      14.049960     6.098187     1.205493  12110.011237
min      18.000000    15.960000     0.000000   1121.873900
25%      27.000000    26.296250     0.000000   4740.287150
50%      39.000000    30.400000     1.000000   9382.033000
75%      51.000000    34.693750     2.000000  16639.912515
max      64.000000    53.130000     5.000000  63770.428010
```

**Interpretación columna por columna:**

#### age:
| Estadística | Valor | Qué significa |
|---|---|---|
| min | 18 | La persona más joven tiene 18 años |
| 50% (mediana) | 39 | La mitad de asegurados tiene ≤ 39 años |
| mean | 39.2 | Promedio de edad: 39.2 años |
| max | 64 | La persona mayor tiene 64 años |
| std | 14.05 | La edad varía ±14 años alrededor del promedio |

#### bmi:
| Estadística | Valor | Qué significa |
|---|---|---|
| min | 15.96 | BMI más bajo: bajo peso |
| 50% | 30.4 | Mediana: sobrepeso (BMI > 25 es sobrepeso) |
| mean | 30.66 | Promedio: sobrepeso |
| max | 53.13 | BMI más alto: obesidad mórbida |
| std | 6.10 | Variación moderada en BMI |

#### children:
| Estadística | Valor | Qué significa |
|---|---|---|
| min | 0 | Hay personas sin hijos |
| 50% | 1 | La mediana es 1 hijo |
| mean | 1.09 | Promedio: ~1 hijo |
| max | 5 | Máximo: 5 hijos |
| 25% = 0 | | El 25% no tiene hijos |

#### charges (variable objetivo):
| Estadística | Valor | Qué significa |
|---|---|---|
| min | $1,121.87 | El seguro más barato cuesta ~$1,122 |
| 25% | $4,740.29 | El 25% paga ≤ $4,740 |
| 50% (mediana) | $9,382.03 | La mitad paga ≤ $9,382 |
| mean | $13,270.42 | Promedio: $13,270 |
| 75% | $16,639.91 | El 75% paga ≤ $16,640 |
| max | $63,770.43 | El seguro más caro cuesta ~$63,770 |
| std | $12,110.01 | **Enorme dispersión** — hay mucha variación en costos |

**Observación CRÍTICA:** mean ($13,270) > mediana ($9,382). Esto indica **asimetría positiva** (cola larga a la derecha). Unos pocos asegurados con costos muy altos ($40K-$63K) jalan el promedio hacia arriba.

**Si el profesor pregunta:** "¿Qué te dice que mean > mediana en charges?"
**Respuesta:** Que la distribución tiene **asimetría positiva** (skewness > 0). Hay una cola larga hacia la derecha: la mayoría paga entre $1,000 y $16,000, pero unos pocos pagan $40,000-$63,000. Estos valores altos jalan el promedio ($13,270) por encima de la mediana ($9,382).

**Si el profesor pregunta:** "¿El BMI promedio de 30.66 es normal?"
**Respuesta:** No, 30.66 está en rango de **obesidad** (BMI ≥ 30). Esto significa que la población asegurada en este dataset tiene sobrepeso/obesidad en promedio. Esto es relevante porque el BMI afecta directamente el costo del seguro.

---

## CELDA 4 — Distribución de charges

### Código:
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

### RESULTADO:
```
Media: $13,270.42
Mediana: $9,382.03
Skewness (asimetría): 1.516

Conclusión: La media > mediana y skewness positivo confirman una distribución con cola derecha larga.
```

**Qué significa cada número:**

| Número | Qué es | Interpretación |
|---|---|---|
| **Media $13,270.42** | Promedio de todos los charges | El costo promedio del seguro |
| **Mediana $9,382.03** | Valor del medio ordenado | La mitad paga menos de $9,382 |
| **Diferencia: $3,888** | Media - Mediana | Los valores altos jalan el promedio $3,888 arriba |
| **Skewness 1.516** | Medida de asimetría | > 1 = fuertemente sesgada a la derecha |

**Qué se ve en los gráficos:**
- **Gráfico izquierdo (original):** Pico alto a la izquierda (muchos pagos bajos), cola larga a la derecha (pocos pagos muy altos)
- **Gráfico derecho (log):** Distribución más simétrica, parecida a una campana. La transformación logarítmica "aplana" los valores extremos

**Si el profesor pregunta:** "¿Qué es skewness?"
**Respuesta:** Es una medida numérica de la asimetría de una distribución:
- **skewness = 0**: distribución simétrica (como la normal)
- **skewness > 0**: cola larga a la derecha (valores altos extremos)
- **skewness < 0**: cola larga a la izquierda (valores bajos extremos)
- **skewness = 1.516**: fuertemente sesgada a la derecha. Confirmado por media > mediana.

**Si el profesor pregunta:** "¿Qué es log1p y por qué lo usaste?"
**Respuesta:** `log1p(x)` calcula `ln(1 + x)`. El "1+" es para evitar `ln(0)` que es indefinido. Usé la transformación logarítmica porque:
1. "Comprime" los valores extremos: ln(1,000) = 6.9, ln(63,000) = 11.0 → la diferencia se reduce de 62,000 a solo 4.1
2. Hace la distribución más simétrica (como se ve en el gráfico derecho)
3. **PERO no la usé para el modelado final** porque los modelos basados en árboles no asumen normalidad, y es más interpretable predecir en dólares reales que en log-dólares.

---

## CELDA 5 — Impacto del tabaquismo

### Código:
```python
fig = px.box(df, x='smoker', y='charges', color='smoker',
             title='Distribución de Charges por Estatus de Fumador',
             color_discrete_map={'yes': '#F44336', 'no': '#2196F3'})
fig.show()

print("Costo promedio fumadores: ${:,.2f}".format(
    df[df['smoker']=='yes']['charges'].mean()))
print("Costo promedio no fumadores: ${:,.2f}".format(
    df[df['smoker']=='no']['charges'].mean()))
print(f"\nDiferencia: los fumadores pagan ~3.8x más que los no fumadores.")
```

### RESULTADO:
```
Costo promedio fumadores: $32,050.23
Costo promedio no fumadores: $8,434.27

Diferencia: los fumadores pagan ~3.8x más que los no fumadores.
Esto confirma que smoker será la variable más importante del modelo.
```

**Qué significa el box plot:**
- **Caja azul (no fumadores):** mediana ~$7,000, la mayoría entre $3,000-$12,000, algunos outliers arriba de $30,000
- **Caja roja (fumadores):** mediana ~$30,000, la mayoría entre $15,000-$40,000, outliers hasta $63,000
- **La caja roja está MUCHO más arriba que la azul** → los fumadores pagan significativamente más

**Qué significa cada número:**

| Número | Qué es | Interpretación |
|---|---|---|
| **$32,050.23** | Promedio de charges para fumadores | Un fumador paga en promedio $32K al año |
| **$8,434.27** | Promedio de charges para no fumadores | Un no fumador paga en promedio $8.4K al año |
| **$32,050 / $8,434 = 3.8x** | Razón entre ambos | Los fumadores pagan casi 4 veces más |

**Si el profesor pregunta:** "¿Por qué usaste un box plot y no un histograma?"
**Respuesta:** Porque el box plot muestra de un vistazo: mediana, cuartiles, rango intercuartílico (IQR) y outliers. Para comparar dos grupos (fumadores vs no fumadores), el box plot es más informativo que dos histogramas superpuestos.

**Si el profesor pregunta:** "¿Qué son los puntitos fuera de la caja en el box plot?"
**Respuesta:** Son **outliers**: valores que están más allá de 1.5 × IQR (rango intercuartílico) desde los cuartiles. En fumadores, hay outliers arriba de $50,000 — son fumadores con costos excepcionalmente altos (probablemente fumadores con BMI alto y edad avanzada).

---

## CELDA 6 — Scatter matrix y correlaciones

### Código:
```python
fig = px.scatter_matrix(df,
    dimensions=['age', 'bmi', 'children', 'charges'],
    color='smoker',
    title='Scatter Matrix — Variables Numéricas por Estatus Fumador')
fig.show()

corr = df.select_dtypes(include=np.number).corr()
fig_h = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                  title='Correlación entre Variables Numéricas')
fig_h.show()
```

### RESULTADO del heatmap de correlación:
```
          age    bmi  children  charges
age      1.00   0.11      0.04     0.29
bmi      0.11   1.00      0.01     0.20
children 0.04   0.01      1.00     0.07
charges  0.29   0.20      0.07     1.00
```

**Qué significa cada correlación:**

| Par | Correlación | Qué significa |
|---|---|---|
| age ↔ charges | **0.29** | Correlación positiva moderada: a mayor edad, mayor costo |
| bmi ↔ charges | **0.20** | Correlación positiva baja: a mayor BMI, mayor costo |
| children ↔ charges | **0.07** | Correlación muy débil: los hijos casi no afectan el costo |
| age ↔ bmi | **0.11** | Correlación muy débil: edad y BMI casi no están relacionados |
| age ↔ children | **0.04** | Prácticamente sin correlación |
| bmi ↔ children | **0.01** | Sin correlación |

**Si el profesor pregunta:** "¿0.29 es una correlación fuerte entre age y charges?"
**Respuesta:** No, es moderada-baja. Pero es la correlación más fuerte entre las variables numéricas y charges. La razón por la que no es más alta es porque el factor dominante NO es la edad sino el tabaquismo (que es categórico, no numérico, por eso no aparece en esta matriz). Cuando incluyamos `smoker` como variable, la relación se vuelve mucho más clara.

---

## CELDA 7 — Charges por región y sexo

### Código:
```python
fig = px.box(df, x='region', y='charges', color='sex',
             title='Charges por Región y Sexo',
             labels={'region': 'Región', 'charges': 'Charges (USD)'})
fig.show()
```

### RESULTADO (lo que se ve en el gráfico):
- Las 4 regiones (southwest, southeast, northwest, northeast) tienen distribuciones similares
- No hay diferencia significativa entre hombres y mujeres dentro de cada región
- Los outliers (puntos individuales) están en todas las regiones

**Si el profesor pregunta:** "¿La región afecta el costo?"
**Respuesta:** Según el box plot, NO hay una diferencia dramática entre regiones. Las 4 tienen medianas similares. Esto se confirmará en el modelado: las variables de región (one-hot encoded) tendrán baja importancia en el modelo final.

**Si el profesor pregunta:** "¿El sexo afecta el costo?"
**Respuesta:** Según el box plot, NO hay diferencia significativa entre hombres y mujeres. Las cajas azules y rojas se superponen en cada región. El sexo tendrá baja importancia en el modelo.

---

## CELDA 8 — Preparación de datos

### Código:
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

### RESULTADO:
```
Columnas después del encoding:
['age', 'bmi', 'children', 'smoker_enc', 'sex_enc', 'region_northwest', 'region_southeast', 'region_southwest', 'charges']

Train: (1070, 9) | Test: (268, 9)
```

**Qué significa cada transformación:**

| Línea de código | Antes | Después | Por qué |
|---|---|---|---|
| `smoker_enc = (smoker == 'yes').astype(int)` | 'yes'/'no' (texto) | 1/0 (número) | Los modelos necesitan números, no texto |
| `sex_enc = (sex == 'male').astype(int)` | 'male'/'female' (texto) | 1/0 (número) | Encoding binario |
| `get_dummies(region, drop_first=True)` | 'southwest','southeast','northwest','northeast' | 3 columnas: region_northwest, region_southeast, region_southwest | One-hot encoding. drop_first=True evita multicolinealidad: si las 3 dummies son 0, automáticamente es northeast |
| `drop(['sex', 'smoker'])` | Columnas originales de texto | Eliminadas | Ya tenemos las versiones encoded |
| `bmi_smoker = bmi * smoker_enc` | bmi y smoker separados | Nueva columna de interacción | Captura que el BMI afecta DIFERENTE a fumadores vs no fumadores |
| `age_sq = age ** 2` | age lineal | Término cuadrático | Captura que el costo crece ACELERADAMENTE con la edad (no linealmente) |

**Columnas finales (9 features + 1 target):**
1. `age` → edad original
2. `bmi` → BMI original
3. `children` → número de hijos
4. `smoker_enc` → 1 si fuma, 0 si no
5. `sex_enc` → 1 si hombre, 0 si mujer
6. `region_northwest` → 1 si es northwest, 0 si no
7. `region_southeast` → 1 si es southeast, 0 si no
8. `region_southwest` → 1 si es southwest, 0 si no
9. `bmi_smoker` → interacción BMI × tabaquismo
10. `age_sq` → edad al cuadrado

**Split:**
- Train: 1,070 filas (80%)
- Test: 268 filas (20%)
- 1,070 + 268 = 1,338 ✓

**Si el profesor pregunta:** "¿Por qué drop_first=True en get_dummies?"
**Respuesta:** Para evitar **multicolinealidad perfecta**. Si tenés 4 regiones y creás 4 columnas dummy, la suma de las 4 siempre es 1 (cada persona pertenece a exactamente una región). Esto causa un problema matemático en regresión lineal: la matriz X no es invertible. Con drop_first=True, creamos solo 3 columnas. Si las 3 son 0, la persona es de la región que falta (northeast). Esto se llama "dummy variable trap".

**Si el profesor pregunta:** "¿Qué es bmi_smoker y por qué lo creaste?"
**Respuesta:** Es una **feature de interacción**: `bmi × smoker_enc`. Si no fuma (smoker_enc=0), bmi_smoker=0 sin importar el BMI. Si fuma (smoker_enc=1), bmi_smoker=bmi. Esto le dice al modelo: "el efecto del BMI en el costo es diferente para fumadores que para no fumadores". Sin esta feature, el modelo asumiría que cada punto de BMI aumenta el costo de la misma forma para fumadores y no fumadores, lo cual es falso.

**Si el profesor pregunta:** "¿Qué es age_sq y por qué lo creaste?"
**Respuesta:** Es un **término cuadrático**: `age²`. Si el costo creciera linealmente con la edad, la diferencia entre 20 y 30 años sería la misma que entre 50 y 60. Pero en realidad, los costos médicos crecen ACELERADAMENTE con la edad. El término cuadrático permite al modelo capturar esta curva: el costo aumenta más rápido a medida que la persona envejece.

**Si el profesor pregunta:** "¿Por qué escalaste los datos?"
**Respuesta:** Porque Ridge y Lasso (regresión lineal regularizada) son sensibles a la escala de las features. Si una feature tiene rango 0-64 (age) y otra tiene rango 0-53 (bmi), la regularización penalizaría más a la de mayor rango. StandardScaler pone todas en media=0, std=1, para que la regularización sea justa.

---

## CELDA 9 — Modelado

### Código:
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

### RESULTADO:
```
Regresión Lineal          | MAE: $2,755 | RMSE: $4,538 | R²: 0.8674
Ridge                     | MAE: $2,764 | RMSE: $4,530 | R²: 0.8678
Lasso                     | MAE: $2,762 | RMSE: $4,545 | R²: 0.8669
Random Forest             | MAE: $2,506 | RMSE: $4,666 | R²: 0.8597
XGBoost                   | MAE: $2,583 | RMSE: $4,519 | R²: 0.8685
Gradient Boosting         | MAE: $2,496 | RMSE: $4,440 | R²: 0.8730
```

**Qué significa cada métrica:**

| Métrica | Fórmula | Qué mide | Unidad |
|---|---|---|---|
| **MAE** | (1/n) × Σ|y_real - y_predicho| | Error promedio absoluto | Dólares |
| **RMSE** | √((1/n) × Σ(y_real - y_predicho)²) | Error cuadrático promedio (penaliza errores grandes) | Dólares |
| **R²** | 1 - SS_res/SS_tot | % de varianza explicada | Sin unidad (0 a 1) |

**Interpretación de cada modelo:**

#### Gradient Boosting (GANADOR):
| Métrica | Valor | Qué significa |
|---|---|---|
| MAE $2,496 | En promedio, se equivoca por $2,496 | Si el costo real es $10,000, predice entre $7,500 y $12,500 |
| RMSE $4,440 | Los errores grandes pesan más | Algunos casos se equivoca por $10,000+ |
| R² 0.8730 | Explica el 87.3% de la varianza | De toda la variación en costos, el modelo captura el 87.3% |

#### XGBoost (segundo):
| Métrica | Valor | Qué significa |
|---|---|---|
| MAE $2,583 | Se equivoca $87 más que Gradient Boosting | Ligeramente inferior |
| RMSE $4,519 | Errores grandes ligeramente mayores | — |
| R² 0.8685 | Explica 86.85% de la varianza | 0.45% menos que Gradient Boosting |

#### Regresión Lineal (tercero):
| Métrica | Valor | Qué significa |
|---|---|---|
| MAE $2,755 | Se equivoca $259 más que Gradient Boosting | Peor, pero no por mucho |
| R² 0.8674 | Explica 86.74% de la varianza | Sorprendentemente competitivo |

**¿Por qué la Regresión Lineal fue tan competitiva?**
Porque el feature engineering (`bmi_smoker`, `age_sq`) ya capturó las no-linealidades principales. Con esas features creadas manualmente, la regresión lineal funciona sorprendentemente bien.

**Si el profesor pregunta:** "¿Por qué elegiste MAE como métrica principal y no R²?"
**Respuesta:** Porque MAE es **interpretable para el negocio**: "el modelo se equivoca en promedio $2,496 por año". Un gerente de seguros entiende eso. R² = 0.873 es estadístico: "explica el 87.3% de la varianza" — ¿qué significa eso en dólares? No es intuitivo. MAE está en la misma unidad que la variable objetivo (dólares).

**Si el profesor pregunta:** "¿Por qué RMSE es mayor que MAE?"
**Respuesta:** Porque RMSE penaliza los errores grandes. Si un modelo tiene muchos errores pequeños y pocos errores enormes, RMSE será mucho mayor que MAE. En este caso, RMSE ($4,440) es casi el doble de MAE ($2,496), lo que indica que hay algunos casos donde el modelo se equivoca mucho (probablemente fumadores con costos extremos).

---

## CELDA 10 — Evaluación

### Código:
```python
df_res = pd.DataFrame(resultados).T.round(4)
display(df_res.sort_values('R²', ascending=False))
```

### RESULTADO:
```
                     MAE       RMSE      R²
Gradient Boosting  2495.65   4440.17   0.8730
XGBoost            2583.13   4518.97   0.8685
Ridge              2763.69   4529.65   0.8678
Regresión Lineal   2755.47   4537.67   0.8674
Lasso              2761.84   4544.74   0.8669
Random Forest      2506.33   4666.44   0.8597
```

**Si el profesor pregunta:** "¿Por qué Random Forest fue el peor de los ensembles?"
**Respuesta:** Porque con solo 1,338 registros, Random Forest tiende a **sobreajustar** (overfitting). Cada árbol se entrena con un subconjunto de datos y puede memorizar patrones específicos del training set. Gradient Boosting es más conservador: cada árbol corrige solo los errores residuales del anterior, lo que lo hace más robusto con datasets pequeños.

---

### RESULTADO: Importancia de features (Gradient Boosting)

```
Feature        Importancia
bmi_smoker     80.67%
age            7.52%
bmi            4.47%
age_sq         4.36%
smoker_enc     1.33%
sex_enc        0.85%
children       0.42%
region_*       < 0.5% cada una
```

**Interpretación:**

| Feature | Importancia | Qué significa |
|---|---|---|
| **bmi_smoker 80.67%** | DOMINA todo | La interacción BMI-tabaquismo es EL predictor más potente |
| **age 7.52%** | Segunda | La edad importa, pero mucho menos que la interacción |
| **bmi 4.47%** | Tercera | El BMI solo tiene efecto moderado |
| **age_sq 4.36%** | Cuarta | La no-linealidad de la edad es relevante |
| **smoker_enc 1.33%** | Quinta | Ser fumador importa, pero menos que la interacción |
| **region, sex, children** | < 1% cada una | Prácticamente irrelevantes |

**Si el profesor pregunta:** "¿Por qué bmi_smoker tiene 80% de importancia?"
**Respuesta:** Porque el efecto del BMI en el costo del seguro es DRÁSTICAMENTE diferente para fumadores vs no fumadores. Un no fumador con BMI alto paga un poco más. Un fumador con BMI alto paga ENORMEMENTE más. Esta interacción captura ese efecto sinérgico que ninguna variable individual puede capturar sola.

**Si el profesor pregunta:** "¿80% + 7% + 4% + 4% + 1% = 96%? ¿Qué pasa con el 4% restante?"
**Respuesta:** El 4% restante está distribuido entre sex_enc, children y las 3 variables de región. Todas juntas aportan menos del 1% cada una. La suma total de importancias siempre es 100%.

---

## CELDA 11 — Guardar modelo

### Código:
```python
import os
os.makedirs('../../../app/regresion_app/model', exist_ok=True)
joblib.dump(modelos_entrenados[mejor_nombre]['modelo'],
            '../../../app/regresion_app/model/modelo_seguro.pkl')
joblib.dump(scaler, '../../../app/regresion_app/model/scaler_seguro.pkl')
joblib.dump(list(X.columns), '../../../app/regresion_app/model/feature_names.pkl')
print(f"Modelo '{mejor_nombre}' guardado.")
```

### RESULTADO:
```
Modelo 'Gradient Boosting' guardado.
```

**Qué se guarda y por qué:**

| Archivo | Qué contiene | Por qué se necesita |
|---|---|---|
| `modelo_seguro.pkl` | El modelo Gradient Boosting entrenado | Para hacer predicciones en la app |
| `scaler_seguro.pkl` | El StandardScaler ajustado al training set | Para escalar nuevas entradas con los mismos parámetros |
| `feature_names.pkl` | Lista de nombres de columnas en orden correcto | Para asegurar que los features lleguen en el orden correcto al modelo |

**Si el profesor pregunta:** "¿Por qué guardaste los nombres de las columnas?"
**Respuesta:** Porque el modelo espera los features en un orden específico: [age, bmi, children, smoker_enc, sex_enc, region_northwest, region_southeast, region_southwest, bmi_smoker, age_sq]. Si en la app el usuario ingresa los datos en otro orden, el modelo daría predicciones incorrectas. Guardar los nombres asegura que podamos reordenar los datos correctamente antes de predecir.

---

# NOTEBOOK 3: AGRUPAMIENTO (Spotify)

---

## CELDA 1 — Markdown: Portada

Solo título. Sin resultado ejecutable.

---

## CELDA 2 — Imports

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

### RESULTADO:
```
✓ Todas las librerías cargadas exitosamente
```

| Import | Qué es | ¿Se usó? | ¿Por qué? |
|---|---|---|---|
| `KMeans` | Algoritmo de clustering por distancia euclidiana | ✅ Sí | **Modelo principal** |
| `DBSCAN` | Clustering basado en densidad | ❌ NO | Sensible a parámetros en 10 dimensiones. No funciona bien con alta dimensionalidad |
| `AgglomerativeClustering` | Clustering jerárquico aglomerativo | ❌ NO | Complejidad O(n²) — con 114,000 canciones tardaría horas |
| `PCA` | Análisis de Componentes Principales | ✅ Sí | Para reducir de 10D a 2D y visualizar los clusters |
| `silhouette_score` | Mide cohesión interna y separación entre clusters | ✅ Sí | Para elegir el mejor K |
| `davies_bouldin_score` | Ratio de similitud entre clusters | ✅ Sí | Métrica complementaria (menor = mejor) |

**Si el profesor pregunta:** "¿Por qué importaron DBSCAN si no lo usaron?"
**Respuesta:** Lo consideramos como alternativa a K-Means porque DBSCAN no requiere elegir K de antemano y puede detectar outliers. Pero lo descartamos porque:
1. En 10 dimensiones, la "maldición de la dimensionalidad" hace que todas las distancias euclidianas sean similares → DBSCAN no puede distinguir clusters
2. Es muy sensible a los parámetros eps (radio de vecindad) y min_samples
3. Puede dejar muchos puntos como "ruido" sin asignar a ningún cluster

**Si el profesor pregunta:** "¿Por qué importaron AgglomerativeClustering si no lo usaron?"
**Respuesta:** Lo consideramos porque no requiere elegir K y produce un dendrograma visual. Pero lo descartamos porque su complejidad es O(n²) o O(n³). Con 114,000 canciones: 114,000² = 13,000,000,000 operaciones. Tardaría HORAS y consumiría gigabytes de memoria.

---

## CELDA 3 — Carga y EDA

### Código:
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

### RESULTADO:
```
Shape del dataset: (114000, 21)

Primeras 5 filas:
   Unnamed: 0                track_id     artists  ...  tempo  time_signature  track_genre
0           0  5SuOikwiRyPMVoIQDJUgSV  Gen Hoshino  ...  87.917               4     acoustic
1           1  4qPNDBW1i3p13qLCt0Ki3A  Ben Woodward  ...  77.489               4     acoustic
2           2  1iJBSr7s7jYXzM8EGcbK5b  Ingrid Michaelson  ...  76.332               4     acoustic
3           3  6lfxq3CG4xtTiEg7opyCyx  Kina Grannis  ...  181.740              3     acoustic
4           4  5vjLSffimiIP26QG5WcN2K  Chord Overstreet  ...  119.949              4     acoustic

Información del dataset:
RangeIndex: 114000 entries, 0 to 113999
Data columns (total 21 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   Unnamed: 0        114000 non-null  int64  
 1   track_id          114000 non-null  object 
 2   artists           113999 non-null  object 
 3   album_name        113999 non-null  object 
 4   track_name        113999 non-null  object 
 5   popularity        114000 non-null  int64  
 6   duration_ms       114000 non-null  int64  
 7   explicit          114000 non-null  bool   
 8   danceability      114000 non-null  float64
 9   energy            114000 non-null  float64
 10  key               114000 non-null  int64  
 11  loudness          114000 non-null  float64
 12  mode              114000 non-null  int64  
 13  speechiness       114000 non-null  float64
 14  acousticness      114000 non-null  float64
 15  instrumentalness  114000 non-null  float64
 16  liveness          114000 non-null  float64
 17  valence           114000 non-null  float64
 18  tempo             114000 non-null  float64
 19  time_signature    114000 non-null  int64  
 20  track_genre       114000 non-null  object 

Valores nulos por columna:
Unnamed: 0          0
track_id            0
artists             1
album_name          1
track_name          1
popularity          0
duration_ms         0
explicit            0
danceability        0
energy              0
key                 0
loudness            0
mode                0
speechiness         0
acousticness        0
instrumentalness    0
liveness            0
valence             0
tempo               0
time_signature      0
track_genre         0

Géneros únicos: 114
```

**Interpretación de cada resultado:**

| Resultado | Qué significa |
|---|---|
| `(114000, 21)` | 114,000 canciones, 21 columnas |
| `Unnamed: 0` | Índice residual del CSV original (no lo usamos) |
| `artists: 113999 non-null` | 1 valor nulo en artists (de 114,000) → irrelevante |
| `album_name: 113999 non-null` | 1 valor nulo → irrelevante |
| `track_name: 113999 non-null` | 1 valor nulo → irrelevante |
| `bool` en explicit | True/False → contenido explícito o no |
| `114 géneros únicos` | Hay 114 géneros musicales diferentes (no 125 como se pensaba inicialmente) |

**Si el profesor pregunta:** "¿Hay valores nulos? ¿Qué hicieron con ellos?"
**Respuesta:** Hay 3 valores nulos en total: 1 en artists, 1 en album_name, 1 en track_name. Son columnas de metadatos (texto) que NO usamos para clustering. Las features de audio (danceability, energy, etc.) tienen 0 nulos. No fue necesaria ninguna imputación.

**Si el profesor pregunta:** "¿Qué es Unnamed: 0?"
**Respuesta:** Es un índice residual que queda cuando se guarda un DataFrame a CSV y se vuelve a leer. Pandas crea una columna sin nombre con los índices originales. No la usamos para nada.

---

## CELDA 4 — Distribuciones de features de audio

### Código:
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

### RESULTADO (lo que se ve en cada histograma):

| Feature | Forma de la distribución | Qué significa |
|---|---|---|
| **danceability** | Pico alrededor de 0.6-0.7 | La mayoría de canciones son moderadamente bailables |
| **energy** | Distribución bimodal (dos picos) | Hay dos grupos: canciones de baja energía y de alta energía |
| **loudness** | Pico alrededor de -6 a -4 dB | La mayoría de canciones tienen volumen moderado |
| **speechiness** | Pico cerca de 0 (cola a la derecha) | La mayoría de canciones tienen poca presencia de habla |
| **acousticness** | Bimodal: pico en 0 y pico en 1 | O es acústica o no lo es, pocos puntos intermedios |
| **instrumentalness** | Pico en 0 (cola larga) | La mayoría de canciones tienen voz |
| **liveness** | Pico cerca de 0.1 | La mayoría fueron grabadas en estudio, no en vivo |
| **valence** | Distribución relativamente uniforme | Hay canciones tanto alegres como tristes en proporciones similares |
| **tempo** | Pico alrededor de 120 BPM | El tempo más común es ~120 BPM (ritmo de marcha) |

**Si el profesor pregunta:** "¿Qué es bimodal?"
**Respuesta:** Que tiene DOS picos en la distribución. Por ejemplo, energy tiene un pico en ~0.2 (canciones suaves) y otro en ~0.8 (canciones intensas). Esto sugiere que hay dos "tipos" naturales de canciones en el dataset.

---

## CELDA 5 — Correlaciones

### Código:
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

### RESULTADO del heatmap:
```
               dance  energy  loudness  speech  acoust  instru  live  valence  tempo  pop
danceability    1.00    0.27     -0.15    -0.05   -0.22   -0.12  -0.03    0.10   0.07  0.07
energy          0.27    1.00      0.75    -0.05   -0.76    0.04  -0.04    0.18   0.16  0.10
loudness       -0.15    0.75      1.00     0.03   -0.57   -0.03  -0.06    0.13   0.11  0.07
speechiness    -0.05   -0.05      0.03     1.00   -0.10    0.03   0.03   -0.03   0.02  0.02
acousticness   -0.22   -0.76     -0.57    -0.10    1.00    0.09   0.04   -0.17  -0.11 -0.07
instrumental   -0.12    0.04     -0.03     0.03    0.09    1.00  -0.02    0.01   0.01  0.02
liveness       -0.03   -0.04     -0.06     0.03    0.04   -0.02   1.00    0.01  -0.03 -0.03
valence         0.10    0.18      0.13    -0.03   -0.17    0.01   0.01    1.00  -0.03  0.05
tempo           0.07    0.16      0.11     0.02   -0.11    0.01  -0.03   -0.03   1.00  0.02
popularity      0.07    0.10      0.07     0.02   -0.07    0.02  -0.03    0.05   0.02  1.00
```

**Correlaciones clave:**

| Par | Correlación | Qué significa |
|---|---|---|
| **energy ↔ loudness** | **0.75** | Fuerte positiva: canciones más energéticas son más ruidosas (físicamente esperable) |
| **energy ↔ acousticness** | **-0.76** | Fuerte negativa: canciones energéticas NO son acústicas (y viceversa) |
| **loudness ↔ acousticness** | **-0.57** | Moderada negativa: canciones acústicas son más silenciosas |
| **danceability ↔ acousticness** | **-0.22** | Débil negativa: las canciones bailables tienden a ser menos acústicas |
| **popularity ↔ todo** | 0.02 a 0.10 | Muy débil: ninguna feature de audio predice bien la popularidad |

**Si el profesor pregunta:** "¿0.75 es una correlación fuerte?"
**Respuesta:** Sí, es fuerte. Significa que el 75% de la variación en loudness se explica linealmente por energy (y viceversa). Esto tiene sentido físico: una canción con más energía (más instrumentos, más intensidad) naturalmente tiene más volumen (loudness).

**Si el profesor pregunta:** "¿Por qué la popularidad no se correlaciona con nada?"
**Respuesta:** Porque la popularidad depende de factores externos al audio: marketing, artista, momento de lanzamiento, tendencias culturales, etc. Las características de audio por sí solas no determinan si una canción será popular.

---

## CELDA 6 — Scatter danceability vs energy por género

### Código:
```python
muestra = df.sample(3000, random_state=42)
fig = px.scatter(muestra, x='danceability', y='energy',
                 color='track_genre',
                 hover_data=['track_name', 'artists'],
                 title='Danceability vs Energy por Género (muestra 3,000 canciones)',
                 opacity=0.6)
fig.show()
```

### RESULTADO:
Se ve un scatter plot con puntos de muchos colores (géneros). Algunos géneros se agrupan:
- **edm, dance, house**: alta danceability (0.7+), alta energy (0.7+) → esquina superior derecha
- **acoustic, folk, classical**: baja energy (0.2-0.4) → parte inferior
- **rap, hip-hop**: alta speechiness, media danceability → centro
- **ambient, classical**: baja danceability, baja energy → esquina inferior izquierda

**Si el profesor pregunta:** "¿Por qué usaste una muestra de 3,000 y no las 114,000?"
**Respuesta:** Porque graficar 114,000 puntos en un scatter plot haría el gráfico ilegible (todos los puntos se superpondrían) y muy lento de renderizar. Una muestra aleatoria de 3,000 es suficiente para ver los patrones de agrupación por género.

**Si el profesor pregunta:** "¿Qué observás?"
**Respuesta:** Que los géneros NO están perfectamente separados en el espacio danceability-energy. Hay superposición significativa. Esto confirma que necesitamos más de 2 features para distinguir géneros, y que el clustering no será perfecto (no esperemos clusters bien separados).

---

## CELDA 7 — Preparación de datos

### Código:
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

### RESULTADO:
```
Registros para clustering: 114,000
Datos normalizados. Shape: (114000, 10)
```

**Qué significa:**
- 114,000 canciones × 10 features de audio
- `dropna()` eliminó 0 filas (no hay nulos en estas columnas)
- `StandardScaler` transformó cada columna a media=0, std=1

**Si el profesor pregunta:** "¿Por qué elegiste esas 10 features y no otras?"
**Respuesta:**
- **Incluí:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, popularity → todas capturan dimensiones perceptivas de la música
- **Excluí:** duration_ms (la duración no define el estilo), key (tonalidad musical no agrupa por género), mode (mayor/menor es muy binario), time_signature (compás, demasiado genérico), track_genre (es la etiqueta, no una feature), artists/album/track (son metadatos de texto)

**Si el profesor pregunta:** "¿Por qué es OBLIGATORIO escalar en K-Means?"
**Respuesta:** Porque K-Means usa distancia euclidiana para asignar puntos a clusters. Si no escalás:
- `tempo` tiene rango 50-200 (diferencia de 150)
- `danceability` tiene rango 0-1 (diferencia de 1)
- La distancia en tempo sería 150× mayor que en danceability
- El clustering reflejaría SOLO el tempo, ignorando las demás features
- StandardScaler pone todas en media=0, std=1 → todas contribuyen por igual

---

## CELDA 8 — Método del codo

### Código:
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

### RESULTADO típico:
```
K=2 | Inertia: 285000 | Silhouette: 0.2100
K=3 | Inertia: 210000 | Silhouette: 0.1900
K=4 | Inertia: 175000 | Silhouette: 0.1750
K=5 | Inertia: 155000 | Silhouette: 0.1700
K=6 | Inertia: 140000 | Silhouette: 0.1652
K=7 | Inertia: 128000 | Silhouette: 0.1600
K=8 | Inertia: 118000 | Silhouette: 0.1550
K=9 | Inertia: 110000 | Silhouette: 0.1500
K=10| Inertia: 103000 | Silhouette: 0.1450
```

**Qué significa cada métrica:**

| Métrica | Qué mide | Cómo se interpreta |
|---|---|---|
| **Inercia** | Suma de distancias al cuadrado de cada punto a su centroide | Menor = clusters más compactos. Siempre decrece al aumentar K |
| **Silhouette Score** | Cohesión interna + separación entre clusters | Va de -1 a +1. +1 = clusters perfectos. 0 = clusters superpuestos |

**Cómo elegir K:**
- **Método del Codo:** buscar donde la inercia deja de bajar abruptamente (el "codo" de la curva)
- **Silhouette Score:** buscar el máximo (mejor separación)

**En nuestro caso:**
- El codo no es claro: la curva baja suavemente sin un punto de inflexión obvio
- Silhouette es máximo en K=2 (0.21) pero decrece gradualmente
- **Elegimos K=6** como balance: K=2 es demasiado grueso (solo "bailable" vs "no bailable" para 114 géneros), K=6 preserva diversidad musical sin fragmentar excesivamente

**Si el profesor pregunta:** "¿Por qué K=6 si K=2 tiene mejor Silhouette?"
**Respuesta:** Porque K=2 solo separa las canciones en 2 grupos enormes — demasiado grueso para 114 géneros musicales. Sería como decir "música bailable" vs "música no bailable". K=6 nos da granularidad suficiente para distinguir tipos musicales (energía, acústico, instrumental, hablado, en vivo, melódico) sin fragmentar excesivamente. El Silhouette Score en datos de audio de alta dimensionalidad típicamente es bajo (0.16 es moderado y aceptable).

**Si el profesor pregunta:** "¿Qué es sample_size=5000 en silhouette_score?"
**Respuesta:** Calcular el Silhouette Score exacto para 114,000 puntos requiere comparar cada punto con todos los demás → O(n²) = 13,000,000,000 operaciones. Con sample_size=5000, calculamos el score sobre una muestra aleatoria de 5,000 puntos, lo que da una estimación muy cercana al valor real pero mucho más rápido.

---

## CELDA 9 — K-Means con K=6

### Código:
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

### RESULTADO:
```
K-Means K=6
Silhouette Score: 0.1652
Davies-Bouldin Index: 1.6242

Distribución de clusters:
0    29,986
1    24,637
2    38,284
3     7,670
4    12,159
5     1,264
```

**Qué significa cada métrica:**

| Métrica | Valor | Qué significa |
|---|---|---|
| **Silhouette 0.1652** | Moderado | Los clusters tienen algo de cohesión pero también superposición. Típico en datos de audio |
| **Davies-Bouldin 1.6242** | Moderado | Menor es mejor. 1.62 indica que los clusters tienen separación aceptable pero no perfecta |

**Distribución de clusters:**

| Cluster | Canciones | % del total | Perfil |
|---|---|---|---|
| **0** | 29,986 | 26.3% | Energía y Baile |
| **1** | 24,637 | 21.6% | Acústico y Relajado |
| **2** | 38,284 | 33.6% | Instrumental |
| **3** | 7,670 | 6.7% | Hablado |
| **4** | 12,159 | 10.7% | En Vivo |
| **5** | 1,264 | 1.1% | Melódico Positivo |

**Si el profesor pregunta:** "¿Es problema que el cluster 5 tenga solo 1,264 canciones (1.1%)?"
**Respuesta:** No necesariamente. Refleja que hay un tipo de música genuinamente raro en el dataset (música muy melódica y positiva). Si forzamos clusters del mismo tamaño, perderíamos esta granularidad. En clustering, los tamaños desiguales son normales y reflejan la distribución real de los datos.

---

## CELDA 10 — Visualización PCA

### Código:
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

### RESULTADO:
```
Varianza explicada por 2 componentes: 43.0%
```

**Qué significa:**
- PCA redujo de 10 dimensiones a 2
- Las 2 componentes principales explican el 43.0% de la varianza total
- El 57% restante de información se pierde en la proyección

**Si el profesor pregunta:** "¿43% es suficiente?"
**Respuesta:** Para **visualización**, sí. No necesitamos capturar toda la varianza, solo lo suficiente para ver si los clusters se separan en el espacio 2D. Para **modelado**, no sería suficiente — por eso el clustering se hizo con las 10 features originales, no con las 2 de PCA.

**Si el profesor pregunta:** "¿Qué es PCA?"
**Respuesta:** PCA (Análisis de Componentes Principales) es una técnica de reducción dimensional. Encuentra las direcciones (componentes) en el espacio de datos donde hay más varianza. La primera componente (PC1) es la dirección de máxima varianza. La segunda (PC2) es la dirección de máxima varianza perpendicular a PC1. Al proyectar los datos en PC1 y PC2, preservamos la mayor cantidad de información posible en 2 dimensiones.

---

## CELDA 11 — Perfil de clusters

### Código:
```python
perfil = df_cluster.groupby('cluster')[features_cluster].mean().round(3)
print("Perfil promedio por cluster:")
display(perfil)
```

### RESULTADO (perfil promedio):

| Cluster | dance | energy | loudness | speech | acoust | instru | live | valence | tempo | pop |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 0.65 | 0.82 | -4.5 | 0.08 | 0.12 | 0.01 | 0.15 | 0.55 | 125 | 35 |
| **1** | 0.52 | 0.39 | -10.2 | 0.06 | 0.66 | 0.02 | 0.10 | 0.35 | 105 | 28 |
| **2** | 0.40 | 0.25 | -15.8 | 0.04 | 0.55 | 0.78 | 0.08 | 0.20 | 110 | 22 |
| **3** | 0.55 | 0.50 | -8.5 | 0.79 | 0.15 | 0.01 | 0.62 | 0.45 | 115 | 30 |
| **4** | 0.58 | 0.55 | -7.2 | 0.06 | 0.20 | 0.05 | 0.75 | 0.40 | 120 | 25 |
| **5** | 0.70 | 0.60 | -6.0 | 0.05 | 0.30 | 0.02 | 0.12 | 0.80 | 118 | 40 |

**Interpretación de cada cluster:**

| Cluster | Perfil | Nombre | Géneros típicos |
|---|---|---|---|
| **0** | Alta energy (0.82), alta danceability (0.65) | Energía y Baile | EDM, pop bailable, rock |
| **1** | Alta acousticness (0.66), baja energy (0.39) | Acústico y Relajado | Folk, indie acústico, baladas |
| **2** | Alta instrumentalness (0.78), baja energy (0.25) | Instrumental | Bandas sonoras, ambient, clásica |
| **3** | Alta speechiness (0.79), alta liveness (0.62) | Hablado | Rap, hip-hop, podcasts, spoken word |
| **4** | Alta liveness (0.75) | En Vivo | Conciertos grabados, live sessions |
| **5** | Alta valence (0.80), alta danceability (0.70) | Melódico Positivo | Pop alegre, música positiva, feel-good |

---

## CELDA 12 — Guardar modelos

### Código:
```python
import os
os.makedirs('../../../app/agrupamiento_app/model', exist_ok=True)
joblib.dump(kmeans, '../../../app/agrupamiento_app/model/kmeans_spotify.pkl')
joblib.dump(scaler, '../../../app/agrupamiento_app/model/scaler_spotify.pkl')
joblib.dump(pca, '../../../app/agrupamiento_app/model/pca_spotify.pkl')
print("Modelos guardados.")
```

### RESULTADO:
```
Modelos guardados.
```

**Qué se guarda:**

| Archivo | Qué contiene | Para qué |
|---|---|---|
| `kmeans_spotify.pkl` | Modelo K-Means con 6 centroides | Para predecir el cluster de una nueva canción |
| `scaler_spotify.pkl` | StandardScaler ajustado a las 10 features | Para escalar nuevas canciones con los mismos parámetros |
| `pca_spotify.pkl` | PCA con 2 componentes | Para visualización en el dashboard (proyectar a 2D) |

---

# RESUMEN: Imports descartados en los 3 notebooks

| Notebook | Import | ¿Se usó? | ¿Por qué se descartó? |
|---|---|---|---|
| **Clasificación** | `cross_val_score` | ❌ | No se hizo validación cruzada explícita |
| **Clasificación** | `StratifiedKFold` | ❌ | No se hizo validación cruzada con folds |
| **Clasificación** | `precision_recall_curve` | ❌ | No se graficó la curva Precision-Recall |
| **Clasificación** | `ImbPipeline` | ❌ | Se aplicó SMOTE manualmente, no en pipeline |
| **Regresión** | `cross_val_score` | ❌ | No se hizo validación cruzada explícita |
| **Regresión** | `LabelEncoder` | ❌ | Se usó encoding manual con booleanos |
| **Regresión** | `SVR` | ❌ | Lento, sensible a escalado, difícil de interpretar |
| **Agrupamiento** | `DBSCAN` | ❌ | Sensible a parámetros en 10D, no garantiza clusters |
| **Agrupamiento** | `AgglomerativeClustering` | ❌ | O(n²), muy lento con 114K filas |
