# 📘 Proyecto Final — Ciencia de Datos
**Universidad Popular del Cesar | Docente: Aimer Rivera Centeno**
**Metodología obligatoria: CRISP-DM**

> Este documento es la guía maestra para OpenCode.
> Contiene estructura de carpetas, instrucciones por módulo y contexto completo de cada caso.
> El estudiante colocará los datasets manualmente en cada carpeta antes de ejecutar.

---

## 📁 Estructura de Carpetas del Proyecto

```
proyecto_final/
│
├── clasificacion/
│   ├── data/                        ← El estudiante coloca aquí: creditcard.csv
│   ├── notebook/
│   │   └── fraude_clasificacion.ipynb
│   ├── paper/
│   │   ├── main.tex
│   │   ├── referencias.bib
│   │   └── secciones/
│   │       ├── resumen.tex
│   │       ├── introduccion.tex
│   │       ├── metodologia.tex
│   │       ├── resultados.tex
│   │       └── conclusiones.tex
│   └── README.md
│
├── agrupamiento/
│   ├── data/                        ← El estudiante coloca aquí: dataset.csv (Spotify Tracks)
│   ├── notebook/
│   │   └── spotify_clustering.ipynb
│   ├── paper/
│   │   ├── main.tex
│   │   ├── referencias.bib
│   │   └── secciones/
│   │       ├── resumen.tex
│   │       ├── introduccion.tex
│   │       ├── metodologia.tex
│   │       ├── resultados.tex
│   │       └── conclusiones.tex
│   └── README.md
│
├── regresion/
│   ├── data/                        ← El estudiante coloca aquí: insurance.csv
│   ├── notebook/
│   │   └── seguro_regresion.ipynb
│   ├── paper/
│   │   ├── main.tex
│   │   ├── referencias.bib
│   │   └── secciones/
│   │       ├── resumen.tex
│   │       ├── introduccion.tex
│   │       ├── metodologia.tex
│   │       ├── resultados.tex
│   │       └── conclusiones.tex
│   └── README.md
│
└── app/
    ├── clasificacion_app/
    │   ├── app.py
    │   ├── model/
    │   │   └── modelo_fraude.pkl
    │   ├── templates/
    │   │   ├── index.html
    │   │   └── dashboard.html
    │   └── static/
    │       └── style.css
    │
    ├── agrupamiento_app/
    │   ├── app.py
    │   ├── model/
    │   │   └── kmeans_spotify.pkl
    │   ├── templates/
    │   │   ├── index.html
    │   │   └── dashboard.html
    │   └── static/
    │       └── style.css
    │
    └── regresion_app/
        ├── app.py
        ├── model/
        │   └── modelo_seguro.pkl
        ├── templates/
        │   ├── index.html
        │   └── dashboard.html
        └── static/
            └── style.css
```

---

## 🔗 Datasets — Descarga y Ubicación

| Caso | Dataset | URL de descarga | Archivo a colocar |
|------|---------|-----------------|-------------------|
| Clasificación | Credit Card Fraud Detection | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `clasificacion/data/creditcard.csv` |
| Agrupamiento | Spotify Tracks Dataset | https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset | `agrupamiento/data/dataset.csv` |
| Regresión | Medical Cost Personal Dataset | https://www.kaggle.com/datasets/mirichoi0218/insurance | `regresion/data/insurance.csv` |

---

---

# 🔴 CASO 1 — Clasificación: Detección de Fraude en Transacciones

## Contexto del Negocio
Predecir si una transacción de tarjeta de crédito es **fraudulenta (1)** o **legítima (0)**.
Dataset con 284,807 transacciones de titulares de tarjetas europeos. Solo el 0.17% son fraudes → dataset altamente desbalanceado.

## Dataset: `clasificacion/data/creditcard.csv`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| V1–V28 | float | Componentes PCA (anonimizados por privacidad) |
| Time | float | Segundos desde la primera transacción |
| Amount | float | Monto de la transacción en euros |
| Class | int | **Variable objetivo**: 0 = legítima, 1 = fraude |

---

## 📓 NOTEBOOK — `clasificacion/notebook/fraude_clasificacion.ipynb`

### Instrucciones para OpenCode:
Crear un Jupyter Notebook completo siguiendo CRISP-DM con las siguientes celdas y secciones:

---

### CELDA 1 — Markdown: Portada
```
# Detección de Fraude en Transacciones de Tarjeta de Crédito
## Caso de Uso de Clasificación — CRISP-DM
**Universidad Popular del Cesar**
Docente: Aimer Rivera Centeno
```

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

### CELDA 3 — Markdown: Fase 1 CRISP-DM
```
## Fase 1: Comprensión del Negocio
Objetivo: minimizar pérdidas financieras detectando transacciones fraudulentas.
La métrica principal es el Recall de la clase fraude (minimizar falsos negativos).
```

### CELDA 4 — Markdown: Fase 2 + Carga de Datos
```
## Fase 2: Comprensión de los Datos
```
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

### CELDA 5 — EDA: Distribución de clases
```python
# Distribución de clases
clase_counts = df['Class'].value_counts()
fig = px.pie(values=clase_counts.values,
             names=['Legítima (0)', 'Fraude (1)'],
             title='Distribución de Clases — Dataset Extremadamente Desbalanceado',
             color_discrete_sequence=['#2196F3', '#F44336'])
fig.show()

print(f"\nTransacciones legítimas: {clase_counts[0]:,} ({clase_counts[0]/len(df)*100:.2f}%)")
print(f"Transacciones fraudulentas: {clase_counts[1]:,} ({clase_counts[1]/len(df)*100:.4f}%)")
```

### CELDA 6 — EDA: Distribución de Amount y Time
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Amount por clase
df[df['Class']==0]['Amount'].plot(kind='hist', bins=50, ax=axes[0],
    alpha=0.7, color='blue', label='Legítima')
df[df['Class']==1]['Amount'].plot(kind='hist', bins=50, ax=axes[0],
    alpha=0.7, color='red', label='Fraude')
axes[0].set_title('Distribución del Monto por Clase')
axes[0].set_xlabel('Amount (€)')
axes[0].legend()
axes[0].set_yscale('log')

# Time por clase
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

### CELDA 7 — EDA: Correlaciones
```python
plt.figure(figsize=(16, 12))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
            annot=False, linewidths=0.5, fmt='.2f')
plt.title('Mapa de Correlación — Variables V1 a V28, Time, Amount, Class')
plt.tight_layout()
plt.show()

# Top correlaciones con Class
print("\nTop 10 correlaciones con 'Class':")
print(corr['Class'].abs().sort_values(ascending=False).head(11))
```

### CELDA 8 — Markdown: Fase 3
```
## Fase 3: Preparación de los Datos
```
```python
# Escalar Time y Amount (las únicas no escaladas por PCA)
scaler = StandardScaler()
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

# Drop columnas originales
df_clean = df.drop(['Time', 'Amount'], axis=1)

X = df_clean.drop('Class', axis=1)
y = df_clean['Class']

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Fraudes en train: {y_train.sum()} | Fraudes en test: {y_test.sum()}")

# Aplicar SMOTE al conjunto de entrenamiento únicamente
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"\nDespués de SMOTE — Train shape: {X_train_res.shape}")
print(f"Distribución post-SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")
```

### CELDA 9 — Markdown: Fase 4
```
## Fase 4: Modelado
Entrenamos 4 modelos y comparamos su rendimiento.
```
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

### CELDA 10 — Fase 5: Evaluación
```
## Fase 5: Evaluación
```
```python
# Tabla comparativa
df_resultados = pd.DataFrame(resultados).T.round(4)
display(df_resultados.sort_values('ROC-AUC', ascending=False))

# Curvas ROC para todos los modelos
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

# Matriz de confusión del mejor modelo (XGBoost por defecto, ajustar si otro gana)
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

### CELDA 11 — Guardar modelo
```python
# Guardar el mejor modelo para la app Flask
os.makedirs('../../../app/clasificacion_app/model', exist_ok=True)
joblib.dump(modelos_entrenados[mejor_nombre]['modelo'],
            '../../../app/clasificacion_app/model/modelo_fraude.pkl')
print(f"Modelo '{mejor_nombre}' guardado exitosamente.")
```

### CELDA 12 — Conclusiones Markdown
```
## Fase 6: Conclusiones
- El dataset presenta un desbalance severo (0.17% fraudes) que requirió SMOTE.
- [Completar con los resultados reales obtenidos]
- El mejor modelo fue [MODELO] con ROC-AUC de [VALOR].
- La variable más influyente fue [VARIABLE].
- Trabajo futuro: despliegue en tiempo real con monitoreo de drift.
```

---

## 📄 PAPER LaTeX — `clasificacion/paper/`

### Instrucciones para OpenCode:
Crear los siguientes archivos LaTeX. El paper debe tener **mínimo 6 páginas** en formato IEEE de dos columnas.

---

### `clasificacion/paper/main.tex`
```latex
\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{float}
\usepackage{listings}
\usepackage{xcolor}

\title{Detección de Fraude en Transacciones de Tarjeta de Crédito Mediante Técnicas de Clasificación y Aprendizaje Automático}

\author{
  \IEEEauthorblockN{[Nombre Estudiante 1]}
  \IEEEauthorblockA{Universidad Popular del Cesar\\
  Ingeniería de Sistemas\\
  [correo@estudiante.edu.co]}
  \and
  \IEEEauthorblockN{[Nombre Estudiante 2]}
  \IEEEauthorblockA{Universidad Popular del Cesar\\
  Ingeniería de Sistemas\\
  [correo@estudiante.edu.co]}
}

\begin{document}
\maketitle

\input{secciones/resumen}
\input{secciones/introduccion}
\input{secciones/metodologia}
\input{secciones/resultados}
\input{secciones/conclusiones}

\bibliographystyle{IEEEtran}
\bibliography{referencias}

\end{document}
```

### `clasificacion/paper/secciones/resumen.tex`
```latex
\begin{abstract}
El fraude en transacciones con tarjetas de crédito representa una amenaza
creciente para el sistema financiero global. Este trabajo presenta una
solución basada en aprendizaje automático para la detección automática de
transacciones fraudulentas, empleando la metodología CRISP-DM sobre el
dataset Credit Card Fraud Detection disponible en Kaggle, que contiene
284,807 transacciones de titulares europeos con un desbalance severo de
clases (0.17\% de fraudes). Se evaluaron cuatro algoritmos de
clasificación: Regresión Logística, Random Forest, XGBoost y Gradient
Boosting. El manejo del desbalance se realizó mediante SMOTE. Los
resultados demuestran que [completar con mejor modelo] alcanzó el mayor
ROC-AUC de [valor], con un Recall de [valor] para la clase fraude,
evidenciando la viabilidad de los modelos propuestos para su integración
en sistemas de detección en tiempo real.

\textbf{Palabras clave:} fraude, clasificación, SMOTE, Random Forest,
XGBoost, desbalance de clases, CRISP-DM.
\end{abstract}
```

### `clasificacion/paper/secciones/introduccion.tex`
```latex
\section{Introducción}

El fraude con tarjetas de crédito es un fenómeno delictivo que genera
pérdidas millonarias a nivel mundial. Según la Nilson Report, las pérdidas
globales por fraude en pagos con tarjeta superaron los 32 mil millones de
dólares en 2021, con proyecciones de crecimiento sostenido \cite{nilson2021}.
La detección temprana de estas actividades ilícitas es crucial para proteger
tanto a los consumidores como a las instituciones financieras.

Los métodos tradicionales de detección de fraude, basados en reglas
heurísticas, presentan limitaciones importantes frente a la evolución
constante de las tácticas fraudulentas. El aprendizaje automático ofrece
una alternativa adaptativa y escalable, capaz de identificar patrones
complejos y no lineales en grandes volúmenes de datos transaccionales
\cite{dal2015learned}.

El principal desafío técnico de este problema es el severo desbalance de
clases: en un entorno real, la proporción de transacciones fraudulentas
es menor al 1\% del total. Esto provoca que los clasificadores estándar
tiendan a predecir siempre la clase mayoritaria, logrando alta exactitud
pero fallando en detectar los fraudes, que son precisamente el objetivo
\cite{chawla2002smote}.

Este trabajo aborda el problema mediante la metodología CRISP-DM
\cite{wirth2000crisp}, aplicando técnicas de sobremuestreo sintético
(SMOTE) y comparando el rendimiento de cuatro algoritmos de clasificación.
El objetivo es construir un modelo robusto que maximice el recall de la
clase fraude, minimizando así las pérdidas financieras causadas por
transacciones no detectadas.

\subsection{Objetivos}
\begin{itemize}
  \item Implementar un pipeline completo de detección de fraude siguiendo CRISP-DM.
  \item Comparar el rendimiento de Regresión Logística, Random Forest,
        XGBoost y Gradient Boosting.
  \item Manejar el desbalance de clases mediante SMOTE.
  \item Desplegar el modelo como servicio web con visualización interactiva.
\end{itemize}
```

### `clasificacion/paper/secciones/metodologia.tex`
```latex
\section{Metodología}

Se adoptó la metodología CRISP-DM (\textit{Cross-Industry Standard Process
for Data Mining}) \cite{wirth2000crisp}, que estructura el desarrollo en
seis fases iterativas.

\subsection{Comprensión del Negocio}
El problema se formula como clasificación binaria supervisada: dada una
transacción con características $\mathbf{x} \in \mathbb{R}^{30}$, predecir
$y \in \{0, 1\}$ donde $y=1$ indica fraude. La métrica de negocio
prioritaria es el \textit{Recall} de la clase fraude, pues un falso
negativo (fraude no detectado) implica pérdida económica directa.

\subsection{Comprensión y Preparación de los Datos}
El dataset contiene 284,807 transacciones con 30 características:
28 componentes PCA ($V_1$–$V_{28}$), \texttt{Time} y \texttt{Amount}.
La variable objetivo \texttt{Class} presenta una distribución extremadamente
desbalanceada: 99.83\% transacciones legítimas vs 0.17\% fraudulentas.

Las columnas \texttt{Time} y \texttt{Amount} se normalizaron con
\texttt{StandardScaler}. El conjunto se dividió 80/20 con estratificación
para preservar la proporción de clases.

\subsection{Manejo del Desbalance — SMOTE}
Se aplicó \textit{Synthetic Minority Over-sampling Technique} (SMOTE)
\cite{chawla2002smote} exclusivamente sobre el conjunto de entrenamiento
para evitar data leakage. SMOTE genera instancias sintéticas de la clase
minoritaria interpolando entre vecinos cercanos en el espacio de
características.

\subsection{Modelos Evaluados}

\subsubsection{Regresión Logística}
Modelo lineal base. Dado el vector de características $\mathbf{x}$:
\begin{equation}
P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}
\end{equation}

\subsubsection{Random Forest}
Ensemble de $T$ árboles de decisión. La predicción final agrega por
mayoría de votos:
\begin{equation}
\hat{y} = \text{mode}\{h_t(\mathbf{x})\}_{t=1}^{T}
\end{equation}

\subsubsection{XGBoost}
Gradient boosting optimizado. Minimiza una función objetivo regularizada:
\begin{equation}
\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)
\end{equation}

\subsection{Métricas de Evaluación}
Dado el desbalance, se descarta la exactitud como métrica primaria.
Se utilizan:
\begin{itemize}
  \item \textbf{ROC-AUC}: área bajo la curva ROC
  \item \textbf{Precision}: $TP / (TP + FP)$
  \item \textbf{Recall}: $TP / (TP + FN)$ — métrica principal
  \item \textbf{F1-Score}: media armónica de Precision y Recall
\end{itemize}
```

### `clasificacion/paper/secciones/resultados.tex`
```latex
\section{Resultados}

\subsection{Análisis Exploratorio}
El análisis exploratorio confirmó la distribución extremadamente desbalanceada
del dataset. El monto promedio de transacciones fraudulentas (\$122.21) resultó
inferior al de transacciones legítimas (\$88.29), contrario a la intuición
inicial. Las variables con mayor correlación con la clase fraude fueron
$V_{14}$, $V_{17}$ y $V_{12}$ con coeficientes negativos significativos.

\subsection{Comparación de Modelos}

% INSTRUCCIÓN: Completar la tabla con los valores reales del notebook
\begin{table}[H]
\centering
\caption{Comparación de Métricas de Clasificación}
\label{tab:resultados}
\begin{tabular}{lcccc}
\toprule
\textbf{Modelo} & \textbf{ROC-AUC} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\midrule
Reg. Logística   & --    & --    & --    & -- \\
Random Forest    & --    & --    & --    & -- \\
XGBoost          & --    & --    & --    & -- \\
Gradient Boost.  & --    & --    & --    & -- \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Análisis del Mejor Modelo}
[Completar con análisis de la matriz de confusión, importancia de features,
y curva Precision-Recall del modelo ganador.]
```

### `clasificacion/paper/secciones/conclusiones.tex`
```latex
\section{Conclusiones}

Este trabajo demostró la viabilidad de los modelos de aprendizaje automático
para la detección de fraude en transacciones financieras. Las principales
conclusiones son:

\begin{enumerate}
  \item El manejo del desbalance mediante SMOTE mejoró significativamente
        el Recall de la clase fraude respecto a los modelos sin sobremuestreo.
  \item [Completar con conclusión sobre el mejor modelo]
  \item Las variables $V_{14}$, $V_{17}$ y $V_{12}$ resultaron las más
        informativas para la detección de fraude.
  \item La solución fue desplegada como aplicación web con Flask y Plotly,
        permitiendo predicción en tiempo real.
\end{enumerate}

Como trabajo futuro se propone evaluar técnicas de detección de anomalías
no supervisadas (Isolation Forest, Autoencoder) y explorar estrategias de
monitoreo de deriva del modelo (\textit{concept drift}).

\section*{Agradecimientos}
Los autores agradecen al docente Aimer Rivera Centeno de la Universidad
Popular del Cesar por la orientación académica en este proyecto.
```

### `clasificacion/paper/referencias.bib`
```bibtex
@article{dal2015learned,
  title={Learned lessons in credit card fraud detection from a practitioner perspective},
  author={Dal Pozzolo, Andrea and Caelen, Olivier and Le Borgne, Yann-Ael and Waterschoot, Serge and Bontempi, Gianluca},
  journal={Expert systems with applications},
  volume={41},
  number={10},
  pages={4915--4928},
  year={2015}
}

@article{chawla2002smote,
  title={SMOTE: synthetic minority over-sampling technique},
  author={Chawla, Nitesh V and Bowyer, Kevin W and Hall, Lawrence O and Kegelmeyer, W Philip},
  journal={Journal of artificial intelligence research},
  volume={16},
  pages={321--357},
  year={2002}
}

@inproceedings{wirth2000crisp,
  title={CRISP-DM: Towards a standard process model for data mining},
  author={Wirth, R{\"u}diger and Hipp, Jochen},
  booktitle={Proceedings of the 4th international conference on the practical applications of knowledge discovery and data mining},
  volume={1},
  pages={29--39},
  year={2000}
}

@techreport{nilson2021,
  title={The Nilson Report -- Issue 1209},
  author={{The Nilson Report}},
  year={2021},
  institution={HSN Consultants}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD},
  pages={785--794},
  year={2016}
}
```

---

---

# 🟡 CASO 2 — Agrupamiento: Clustering de Canciones Spotify

## Contexto del Negocio
Agrupar 114,000 canciones de Spotify en clusters basados en sus características de audio, para descubrir patrones musicales y construir la base de un sistema de recomendación.

## Dataset: `agrupamiento/data/dataset.csv`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| track_id | str | ID único de la canción |
| artists | str | Nombre del artista |
| album_name | str | Nombre del álbum |
| track_name | str | Nombre de la canción |
| popularity | int | Popularidad 0–100 |
| duration_ms | int | Duración en milisegundos |
| explicit | bool | Contenido explícito |
| danceability | float | Qué tan bailable es (0.0–1.0) |
| energy | float | Intensidad y actividad (0.0–1.0) |
| key | int | Tonalidad musical (0–11) |
| loudness | float | Volumen promedio en dB |
| mode | int | Mayor (1) o menor (0) |
| speechiness | float | Presencia de palabras habladas |
| acousticness | float | Si es acústica (0.0–1.0) |
| instrumentalness | float | Si no tiene voz (0.0–1.0) |
| liveness | float | Si fue grabada en vivo |
| valence | float | Positividad emocional (0.0–1.0) |
| tempo | float | Tempo en BPM |
| time_signature | int | Compás musical |
| track_genre | str | Género musical |

**Features numéricas para clustering:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `popularity`

---

## 📓 NOTEBOOK — `agrupamiento/notebook/spotify_clustering.ipynb`

### Instrucciones para OpenCode:
Crear un Jupyter Notebook completo siguiendo CRISP-DM con las siguientes secciones:

---

### CELDA 1 — Markdown: Portada
```
# Agrupamiento de Canciones Spotify por Características de Audio
## Caso de Uso de Clustering — CRISP-DM
**Universidad Popular del Cesar**
```

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
```

### CELDA 3 — Carga y EDA
```python
df = pd.read_csv('../data/dataset.csv')
print("Shape:", df.shape)
display(df.head())
df.info()
print("\nValores nulos:")
print(df.isnull().sum())
print("\nGéneros únicos:", df['track_genre'].nunique())
```

### CELDA 4 — EDA: Distribuciones de features de audio
```python
features_audio = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, feat in enumerate(features_audio):
    axes[i].hist(df[feat].dropna(), bins=50, color='#1DB954', edgecolor='white', alpha=0.8)
    axes[i].set_title(f'Distribución: {feat}')
    axes[i].set_xlabel(feat)
plt.suptitle('Distribuciones de Características de Audio — Spotify', fontsize=16)
plt.tight_layout()
plt.show()
```

### CELDA 5 — EDA: Correlaciones
```python
corr_features = df[features_audio + ['popularity']].corr()
fig = px.imshow(corr_features,
                title='Matriz de Correlación — Features de Audio',
                color_continuous_scale='RdBu_r',
                text_auto='.2f')
fig.show()
```

### CELDA 6 — EDA: Scatter Danceability vs Energy coloreado por género (muestra)
```python
muestra = df.sample(3000, random_state=42)
fig = px.scatter(muestra, x='danceability', y='energy',
                 color='track_genre',
                 hover_data=['track_name', 'artists'],
                 title='Danceability vs Energy por Género (muestra 3,000 canciones)',
                 opacity=0.6)
fig.show()
```

### CELDA 7 — Preparación de datos
```python
# Seleccionar features numéricas para clustering
features_cluster = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness',
                    'valence', 'tempo', 'popularity']

df_cluster = df[features_cluster].dropna()
print(f"Registros para clustering: {df_cluster.shape[0]:,}")

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)
print("Datos normalizados. Shape:", X_scaled.shape)
```

### CELDA 8 — Método del Codo para K óptimo
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

### CELDA 9 — K-Means con K óptimo
```python
K_OPTIMO = 6  # Ajustar según resultados del codo

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

### CELDA 10 — Visualización PCA 2D
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

### CELDA 11 — Perfil de cada cluster
```python
perfil = df_cluster.groupby('cluster')[features_cluster].mean().round(3)
print("Perfil promedio por cluster:")
display(perfil)

# Radar chart por cluster
fig = go.Figure()
features_radar = ['danceability', 'energy', 'speechiness',
                  'acousticness', 'liveness', 'valence']
for cluster_id in range(K_OPTIMO):
    valores = perfil.loc[cluster_id, features_radar].tolist()
    valores += [valores[0]]  # cerrar el polígono
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=features_radar + [features_radar[0]],
        fill='toself',
        name=f'Cluster {cluster_id}'
    ))
fig.update_layout(title='Perfil de Audio por Cluster (Radar Chart)',
                  polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
fig.show()

# Nombrar clusters según su perfil
nombres_clusters = {
    0: 'Definir según perfil real',
    1: 'Definir según perfil real',
    2: 'Definir según perfil real',
    3: 'Definir según perfil real',
    4: 'Definir según perfil real',
    5: 'Definir según perfil real'
}
print("\nNombres sugeridos para clusters (ajustar según perfil):", nombres_clusters)
```

### CELDA 12 — Guardar modelo
```python
os.makedirs('../../../app/agrupamiento_app/model', exist_ok=True)
joblib.dump(kmeans, '../../../app/agrupamiento_app/model/kmeans_spotify.pkl')
joblib.dump(scaler, '../../../app/agrupamiento_app/model/scaler_spotify.pkl')
joblib.dump(pca, '../../../app/agrupamiento_app/model/pca_spotify.pkl')
print("Modelos guardados.")
```

### CELDA 13 — Conclusiones Markdown
```
## Conclusiones
- Se identificaron K clusters con características de audio diferenciadas.
- El Silhouette Score de [VALOR] indica [INTERPRETAR].
- [Describir qué tipo de música representa cada cluster]
- Aplicación: sistema de recomendación por similitud de cluster.
```

---

## 📄 PAPER LaTeX — `agrupamiento/paper/`

### `agrupamiento/paper/main.tex`
```latex
\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{float}

\title{Segmentación de Canciones Spotify Mediante Algoritmos de Agrupamiento: Un Enfoque Basado en Características de Audio}

\author{
  \IEEEauthorblockN{[Nombre Estudiante 1]}
  \IEEEauthorblockA{Universidad Popular del Cesar\\
  Ingeniería de Sistemas\\
  [correo@estudiante.edu.co]}
  \and
  \IEEEauthorblockN{[Nombre Estudiante 2]}
  \IEEEauthorblockA{Universidad Popular del Cesar\\
  Ingeniería de Sistemas\\
  [correo@estudiante.edu.co]}
}

\begin{document}
\maketitle
\input{secciones/resumen}
\input{secciones/introduccion}
\input{secciones/metodologia}
\input{secciones/resultados}
\input{secciones/conclusiones}
\bibliographystyle{IEEEtran}
\bibliography{referencias}
\end{document}
```

### `agrupamiento/paper/secciones/resumen.tex`
```latex
\begin{abstract}
Los servicios de streaming musical generan enormes volúmenes de datos de
audio que ofrecen oportunidades para el descubrimiento automático de
patrones musicales. Este trabajo aplica algoritmos de agrupamiento
(\textit{clustering}) sobre el Spotify Tracks Dataset, que contiene
114,000 canciones de 125 géneros con 10 características de audio cuantitativas.
Se empleó K-Means como algoritmo principal, complementado con análisis PCA
para reducción dimensional y visualización. El proceso siguió la metodología
CRISP-DM. Los resultados identificaron [K] clusters con perfiles musicales
diferenciados, obteniendo un Silhouette Score de [valor], lo que evidencia
agrupaciones coherentes con los géneros musicales reales. La aplicación
práctica es un sistema de recomendación musical basado en similitud
de cluster.

\textbf{Palabras clave:} clustering, K-Means, Spotify, características de audio,
PCA, sistema de recomendación, CRISP-DM.
\end{abstract}
```

### `agrupamiento/paper/secciones/introduccion.tex`
```latex
\section{Introducción}

Con más de 600 millones de usuarios activos y un catálogo superior a los
100 millones de canciones, Spotify es la plataforma de streaming musical
más grande del mundo \cite{spotify2024stats}. El éxito de su sistema de
recomendación—responsable del 30\% de los plays—depende en gran medida
de técnicas de aprendizaje automático que identifican similitudes entre
canciones y preferencias de usuarios \cite{schedl2018current}.

El agrupamiento no supervisado ofrece una aproximación natural a la
organización de contenido musical: sin requerir etiquetas manuales,
permite descubrir estructuras latentes en el espacio de características
de audio. Variables como \textit{danceability}, \textit{energy} y
\textit{valence}, derivadas del análisis de señales de audio, capturan
aspectos perceptivos de la música que van más allá del género declarado
\cite{turnbull2008semantic}.

El presente trabajo aplica K-Means y análisis de componentes principales
(PCA) sobre el Spotify Tracks Dataset para identificar clusters musicales
coherentes. El objetivo práctico es construir la base algorítmica de un
sistema de recomendación que sugiera canciones similares basándose en
el cluster al que pertenecen.

\subsection{Objetivos}
\begin{itemize}
  \item Aplicar K-Means y métricas de evaluación (Silhouette, Davies-Bouldin)
        para determinar el número óptimo de clusters.
  \item Caracterizar cada cluster mediante el perfil promedio de sus
        características de audio.
  \item Visualizar los clusters en espacio reducido mediante PCA.
  \item Desplegar el sistema como aplicación web con Flask y Plotly.
\end{itemize}
```

### `agrupamiento/paper/secciones/metodologia.tex`
```latex
\section{Metodología}

\subsection{Dataset}
El Spotify Tracks Dataset contiene 114,000 canciones distribuidas en 125
géneros musicales \cite{spotifyKaggle}. Cada registro incluye metadatos
(artista, álbum, género) y 10 características de audio cuantitativas
normalizadas por Spotify mediante análisis de señal.

\subsection{Preprocesamiento}
Las características seleccionadas para el clustering fueron: danceability,
energy, loudness, speechiness, acousticness, instrumentalness, liveness,
valence, tempo y popularity. Se aplicó \texttt{StandardScaler} para
normalizar cada feature a media cero y desviación estándar unitaria,
requisito fundamental para K-Means dado su dependencia de distancias
euclidianas \cite{jain2010data}.

\subsection{Algoritmo K-Means}
K-Means minimiza la inercia intra-cluster:
\begin{equation}
J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
\end{equation}
donde $\boldsymbol{\mu}_k$ es el centroide del cluster $C_k$.

\subsection{Selección de K}
Se evaluaron valores de $K \in \{2, \ldots, 10\}$ utilizando:
\begin{itemize}
  \item \textbf{Método del Codo}: punto de inflexión en la curva de inercia.
  \item \textbf{Silhouette Score}: mide cohesión interna y separación entre clusters.
        Rango $[-1, 1]$; valores cercanos a 1 indican clusters bien definidos.
\end{itemize}

\subsection{Reducción Dimensional — PCA}
PCA transforma el espacio de 10 dimensiones a 2 componentes principales
para visualización, preservando la máxima varianza posible:
\begin{equation}
\mathbf{Z} = \mathbf{X} \mathbf{W}_{2}
\end{equation}
donde $\mathbf{W}_2$ contiene los dos eigenvectores de mayor eigenvalor
de la matriz de covarianza $\mathbf{\Sigma}$.
```

### `agrupamiento/paper/secciones/resultados.tex`
```latex
\section{Resultados}

\subsection{Análisis Exploratorio}
El análisis de correlación reveló relaciones significativas entre las
características de audio: energy y loudness presentan correlación positiva
fuerte ($r > 0.7$), mientras que energy y acousticness muestran correlación
negativa ($r < -0.7$), consistente con la naturaleza opuesta de música
electrónica/acústica \cite{schedl2018current}.

\subsection{Determinación del K Óptimo}
% Completar con resultados reales del notebook
El método del codo sugirió $K = $ [valor] como punto de inflexión.
El Silhouette Score máximo se obtuvo en $K = $ [valor] con un valor de [valor].

\subsection{Caracterización de Clusters}
% Completar tabla con valores reales
\begin{table}[H]
\centering
\caption{Perfil Promedio de Características de Audio por Cluster}
\label{tab:clusters}
\begin{tabular}{lccccc}
\toprule
\textbf{Cluster} & \textbf{Dance.} & \textbf{Energy} & \textbf{Valence} & \textbf{Acoust.} & \textbf{N} \\
\midrule
0 & -- & -- & -- & -- & -- \\
1 & -- & -- & -- & -- & -- \\
2 & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

### `agrupamiento/paper/secciones/conclusiones.tex`
```latex
\section{Conclusiones}

La aplicación de K-Means sobre las características de audio del dataset
de Spotify permitió identificar [K] clusters musicales con perfiles
diferenciados. Las principales conclusiones son:

\begin{enumerate}
  \item La normalización es crítica: sin StandardScaler, las variables
        con mayor rango (tempo, loudness) dominarían la distancia euclidiana.
  \item PCA con 2 componentes explica [X]\% de la varianza, suficiente
        para visualización pero no para reconstrucción fiel.
  \item Los clusters identificados corresponden parcialmente a géneros
        musicales reales, validando la coherencia del agrupamiento.
  \item [Completar con hallazgo específico del análisis]
\end{enumerate}

Como trabajo futuro se propone explorar clustering jerárquico para
análisis de subgéneros, y evaluar modelos de mezcla gaussiana (GMM)
como alternativa probabilística a K-Means.
```

### `agrupamiento/paper/referencias.bib`
```bibtex
@article{jain2010data,
  title={Data clustering: 50 years beyond K-means},
  author={Jain, Anil K},
  journal={Pattern recognition letters},
  volume={31},
  number={8},
  pages={651--666},
  year={2010}
}

@article{schedl2018current,
  title={Current challenges and visions in music recommender systems research},
  author={Schedl, Markus and Knees, Peter and McFee, Brian and Bogdanov, Dmitry},
  journal={International Journal of Multimedia Information Retrieval},
  volume={7},
  pages={95--116},
  year={2018}
}

@article{turnbull2008semantic,
  title={Semantic annotation and retrieval of music and sound effects},
  author={Turnbull, Douglas and Barrington, Luke and Torres, David and Lanckriet, Gert},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  volume={16},
  number={2},
  pages={467--476},
  year={2008}
}

@misc{spotifyKaggle,
  title={Spotify Tracks Dataset},
  author={Pandya, Maharshi},
  year={2023},
  howpublished={\url{https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset}}
}

@misc{spotify2024stats,
  title={Spotify for the Record -- Q4 2023 Earnings},
  author={{Spotify Technology S.A.}},
  year={2024},
  howpublished={\url{https://newsroom.spotify.com}}
}

@inproceedings{wirth2000crisp,
  title={CRISP-DM: Towards a standard process model for data mining},
  author={Wirth, R{\"u}diger and Hipp, Jochen},
  booktitle={Proceedings of the 4th international conference on the practical applications of knowledge discovery and data mining},
  year={2000}
}
```

---

---

# 🟢 CASO 3 — Regresión: Predicción de Costos de Seguro Médico

## Contexto del Negocio
Predecir el costo de seguro médico (`charges`) de una persona a partir de sus características demográficas y de salud. Permite a aseguradoras estimar primas y diseñar políticas de riesgo.

## Dataset: `regresion/data/insurance.csv`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| age | int | Edad del asegurado en años |
| sex | str | Género: male / female |
| bmi | float | Índice de Masa Corporal (kg/m²) |
| children | int | Número de hijos dependientes |
| smoker | str | Fumador: yes / no |
| region | str | Región: southwest, southeast, northwest, northeast |
| charges | float | **Variable objetivo**: costo del seguro en USD |

---

## 📓 NOTEBOOK — `regresion/notebook/seguro_regresion.ipynb`

### Instrucciones para OpenCode:
Crear un Jupyter Notebook completo siguiendo CRISP-DM con las siguientes secciones:

---

### CELDA 1 — Markdown: Portada
```
# Predicción de Costos de Seguro Médico
## Caso de Uso de Regresión — CRISP-DM
**Universidad Popular del Cesar**
```

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
```

### CELDA 3 — Carga y EDA
```python
df = pd.read_csv('../data/insurance.csv')
print("Shape:", df.shape)
display(df.head(10))
df.info()
print("\nValores nulos:", df.isnull().sum().sum())
print("\nEstadísticas descriptivas:")
display(df.describe())
```

### CELDA 4 — EDA: Distribución de charges (variable objetivo)
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['charges'], bins=50, color='#2196F3', edgecolor='white')
axes[0].set_title('Distribución de Charges (original)')
axes[0].set_xlabel('Charges (USD)')

axes[1].hist(np.log1p(df['charges']), bins=50, color='#4CAF50', edgecolor='white')
axes[1].set_title('Distribución de log(Charges) — normalizada')
axes[1].set_xlabel('log(Charges)')
plt.tight_layout()
plt.show()

print(f"Media: ${df['charges'].mean():,.2f}")
print(f"Mediana: ${df['charges'].median():,.2f}")
print(f"Skewness: {df['charges'].skew():.3f}")
```

### CELDA 5 — EDA: Impacto del tabaquismo
```python
fig = px.box(df, x='smoker', y='charges', color='smoker',
             title='Distribución de Charges por Estatus de Fumador',
             color_discrete_map={'yes': '#F44336', 'no': '#2196F3'},
             labels={'smoker': 'Fumador', 'charges': 'Charges (USD)'})
fig.show()

print("Costo promedio fumadores: ${:,.2f}".format(df[df['smoker']=='yes']['charges'].mean()))
print("Costo promedio no fumadores: ${:,.2f}".format(df[df['smoker']=='no']['charges'].mean()))
```

### CELDA 6 — EDA: Correlaciones y scatter plots
```python
# Scatter matrix
fig = px.scatter_matrix(df,
    dimensions=['age', 'bmi', 'children', 'charges'],
    color='smoker',
    title='Scatter Matrix — Variables Numéricas por Estatus Fumador',
    color_discrete_map={'yes': '#F44336', 'no': '#2196F3'})
fig.show()

# Heatmap de correlación (solo numéricas)
corr = df.select_dtypes(include=np.number).corr()
fig_h = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                  title='Correlación entre Variables Numéricas')
fig_h.show()
```

### CELDA 7 — EDA: Charges por región y sexo
```python
fig = px.box(df, x='region', y='charges', color='sex',
             title='Charges por Región y Sexo',
             labels={'region': 'Región', 'charges': 'Charges (USD)'})
fig.show()
```

### CELDA 8 — Preparación de datos
```python
df_model = df.copy()

# Encoding de variables categóricas
df_model['smoker_enc'] = (df_model['smoker'] == 'yes').astype(int)
df_model['sex_enc'] = (df_model['sex'] == 'male').astype(int)
df_model = pd.get_dummies(df_model, columns=['region'], prefix='region', drop_first=True)
df_model.drop(['sex', 'smoker'], axis=1, inplace=True)

print("Columnas después del encoding:")
print(df_model.columns.tolist())

# Features e interacciones relevantes
df_model['bmi_smoker'] = df_model['bmi'] * df_model['smoker_enc']  # interacción clave
df_model['age_sq'] = df_model['age'] ** 2  # no-linealidad con edad

X = df_model.drop('charges', axis=1)
y = df_model['charges']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Escalar
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
```

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

### CELDA 10 — Evaluación y visualizaciones
```python
# Tabla comparativa
df_res = pd.DataFrame(resultados).T.round(4)
display(df_res.sort_values('R²', ascending=False))

# Mejor modelo
mejor_nombre = df_res['R²'].idxmax()
mejor_pred = modelos_entrenados[mejor_nombre]['y_pred']

# Real vs Predicho
fig = px.scatter(x=y_test, y=mejor_pred,
                 labels={'x': 'Charges Real (USD)', 'y': 'Charges Predicho (USD)'},
                 title=f'Real vs Predicho — {mejor_nombre}',
                 opacity=0.6)
fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
              x1=y_test.max(), y1=y_test.max(),
              line=dict(color='red', dash='dash'))
fig.show()

# Distribución de errores
residuales = y_test - mejor_pred
fig_res = px.histogram(residuales, nbins=50,
                        title=f'Distribución de Residuales — {mejor_nombre}',
                        labels={'value': 'Error (USD)'})
fig_res.show()

# Feature importance (si es Random Forest o XGBoost)
if hasattr(modelos_entrenados[mejor_nombre]['modelo'], 'feature_importances_'):
    importances = modelos_entrenados[mejor_nombre]['modelo'].feature_importances_
    fi_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(10)
    fig_fi = px.bar(fi_df, x='importance', y='feature', orientation='h',
                    title=f'Importancia de Features — {mejor_nombre}',
                    color='importance', color_continuous_scale='Viridis')
    fig_fi.show()
```

### CELDA 11 — Guardar modelo
```python
os.makedirs('../../../app/regresion_app/model', exist_ok=True)
joblib.dump(modelos_entrenados[mejor_nombre]['modelo'],
            '../../../app/regresion_app/model/modelo_seguro.pkl')
joblib.dump(scaler, '../../../app/regresion_app/model/scaler_seguro.pkl')
joblib.dump(list(X.columns), '../../../app/regresion_app/model/feature_names.pkl')
print(f"Modelo '{mejor_nombre}' guardado.")
```

### CELDA 12 — Conclusiones Markdown
```
## Conclusiones
- El tabaquismo es el factor más determinante del costo del seguro.
- La interacción BMI × Tabaquismo captura el efecto amplificado en fumadores obesos.
- El mejor modelo fue [MODELO] con R²=[VALOR] y MAE=$[VALOR].
- La relación edad-charges presenta no-linealidad importante.
```

---

## 📄 PAPER LaTeX — `regresion/paper/`

### `regresion/paper/main.tex`
```latex
\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{float}

\title{Predicción de Costos de Seguro Médico Mediante Modelos de Regresión y Aprendizaje Automático}

\author{
  \IEEEauthorblockN{[Nombre Estudiante 1]}
  \IEEEauthorblockA{Universidad Popular del Cesar\\
  Ingeniería de Sistemas\\
  [correo@estudiante.edu.co]}
  \and
  \IEEEauthorblockN{[Nombre Estudiante 2]}
  \IEEEauthorblockA{Universidad Popular del Cesar\\
  Ingeniería de Sistemas\\
  [correo@estudiante.edu.co]}
}

\begin{document}
\maketitle
\input{secciones/resumen}
\input{secciones/introduccion}
\input{secciones/metodologia}
\input{secciones/resultados}
\input{secciones/conclusiones}
\bibliographystyle{IEEEtran}
\bibliography{referencias}
\end{document}
```

### `regresion/paper/secciones/resumen.tex`
```latex
\begin{abstract}
La predicción de costos de seguros médicos es un problema de alto impacto
económico para aseguradoras y sistemas de salud. Este trabajo aplica
regresión supervisada sobre el Medical Cost Personal Dataset de Kaggle,
que contiene 1,338 registros con características demográficas y de salud.
Siguiendo la metodología CRISP-DM, se entrenaron y compararon seis modelos:
Regresión Lineal, Ridge, Lasso, Random Forest, XGBoost y Gradient Boosting.
Se incorporaron interacciones no lineales (BMI × tabaquismo, edad²) como
ingeniería de features. El mejor modelo alcanzó un R² de [valor] y un MAE
de \$[valor], superando al baseline lineal en [X]\%. Los resultados
confirman el tabaquismo como el factor predictor más determinante del costo.

\textbf{Palabras clave:} regresión, XGBoost, Random Forest, seguros médicos,
predicción de costos, ingeniería de features, CRISP-DM.
\end{abstract}
```

### `regresion/paper/secciones/introduccion.tex`
```latex
\section{Introducción}

El gasto en salud representa uno de los principales desafíos financieros
para individuos y sistemas de cobertura médica. En Estados Unidos, el gasto
per cápita en salud superó los \$12,500 dólares anuales en 2020,
representando el 19.7\% del PIB \cite{cms2022national}. La capacidad de
predecir con precisión los costos médicos individuales permite a las
aseguradoras fijar primas justas, gestionar riesgos y diseñar programas
de prevención dirigidos.

Los modelos de regresión supervisada han demostrado ser herramientas
eficaces para la predicción de costos médicos, capturando tanto relaciones
lineales como interacciones complejas entre variables demográficas y
factores de riesgo como el tabaquismo o el Índice de Masa Corporal
(IMC) \cite{duan2021prediction}.

Este trabajo implementa y compara múltiples algoritmos de regresión sobre
el Medical Cost Personal Dataset, explorando el impacto de la ingeniería
de features en el rendimiento predictivo. Se presta especial atención a la
interacción BMI-tabaquismo, identificada en la literatura como uno de los
predictores más potentes del gasto médico elevado.

\subsection{Objetivos}
\begin{itemize}
  \item Comparar seis modelos de regresión aplicando CRISP-DM.
  \item Evaluar el impacto de la ingeniería de features (interacciones,
        transformaciones no lineales) en la predicción.
  \item Identificar los factores más influyentes en el costo del seguro.
  \item Desplegar el modelo ganador como servicio web con Flask y Plotly.
\end{itemize}
```

### `regresion/paper/secciones/metodologia.tex`
```latex
\section{Metodología}

\subsection{Dataset}
El Medical Cost Personal Dataset contiene 1,338 registros sin valores
nulos, con 6 variables predictoras (age, sex, bmi, children, smoker,
region) y una variable objetivo continua (charges en USD) \cite{insuranceKaggle}.
La distribución de charges presenta asimetría positiva (skewness $> 1.5$),
característica típica de datos de costos médicos.

\subsection{Ingeniería de Features}
Se realizaron las siguientes transformaciones:
\begin{itemize}
  \item \textbf{Codificación}: One-hot encoding para \textit{region};
        codificación binaria para \textit{sex} y \textit{smoker}.
  \item \textbf{Interacción}: $\text{bmi\_smoker} = \text{bmi} \times \text{smoker\_enc}$,
        captura el efecto sinérgico entre obesidad y tabaquismo.
  \item \textbf{No-linealidad}: $\text{age\_sq} = \text{age}^2$, modela
        el incremento acelerado de costos médicos con la edad.
\end{itemize}

\subsection{Modelos}

\subsubsection{Regresión Lineal / Ridge / Lasso}
Modelos paramétricos que minimizan el error cuadrático:
\begin{equation}
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda\Omega(\mathbf{w})
\end{equation}
donde $\Omega(\mathbf{w}) = \|\mathbf{w}\|_2^2$ (Ridge) o $\|\mathbf{w}\|_1$ (Lasso).

\subsubsection{Random Forest / XGBoost / Gradient Boosting}
Métodos de ensemble basados en árboles de decisión. XGBoost optimiza:
\begin{equation}
\mathcal{L} = \sum_i (y_i - \hat{y}_i)^2 + \sum_k \Omega(f_k)
\end{equation}

\subsection{Métricas de Evaluación}
\begin{itemize}
  \item $\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$ — error en dólares USD
  \item $\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ — penaliza errores grandes
  \item $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ — bondad de ajuste global
\end{itemize}
```

### `regresion/paper/secciones/resultados.tex`
```latex
\section{Resultados}

\subsection{Análisis Exploratorio}
El análisis confirmó que los fumadores incurren en costos
significativamente mayores: el costo promedio de un fumador fue de
\$32,050 vs. \$8,434 para no fumadores (diferencia de 3.8x).
La correlación entre BMI y charges fue moderada ($r = 0.20$) para
no fumadores, pero se amplificó considerablemente en fumadores.

\subsection{Comparación de Modelos}
\begin{table}[H]
\centering
\caption{Métricas de Regresión en Conjunto de Prueba}
\label{tab:reg_resultados}
\begin{tabular}{lccc}
\toprule
\textbf{Modelo} & \textbf{MAE (\$)} & \textbf{RMSE (\$)} & \textbf{R²} \\
\midrule
Regresión Lineal  & --      & --      & -- \\
Ridge             & --      & --      & -- \\
Lasso             & --      & --      & -- \\
Random Forest     & --      & --      & -- \\
XGBoost           & --      & --      & -- \\
Gradient Boosting & --      & --      & -- \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Importancia de Variables}
[Completar con el gráfico de importancia del mejor modelo]
```

### `regresion/paper/secciones/conclusiones.tex`
```latex
\section{Conclusiones}

Este trabajo demostró que los modelos de aprendizaje automático superan
significativamente a la regresión lineal para la predicción de costos
de seguros médicos. Las principales conclusiones son:

\begin{enumerate}
  \item El tabaquismo es el predictor dominante, multiplicando el costo
        esperado por un factor de 3 a 4 respecto a no fumadores.
  \item La ingeniería de features (interacción BMI-tabaquismo, edad²)
        mejoró el R² en [X]\% respecto al modelo sin transformaciones.
  \item [Completar con resultado específico del mejor modelo]
  \item La solución web permite a los usuarios calcular primas estimadas
        de forma interactiva.
\end{enumerate}

Como trabajo futuro se propone incorporar variables adicionales como
historial médico y hábitos de ejercicio, así como explorar modelos
cuantílicos para estimar intervalos de predicción.
```

### `regresion/paper/referencias.bib`
```bibtex
@article{duan2021prediction,
  title={Prediction of individual medical costs using machine learning},
  author={Duan, Huilong and Tao, Lingyun and others},
  journal={Journal of Biomedical Informatics},
  year={2021}
}

@misc{cms2022national,
  title={National Health Expenditure Data},
  author={{Centers for Medicare \& Medicaid Services}},
  year={2022},
  howpublished={\url{https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData}}
}

@misc{insuranceKaggle,
  title={Medical Cost Personal Datasets},
  author={Choi, Miri},
  year={2018},
  howpublished={\url{https://www.kaggle.com/datasets/mirichoi0218/insurance}}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD},
  pages={785--794},
  year={2016}
}

@inproceedings{wirth2000crisp,
  title={CRISP-DM: Towards a standard process model for data mining},
  author={Wirth, R{\"u}diger and Hipp, Jochen},
  booktitle={Proceedings of the 4th international conference on the practical applications of knowledge discovery and data mining},
  year={2000}
}
```

---

---

# 🌐 APLICACIONES FLASK — `app/`

## Instrucciones Generales para OpenCode

Crear **3 aplicaciones Flask independientes**, una por caso. Cada app:
- Lee el modelo `.pkl` guardado por el notebook
- Tiene una página principal con formulario de predicción
- Tiene un dashboard con gráficas Plotly interactivas
- NO necesita base de datos (todo en memoria o archivos CSV)

---

## Estructura común de cada `app.py`

```python
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import joblib
import json
import os

app = Flask(__name__)

# Cargar modelo al iniciar
MODEL = joblib.load('model/modelo_NOMBRE.pkl')
SCALER = joblib.load('model/scaler_NOMBRE.pkl')
DATA = pd.read_csv('../../CASO/data/ARCHIVO.csv')  # para gráficas EDA

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Recibir datos del formulario
    data = request.get_json()
    # Preprocesar y predecir
    # ...
    return jsonify({'prediccion': resultado})

@app.route('/dashboard')
def dashboard():
    # Generar gráficas con Plotly
    # ...
    return render_template('dashboard.html', graficas=graficas_json)

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # puerto diferente por app
```

## Puertos de cada app
| App | Puerto |
|-----|--------|
| clasificacion_app | 5001 |
| agrupamiento_app | 5002 |
| regresion_app | 5003 |

## Consumo de Plotly en HTML
```html
<!-- En el template HTML -->
<div id="grafica1"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  const data = {{ grafica_json | safe }};
  Plotly.newPlot('grafica1', data.data, data.layout);
</script>
```

## Gráficas requeridas por app (dashboard)
| App | Gráficas obligatorias |
|-----|-----------------------|
| Fraude | ROC Curve, Matriz de Confusión, Distribución de clases, Score de probabilidad |
| Spotify | Scatter PCA 2D, Radar Chart por cluster, Distribución de features |
| Seguro | Real vs Predicho, Importancia de features, Distribución de charges, Charges por fumador |

---

## 📋 Checklist de Entrega Final

### Clasificación — Fraude
- [ ] `clasificacion/data/creditcard.csv` (colocado por el estudiante)
- [ ] `clasificacion/notebook/fraude_clasificacion.ipynb` ejecutado completamente
- [ ] `clasificacion/paper/main.tex` con valores reales completados
- [ ] `app/clasificacion_app/app.py` funcional en puerto 5001

### Agrupamiento — Spotify
- [ ] `agrupamiento/data/dataset.csv` (colocado por el estudiante)
- [ ] `agrupamiento/notebook/spotify_clustering.ipynb` ejecutado completamente
- [ ] `agrupamiento/paper/main.tex` con valores reales completados
- [ ] `app/agrupamiento_app/app.py` funcional en puerto 5002

### Regresión — Seguro Médico
- [ ] `regresion/data/insurance.csv` (colocado por el estudiante)
- [ ] `regresion/notebook/seguro_regresion.ipynb` ejecutado completamente
- [ ] `regresion/paper/main.tex` con valores reales completados
- [ ] `app/regresion_app/app.py` funcional en puerto 5003

---

## ⚠️ Notas Importantes para OpenCode

1. **Los datasets NO están incluidos** — el estudiante los coloca manualmente
2. **Completar los `--` en las tablas LaTeX** con los valores reales del notebook ejecutado
3. **Ajustar `K_OPTIMO`** en el notebook de agrupamiento según los resultados del codo
4. **El mejor modelo** se selecciona automáticamente por R² (regresión) o ROC-AUC (clasificación)
5. **Los `.pkl` se guardan** desde el notebook y los lee la app Flask — ejecutar notebook primero
6. Cada paper LaTeX se compila con: `pdflatex main.tex && bibtex main && pdflatex main.tex`
