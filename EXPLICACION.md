# Proyecto Final — Ciencia de Datos

> **Universidad Popular del Cesar** · Ingeniería de Sistemas  
> **Docente:** Aimer J. Rivera Centeno · **Metodología:** CRISP-DM  
> **Tecnologías:** Python 3.12, Scikit-learn, XGBoost, Flask, Plotly, Jupyter

---

## 📋 ¿Qué es este proyecto?

Un pipeline completo de ciencia de datos que aborda **3 tipos distintos de problemas** de machine learning usando la metodología **CRISP-DM**: clasificación (fraude), agrupamiento (música) y regresión (seguros). Cada caso incluye notebook de análisis, paper académico en LaTeX y aplicación web interactiva.

---

## 🔴 Caso 1 — Clasificación: Detección de Fraude

### ¿Qué problema resuelve?

Detectar si una transacción de tarjeta de crédito es **fraudulenta o legítima** antes de que se procese. Es un problema de **clasificación binaria supervisada**.

### ¿Por qué es difícil?

Solo el **0.17%** de las transacciones son fraude. Esto significa que un modelo que diga "todo es legítimo" tendría 99.83% de exactitud... pero no detectaría **ningún** fraude. Por eso usamos **Recall como métrica principal** (qué porcentaje de fraudes reales logramos detectar).

### ¿Qué hicimos?

| Fase CRISP-DM | Acción |
|---|---|
| **1. Negocio** | Minimizar pérdidas. Métrica: Recall (falsos negativos = dinero perdido). |
| **2. Datos** | 284,807 transacciones, 30 variables (28 PCA + Time + Amount). 0.17% fraude. |
| **3. Preparación** | StandardScaler en Time/Amount. Split 80/20 estratificado. **SMOTE** para balancear (227K → 454K filas). |
| **4. Modelado** | 4 algoritmos: Regresión Logística, Random Forest, XGBoost, Gradient Boosting. |
| **5. Evaluación** | ROC-AUC, F1, Precision, Recall, Matriz de Confusión. |

### ¿Qué resultado obtuvimos?

| Modelo | ROC-AUC | F1-Score | Recall | Precision |
|---|---|---|---|---|
| **🏅 Random Forest** | **0.9841** | **0.8229** | **0.8061** | **0.8404** |
| Gradient Boosting | 0.9807 | 0.1888 | 0.8980 | 0.1055 |
| XGBoost | 0.9792 | 0.8018 | 0.8163 | 0.7878 |
| Regresión Logística | 0.9698 | 0.1094 | 0.0612 | 0.5217 |

### ¿Qué significa?

- **Random Forest gana** con ROC-AUC de 0.9841 (excelente poder discriminativo).
- **Detecta el 81% de los fraudes reales** (Recall 0.81). De cada 100 fraudes, atrapa 81.
- **Precision de 84%**: cuando dice "esto es fraude", acierta el 84% de las veces.
- **SMOTE fue crítico**: sin balanceo, los modelos colapsan hacia predecir siempre "legítimo".
- Las variables **V17, V14 y V12** son las más correlacionadas con fraude.

### ¿Cómo funciona la predicción?

Ingresás los valores V1–V28, Time y Amount → el modelo (Random Forest) predice si es FRAUDE o LEGÍTIMA, con su probabilidad.

---

## 🎵 Caso 2 — Agrupamiento: Spotify Clustering

### ¿Qué problema resuelve?

Agrupar **114,000 canciones** en clusters según sus características de audio, **sin etiquetas previas**. Es un problema de **aprendizaje no supervisado**.

### ¿Por qué clustering y no clasificación?

Porque no tenemos etiquetas de "tipo de canción". Queremos que el algoritmo **descubra solo** los patrones. K-Means agrupa canciones que "suenan parecido" basándose en 10 features de audio.

### ¿Qué hicimos?

| Fase CRISP-DM | Acción |
|---|---|
| **1. Negocio** | Base para un sistema de recomendación: "canciones similares a esta". |
| **2. Datos** | 114,000 canciones, 125 géneros, 10 features de audio. Sin nulos. |
| **3. Preparación** | StandardScaler (obligatorio: tempo y loudness tienen rangos muy distintos). |
| **4. Modelado** | K-Means con K=2 a 10. Método del codo + Silhouette Score para elegir K. |
| **5. Evaluación** | Silhouette Score, Davies-Bouldin, PCA 2D para visualizar. |

### ¿Qué resultado obtuvimos?

**K=6 clusters** (elegido por balance entre granularidad e interpretabilidad):

| Cluster | Canciones | Perfil |
|---|---|---|
| **0 — Energía y Baile** | 29,986 | Alta energy (0.82). Electrónica, pop bailable. |
| **1 — Acústico y Relajado** | 24,637 | Alta acousticness, baja energy. Folk, indie. |
| **2 — Instrumental** | 38,284 | Alta instrumentalness. Bandas sonoras, ambient, clásica. |
| **3 — Hablado** | 7,670 | Alta speechiness. Rap, hip-hop, podcasts. |
| **4 — En Vivo** | 12,159 | Alta liveness. Conciertos grabados. |
| **5 — Melódico Positivo** | 1,264 | Alta valence. Pop alegre, música positiva. |

**Métricas:** Silhouette Score: 0.1652 | Davies-Bouldin: 1.6242 | PCA varianza: 43.0%

### ¿Qué significa?

- El Silhouette Score de 0.1652 es **moderado** (típico en datos de audio de alta dimensionalidad).
- Los clusters **sí corresponden parcialmente a géneros reales**, validando que las features de Spotify capturan dimensiones musicales significativas.
- Energy y loudness tienen correlación **positiva fuerte** (r > 0.7). Energy y acousticness, **negativa fuerte** (r < −0.7). Esto es físicamente esperable.
- PCA con 2 componentes solo explica el 43% de la varianza: la visualización es orientativa, no perfecta.

### ¿Por qué K=6 y no K=2 (que tenía mejor Silhouette)?

K=2 solo separa "bailable" vs "no bailable" — demasiado grueso para 125 géneros. K=6 preserva la diversidad musical real sin fragmentar excesivamente.

---

## 🏥 Caso 3 — Regresión: Predicción de Seguro Médico

### ¿Qué problema resuelve?

Predecir el **costo anual del seguro médico** de una persona según sus características. Es un problema de **regresión supervisada** (predecir un valor numérico continuo).

### ¿Por qué es relevante?

Las aseguradoras necesitan estimar primas justas. Si subestiman, pierden dinero. Si sobreestiman, pierden clientes. Un modelo preciso permite fijar primas ajustadas al riesgo real.

### ¿Qué hicimos?

| Fase CRISP-DM | Acción |
|---|---|
| **1. Negocio** | Predecir costo anual (USD). Métrica: MAE (error en dólares, interpretable). |
| **2. Datos** | 1,338 asegurados, 6 variables + target (charges). Distribución asimétrica. |
| **3. Preparación** | One-hot encoding (región), binario (sex, smoker). **Ingeniería de features**: `bmi_smoker` (interacción) y `age_sq` (término cuadrático). Split 80/20. StandardScaler. |
| **4. Modelado** | 6 modelos: Reg. Lineal, Ridge, Lasso, Random Forest, XGBoost, Gradient Boosting. |
| **5. Evaluación** | MAE, RMSE, R². Residuales. Importancia de features. |

### ¿Qué resultado obtuvimos?

| Modelo | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| **🏅 Gradient Boosting** | **2,495.65** | **4,440.17** | **0.8730** |
| XGBoost | 2,583.13 | 4,518.97 | 0.8685 |
| Ridge | 2,763.69 | 4,529.65 | 0.8678 |
| Regresión Lineal | 2,755.47 | 4,537.67 | 0.8674 |
| Lasso | 2,761.84 | 4,544.74 | 0.8669 |
| Random Forest | 2,506.33 | 4,666.44 | 0.8597 |

**Top 5 features más importantes (Gradient Boosting):**

| # | Variable | Importancia |
|---|---|---|
| 1 | `bmi_smoker` | 80.67% |
| 2 | `age` | 7.52% |
| 3 | `bmi` | 4.47% |
| 4 | `age_sq` | 4.36% |
| 5 | `smoker_enc` | 1.33% |

### ¿Qué significa?

- **Gradient Boosting gana** con R² = 0.8730 (explica el 87.3% de la varianza del costo).
- **MAE de $2,495**: el modelo estima la prima con un error promedio de ~$2,500 al año.
- **`bmi_smoker` DOMINA con 80.67%**: la interacción entre obesidad y tabaquismo es el predictor más potente, confirmando el efecto sinérgico.
- **Fumadores pagan 3.8× más**: $32,050 vs $8,434 en promedio.
- La ingeniería de features (`bmi_smoker`, `age_sq`) fue clave: sin estos términos, los modelos lineales no capturan la sinergia BMI-tabaquismo.

### ¿Qué limitaciones tiene?

- Dataset pequeño (1,338 registros). Con más datos, modelos más complejos podrían generalizar mejor.
- `charges` captura solo el costo al momento de la póliza, no siniestros posteriores.
- No incluye historial médico ni hábitos de ejercicio.

---

## 🖥️ Aplicación Web Unificada

Una sola app Flask (puerto 5000) con sidebar navegable entre los 3 casos. Cada caso tiene:

- **Formulario de predicción** (sliders, inputs numéricos)
- **Dashboard interactivo** con gráficas Plotly (carga bajo demanda)
- **Sección de metodología y resultados** con los valores reales del análisis

### Cómo ejecutar

```bash
cd "corte3/Proyecto Final/app"
python app.py
# Abrir http://localhost:5000
```

### Estructura del proyecto

```
corte3/Proyecto Final/
├── app/                        ← App Flask unificada
│   ├── app.py                  ← Servidor (carga 3 modelos)
│   ├── templates/index.html    ← Frontend (sidebar + 3 pestañas)
│   └── static/style.css        ← Diseño Databricks
├── Agrupamiento/               ← Spotify clustering
│   ├── notebook/               ← Notebook Jupyter ejecutado
│   ├── Data/dataset.csv        ← Dataset (colocado manualmente)
│   └── paper/                  ← Paper LaTeX + figuras
├── Clasificacion/              ← Detección de fraude
│   ├── notebook/               ← Notebook Jupyter ejecutado
│   ├── Data/creditcard.csv     ← Dataset
│   └── paper/                  ← Paper LaTeX + figuras
├── Regresion/                  ← Predicción de seguro
│   ├── notebook/               ← Notebook Jupyter ejecutado
│   ├── Data/insurance.csv      ← Dataset
│   └── paper/                  ← Paper LaTeX + figuras
├── paper_clasificacion.tex     ← Paper unificado (corregido con resultados)
├── paper_agrupamiento.tex      ← Paper unificado (corregido con resultados)
├── paper_regresion.tex         ← Paper unificado (corregido con resultados)
├── EXPLICACION.md              ← Este archivo
└── proyecto_ciencia_datos.md   ← Guía original del proyecto
```

---

## 📚 Referencias

1. **CRISP-DM**: Wirth & Hipp (2000). *CRISP-DM: Towards a standard process model for data mining.*
2. **SMOTE**: Chawla et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* Journal of AI Research.
3. **XGBoost**: Chen & Guestrin (2016). *XGBoost: A scalable tree boosting system.* ACM SIGKDD.
4. **Fraude**: Dal Pozzolo et al. (2015). *Learned lessons in credit card fraud detection.* Expert Systems with Applications.
5. **Spotify**: Schedl et al. (2018). *Current challenges in music recommender systems.* IJMR.
6. **Seguros**: Duan et al. (2021). *Prediction of individual medical costs using machine learning.* J. Biomed. Informatics.
