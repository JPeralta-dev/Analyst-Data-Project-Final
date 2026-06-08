# 📚 GUÍA COMPLETA DE SUSTENTACIÓN — Proyecto Final Ciencia de Datos

> **Universidad Popular del Cesar** · Ingeniería de Sistemas
> **Docente:** Aimer J. Rivera Centeno · **Metodología:** CRISP-DM
> **Tecnologías:** Python 3.12, Scikit-learn, XGBoost, Flask, Plotly, Jupyter

---

# 📋 ÍNDICE

1. [Visión General del Proyecto](#1-visión-general-del-proyecto)
2. [Metodología CRISP-DM](#2-metodología-crisp-dm)
3. [CASO 1 — Clasificación: Detección de Fraude](#3-caso-1--clasificación-detección-de-fraude)
4. [CASO 2 — Agrupamiento: Spotify Clustering](#4-caso-2--agrupamiento-spotify-clustering)
5. [CASO 3 — Regresión: Predicción de Seguro Médico](#5-caso-3--regresión-predicción-de-seguro-médico)
6. [Aplicación Web y Dashboard](#6-aplicación-web-y-dashboard)
7. [Preguntas Frecuentes del Profesor](#7-preguntas-frecuentes-del-profesor)
8. [Glosario de Conceptos Clave](#8-glosario-de-conceptos-clave)

---

# 1. VISIÓN GENERAL DEL PROYECTO

## ¿Qué es este proyecto?

Es un **pipeline completo de ciencia de datos** que aborda **3 tipos distintos de problemas** de machine learning usando la metodología **CRISP-DM**:

| Caso | Tipo de Problema | Dataset | Objetivo |
|------|-----------------|---------|----------|
| **1** | Clasificación Binaria | Credit Card Fraud (284,807 filas) | Detectar fraude en transacciones |
| **2** | Agrupamiento (Clustering) | Spotify Tracks (114,000 canciones) | Agrupar canciones por similitud |
| **3** | Regresión | Medical Cost (1,338 registros) | Predecir costo de seguro médico |

## ¿Por qué 3 casos diferentes?

Porque la ciencia de datos tiene **3 paradigmas fundamentales**:
- **Supervisado con etiqueta discreta** → Clasificación (fraude sí/no)
- **No supervisado sin etiquetas** → Agrupamiento (descubrir patrones solos)
- **Supervisado con etiqueta continua** → Regresión (predecir un número)

Cada uno requiere enfoque, métricas y técnicas distintas.

---

# 2. METODOLOGÍA CRISP-DM

## ¿Qué es CRISP-DM?

**Cross-Industry Standard Process for Data Mining**. Es el estándar de la industria para proyectos de minería de datos. Tiene **6 fases iterativas**:

```
1. Comprensión del Negocio → ¿Qué problema queremos resolver?
2. Comprensión de los Datos → ¿Qué datos tenemos? ¿Cómo son?
3. Preparación de los Datos → Limpieza, transformación, feature engineering
4. Modelado → Entrenar algoritmos de ML
5. Evaluación → ¿El modelo cumple con los objetivos de negocio?
6. Despliegue → Poner el modelo en producción (nuestra app Flask)
```

### ¿Por qué CRISP-DM y no otra metodología?

- Es el **estándar más usado** en la industria (no es académico, es real)
- Es **iterativo**: podés volver a fases anteriores si encontrás problemas
- Es **agnóstico al dominio**: funciona para fraude, música, seguros, etc.
- Separa claramente el **entendimiento del negocio** del modelado técnico

---

# 3. CASO 1 — CLASIFICACIÓN: DETECCIÓN DE FRAUDE

## 3.1 Variable Objetivo

**Variable:** `Class`
- `0` = Transacción legítima
- `1` = Transacción fraudulenta

**¿Por qué esta es la variable objetivo?**
Porque el problema de negocio es: "¿Esta transacción es fraude o no?" Es una pregunta de SÍ/NO → clasificación binaria.

## 3.2 Dataset: Credit Card Fraud Detection

| Característica | Valor |
|---|---|
| **Filas** | 284,807 transacciones |
| **Columnas** | 31 (30 features + 1 target) |
| **Features** | V1–V28 (PCA anonimizado), Time, Amount |
| **Distribución** | 99.83% legítimas, **0.17% fraude** |
| **Valores nulos** | **NINGUNO** (0 nulos) |
| **Origen** | Tarjetas europeas, Kaggle |

### ¿Por qué las variables V1-V28 están anonimizadas (PCA)?

Por **privacidad**. El banco original no podía revelar las variables reales (monto, hora, ubicación, tipo de comercio, etc.) porque eso identificaría a los clientes. Aplicaron **PCA (Análisis de Componentes Principales)** para transformar las variables originales en componentes que no revelan información personal pero sí capturan los patrones de fraude.

### ¿Por qué Time y Amount NO están en PCA?

Porque son **variables que sí se pueden compartir sin problema**: el tiempo transcurrido y el monto de la transacción no identifican a nadie por sí solos.

## 3.3 ¿Hubo imputación de datos?

**NO.** El dataset **no tiene valores nulos** (0 nulos en las 31 columnas). No fue necesario imputar nada.

**¿Qué es imputación?**
Es el proceso de reemplazar valores faltantes (NaN) con algún valor estimado: la media, la mediana, un modelo predictivo, etc. En este caso no fue necesario porque el dataset ya venía limpio.

## 3.4 Valores Atípicos (Outliers)

### ¿Salieron outliers?

**SÍ, en la variable `Amount`.** Hay transacciones con montos extremadamente altos (hasta $25,691) que están muy lejos de la media ($88.29 para legítimas, $122.21 para fraude).

### ¿Qué hicimos con los outliers?

**NO los eliminamos.** Razones:
1. **En fraude, los outliers son información valiosa**: una transacción de $25,000 podría ser precisamente un fraude. Eliminarlos sería perder las transacciones más sospechosas.
2. **Amount se escaló con StandardScaler**: esto reduce el impacto de los valores extremos sin eliminarlos.
3. **Los árboles de decisión (Random Forest, XGBoost) son robustos a outliers**: no se ven tan afectados como la regresión lineal.

### ¿Qué tan balanceados están los outliers con respecto a la variable objetivo?

Los outliers de `Amount` están **distribuidos en ambas clases**:
- Transacciones legítimas con montos altos: existen pero son pocas
- Transacciones fraudulentas con montos altos: también existen

**Dato contraintuitivo**: el monto promedio de fraude ($122.21) es **menor** que el de transacciones legítimas. Esto significa que los defraudadores hacen muchas transacciones pequeñas para no ser detectados. Los outliers de Amount están más correlacionados con transacciones legítimas de alto valor (compras grandes reales).

## 3.5 Data Desbalanceada

### ¿El dataset está desbalanceado?

**EXTREmadamente.** Solo el **0.17%** son fraudes:
- Legítimas: 284,315 (99.83%)
- Fraude: 492 (0.17%)

### ¿Qué hicimos con el desbalance?

**SMOTE (Synthetic Minority Over-sampling Technique)**

**¿Qué es SMOTE?**
Crea instancias **sintéticas** de la clase minoritaria (fraude) interpolando entre vecinos cercanos en el espacio de features. No simplemente duplica filas, crea nuevas combinaciones realistas.

**¿Por qué SMOTE y no otras técnicas?**

| Técnica | ¿Por qué NO la usamos? |
|---|---|
| **Undersampling** | Perderíamos 283,000+ filas de datos legítimos valiosos |
| **Oversampling simple** | Duplicar filas causa overfitting (el modelo memoriza) |
| **SMOTE** ✅ | Genera datos sintéticos variados sin perder información |
| **class_weight='balanced'** | Lo probamos implícitamente pero SMOTE dio mejor Recall |

**¿Dónde aplicamos SMOTE?**
**SOLO en el conjunto de entrenamiento** (X_train, y_train). NUNCA en el test. Si aplicás SMOTE antes del split, hay **data leakage**: el modelo ve datos sintéticos derivados del test y parece mejor de lo que es.

**Resultado de SMOTE:**
- Antes: 227,000 train rows (492 fraudes)
- Después: 454,000 train rows (227,000 fraudes sintéticos + 227,000 legítimas)
- **El dataset de entrenamiento quedó 50/50**

## 3.6 Feature Engineering en Clasificación

### ¿Qué transformación hicimos?

Solo **escalado** con `StandardScaler` en `Time` y `Amount`:

```python
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
```

**¿Por qué escalar Time y Amount?**
- Las variables V1-V28 ya vienen escaladas por el PCA
- Time (0 a 172,792 segundos) y Amount (0 a $25,691) tienen rangos totalmente distintos
- Sin escalar, Amount dominaría la distancia euclidiana en SMOTE y en los modelos

**¿Por qué NO hicimos más feature engineering?**
Porque las variables V1-V28 ya son transformaciones PCA. No sabemos qué representan originalmente, así no podemos crear interacciones significativas entre ellas.

## 3.7 Selección de Variables

### ¿Qué variables usamos?

**TODAS las 30 variables**: V1-V28, Time_scaled, Amount_scaled

**¿Por qué todas y no un subconjunto?**
1. Las 28 variables PCA ya son una reducción dimensional (de probablemente 50+ variables originales)
2. Cada componente PCA captura información diferente
3. Eliminar variables PCA podría quitar información de fraude

### ¿Cuáles variables son más importantes?

Las más correlacionadas con fraude (correlación negativa):
1. **V17** → correlación más fuerte con fraude
2. **V14** → segunda más correlacionada
3. **V12** → tercera más correlacionada

Estas variables tienen coeficientes **negativos**: cuando V17, V14 o V12 son bajos, hay más probabilidad de fraude.

## 3.8 Modelos de Machine Learning

### ¿Cuáles modelos evaluamos?

| # | Modelo | ¿Lo usamos? | ¿Por qué? |
|---|--------|-------------|-----------|
| 1 | **Regresión Logística** | ✅ Sí | Baseline lineal. Simple, interpretable, rápido. |
| 2 | **Random Forest** | ✅ Sí | Ensemble robusto, maneja no-linealidades, no necesita tuning agresivo. |
| 3 | **XGBoost** | ✅ Sí | State-of-the-art en competencias, gradient boosting optimizado. |
| 4 | **Gradient Boosting** | ✅ Sí | Similar a XGBoost pero implementación de scikit-learn. |

### ¿Cuáles modelos NO evaluamos y por qué?

| Modelo | ¿Por qué NO? |
|---|---|
| **SVM** | Muy lento con 284K filas. No escala bien a datasets grandes. |
| **K-Nearest Neighbors** | Pésimo rendimiento en datos de alta dimensionalidad (28 features). |
| **Naive Bayes** | Asume independencia entre features, pero las PCA están correlacionadas. |
| **Redes Neuronales** | Requerirían mucho más tuning y tiempo. Fuera del scope del proyecto. |
| **Isolation Forest** | Es no supervisado (detección de anomalías). Nuestro problema es supervisado. |

### ¿Cuál modelo ganó y por qué?

**🏅 Random Forest** fue el ganador:

| Modelo | ROC-AUC | F1-Score | Recall | Precision |
|---|---|---|---|---|
| **Random Forest** | **0.9841** | **0.8229** | **0.8061** | **0.8404** |
| Gradient Boosting | 0.9807 | 0.1888 | 0.8980 | 0.1055 |
| XGBoost | 0.9792 | 0.8018 | 0.8163 | 0.7878 |
| Reg. Logística | 0.9698 | 0.1094 | 0.0612 | 0.5217 |

**¿Por qué Random Forest ganó?**
1. **Mejor balance general**: mayor ROC-AUC (0.9841) y F1 (0.8229)
2. **Recall de 81%**: detecta 81 de cada 100 fraudes reales
3. **Precision de 84%**: cuando dice "fraude", acierta el 84% del tiempo
4. **Robusto a overfitting**: el ensemble de 100 árboles promedia las predicciones
5. **No necesita tuning agresivo**: funciona bien con parámetros por defecto

**¿Por qué Gradient Boosting tiene Recall más alto (0.8980) pero NO es el mejor?**
Porque su **Precision es pésima (0.1055)**: de cada 100 veces que dice "fraude", solo 10 son realmente fraude. Generaría 90% de falsas alarmas. En la práctica, el banco perdería confianza en el sistema.

**¿Por qué Regresión Logística fue tan mala?**
Porque es un modelo **lineal** y la relación entre las variables PCA y el fraude es **no lineal**. No puede capturar patrones complejos.

## 3.9 ¿Por qué Recall como métrica principal?

### ¿Qué es Recall?

```
Recall = Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos)
Recall = TP / (TP + FN)
```

**En fraude:**
- **Verdadero Positivo (TP)**: Dijimos "fraude" y ERA fraude ✅
- **Falso Negativo (FN)**: Dijimos "legítima" pero ERA fraude ❌ ← **ESTO CUESTA DINERO**

### ¿Por qué Recall y no Accuracy?

Con 0.17% de fraude, un modelo que diga **SIEMPRE "legítima"** tendría:
- **Accuracy: 99.83%** ← Parece perfecto
- **Recall: 0%** ← No detectó NINGÚN fraude

La accuracy es **engañosa** en datasets desbalanceados.

### ¿Por qué Recall y no Precision?

| Escenario | Consecuencia |
|---|---|
| **Falso Negativo** (fraude no detectado) | 💰 El banco pierde dinero real |
| **Falso Positivo** (legítima marcada como fraude) | 😤 El cliente se molesta, pero no hay pérdida directa |

**Es peor dejar pasar un fraude que bloquear una transacción legítima.** Por eso maximizamos Recall.

### ¿Por qué no solo Recall?

Porque un modelo que diga **SIEMPRE "fraude"** tendría Recall = 100% pero sería inútil. Por eso usamos **múltiples métricas**:
- **ROC-AUC**: poder discriminativo general
- **F1-Score**: balance entre Precision y Recall
- **Precision**: qué tan confiable es cuando dice "fraude"
- **Recall**: qué porcentaje de fraudes reales detecta

## 3.10 Inferencia en Clasificación

### ¿Cómo funciona la inferencia?

1. El usuario ingresa los valores V1-V28, Time y Amount en la app web
2. La app escala Time y Amount con el mismo StandardScaler del entrenamiento
3. El modelo Random Forest recibe el vector de 30 features
4. Devuelve: predicción (0 o 1) + probabilidad de fraude

### ¿Dónde se generó el modelo?

En el **notebook** `Clasificacion/notebook/fraude_clasificacion.ipynb`:

```python
joblib.dump(modelo, '../../../app/clasificacion_app/model/modelo_fraude.pkl')
```

El modelo se guarda como archivo `.pkl` (pickle) en `app/clasificacion_app/model/`.

### ¿Dónde se carga el modelo para inferencia?

En `app/app.py`:

```python
models['fraude_modelo'] = joblib.load(
    os.path.join(BASE_DIR, 'clasificacion_app', 'model', 'modelo_fraude.pkl'))
```

Y se usa en el endpoint `/api/predict/fraude`.

---

# 4. CASO 2 — AGRUPAMIENTO: SPOTIFY CLUSTERING

## 4.1 Variable Objetivo

**NO HAY variable objetivo.** Este es un problema **no supervisado**.

**¿Por qué no hay variable objetivo?**
Porque no tenemos etiquetas. No sabemos de antemano qué "tipo" de canción es cada una. El algoritmo debe **descubrir solo** los patrones y agrupar canciones que "suenan parecido".

**¿Qué usamos en lugar de variable objetivo?**
Las **10 features de audio** que usamos para clustering:
`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `popularity`

## 4.2 Dataset: Spotify Tracks

| Característica | Valor |
|---|---|
| **Filas** | ~114,000 canciones |
| **Columnas** | 20+ (metadatos + features de audio) |
| **Géneros** | 125 géneros diferentes |
| **Valores nulos** | **NINGUNO** (0 nulos) |
| **Origen** | Kaggle, API de Spotify |

### ¿Por qué elegimos esas 10 features para clustering?

| Feature | ¿Qué mide? | ¿Por qué incluirla? |
|---|---|---|
| **danceability** | Qué tan bailable es (0-1) | Diferencia música de baile vs. reposo |
| **energy** | Intensidad y actividad (0-1) | Diferencia música intensa vs. suave |
| **loudness** | Volumen en dB | Correlacionado con energy, pero aporta info única |
| **speechiness** | Presencia de palabras habladas | Diferencia rap/podcast de música melódica |
| **acousticness** | Si es acústica (0-1) | Diferencia música electrónica de acústica |
| **instrumentalness** | Si no tiene voz (0-1) | Diferencia canciones de instrumentales |
| **liveness** | Si fue grabada en vivo | Diferencia estudio de conciertos |
| **valence** | Positividad emocional (0-1) | Diferencia música alegre de triste |
| **tempo** | BPM | Diferencia ritmos rápidos de lentos |
| **popularity** | Popularidad 0-100 | Diferencia hits de canciones nicho |

### ¿Por qué NO usamos `duration_ms`, `key`, `mode`, `time_signature`?

- **duration_ms**: La duración no dice nada sobre el "estilo" musical. Una canción de 3 min y una de 5 min pueden sonar igual.
- **key**: La tonalidad musical (Do, Re, Mi...) no agrupa por género. Todos los géneros usan todas las tonalidades.
- **mode**: Mayor/menor es muy binario y no aporta suficiente diferenciación.
- **time_signature**: El compás (3/4, 4/4) es demasiado genérico.

## 4.3 ¿Hubo imputación de datos?

**NO.** El dataset **no tiene valores nulos**. Solo hicimos `.dropna()` como precaución en las features seleccionadas.

## 4.4 Valores Atípicos (Outliers)

### ¿Salieron outliers?

**SÍ, en `tempo` y `loudness`:**
- **tempo**: algunas canciones tienen BPM extremos (muy lentas o muy rápidas)
- **loudness**: algunas canciones son extremadamente silenciosas o ruidosas

### ¿Qué hicimos con los outliers?

**NO los eliminamos.** En su lugar:
1. **StandardScaler**: normaliza todas las features a media 0 y desviación 1, reduciendo el impacto de outliers
2. **K-Means es sensible a outliers**, pero con 114K canciones, los outliers individuales no afectan significativamente los centroides

### ¿Qué tan balanceados están los outliers con respecto a... qué?

**En clustering no hay variable objetivo**, así que no se habla de "balance con respecto al target". Lo que sí analizamos fue:
- Si los outliers se concentraban en un cluster específico
- Si distorsionaban los centroides

La respuesta: **no hubo distorsión significativa** gracias al escalado y al tamaño del dataset.

## 4.5 Data Desbalanceada

### ¿Hay desbalance en clustering?

**No aplica el concepto de "desbalance"** como en clasificación supervisada, porque no hay clases predefinidas.

Sin embargo, los clusters resultantes tienen **tamaños desiguales**:

| Cluster | Canciones | % |
|---|---|---|
| 0 — Energía y Baile | 29,986 | 26.3% |
| 1 — Acústico y Relajado | 24,637 | 21.6% |
| 2 — Instrumental | 38,284 | 33.6% |
| 3 — Hablado | 7,670 | 6.7% |
| 4 — En Vivo | 12,159 | 10.7% |
| 5 — Melódico Positivo | 1,264 | 1.1% |

**¿Es problema que el cluster 5 tenga solo 1,264 canciones?**
No necesariamente. Refleja que hay un tipo de música genuinamente raro en el dataset (música muy melódica y positiva). Si forzamos clusters del mismo tamaño, perderíamos esta granularidad.

## 4.6 Feature Engineering en Agrupamiento

### ¿Qué transformación hicimos?

Solo **StandardScaler**:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[features_cluster])
```

**¿Por qué es OBLIGATORIO escalar en K-Means?**
Porque K-Means usa **distancia euclidiana**. Si no escalás:
- `tempo` (rango: 50-200 BPM) dominaría la distancia
- `danceability` (rango: 0-1) sería irrelevante
- El clustering reflejaría solo el tempo, no las demás features

### ¿Por qué NO hicimos más feature engineering?

Porque en clustering no supervisado, crear features derivadas (interacciones, polinomios) es arriesgado: no tenés una variable objetivo para validar si la nueva feature ayuda o no. Las features originales de Spotify ya están bien diseñadas.

## 4.7 Selección de K (Número de Clusters)

### ¿Cómo elegimos K=6?

Usamos **dos métodos**:

1. **Método del Codo**: graficamos Inercia vs K y buscamos el "codo" (punto de inflexión)
2. **Silhouette Score**: medimos qué tan bien separados están los clusters

| K | Silhouette Score | ¿Por qué NO? |
|---|---|---|
| 2 | Más alto | Demasiado grueso: solo "bailable" vs "no bailable" |
| 3-4 | Bueno | Todavía muy general para 125 géneros |
| **6** | **Moderado (0.1652)** | **Balance ideal: granularidad sin fragmentar** |
| 7-10 | Bajo | Fragmentación excesiva, clusters muy similares |

**¿Por qué K=6 y no K=2 (que tenía mejor Silhouette)?**
K=2 solo separa "bailable" vs "no bailable" — demasiado grueso para 125 géneros. K=6 preserva la diversidad musical real sin fragmentar excesivamente.

## 4.8 Modelos de Machine Learning

### ¿Cuáles modelos evaluamos?

| # | Modelo | ¿Lo usamos? | ¿Por qué? |
|---|--------|-------------|-----------|
| 1 | **K-Means** | ✅ Sí | Algoritmo principal. Rápido, escalable, interpretable. |
| 2 | DBSCAN | ❌ No | No funciona bien en alta dimensionalidad. Sensible a parámetros. |
| 3 | Agglomerative Clustering | ❌ No | Muy lento con 114K filas. O(n²) o O(n³). |

### ¿Por qué K-Means y no otros?

| Modelo | Ventaja | Desventaja (por qué NO) |
|---|---|---|
| **K-Means** ✅ | Rápido O(n), escalable, fácil de interpretar | Necesitas elegir K |
| **DBSCAN** | No necesitas elegir K, detecta outliers | Sensible a ε y min_samples. No funciona bien con 10 dimensiones. |
| **Agglomerative** | No necesitas elegir K, dendrograma visual | O(n²) de complejidad. Con 114K filas tardaría HORAS. |
| **GMM** | Clusters probabilísticos (suaves) | Más complejo, converge a óptimos locales |

### Métricas de evaluación

| Métrica | Valor | ¿Qué significa? |
|---|---|---|
| **Silhouette Score** | 0.1652 | Moderado. Típico en datos de audio de alta dimensionalidad. |
| **Davies-Bouldin** | 1.6242 | Menor es mejor. 1.62 indica separación aceptable. |
| **PCA varianza** | 43.0% | 2 componentes explican el 43% de la varianza. La visualización es orientativa. |

## 4.9 Inferencia en Agrupamiento

### ¿Cómo funciona la inferencia?

1. El usuario ingresa 10 valores de audio en la app
2. La app escala con el mismo StandardScaler del entrenamiento
3. K-Means predice a qué cluster pertenece la canción
4. Se devuelve: cluster ID, nombre descriptivo, distancia al centroide

### ¿Dónde se generó el modelo?

En el **notebook** `Agrupamiento/notebook/spotify_clustering.ipynb`:

```python
joblib.dump(kmeans, '../../../app/agrupamiento_app/model/kmeans_spotify.pkl')
joblib.dump(scaler, '../../../app/agrupamiento_app/model/scaler_spotify.pkl')
joblib.dump(pca, '../../../app/agrupamiento_app/model/pca_spotify.pkl')
```

Se guardan **3 archivos**: el modelo K-Means, el scaler y el PCA (para visualización).

---

# 5. CASO 3 — REGRESIÓN: PREDICCIÓN DE SEGURO MÉDICO

## 5.1 Variable Objetivo

**Variable:** `charges`
- Tipo: float (continua)
- Unidad: USD (dólares estadounidenses)
- Rango: ~$1,122 a ~$63,770
- Media: ~$13,270
- Mediana: ~$9,382

**¿Por qué esta es la variable objetivo?**
Porque el problema de negocio es: "¿Cuánto va a costar el seguro médico de esta persona?" Es una pregunta de **cuánto** → regresión (predecir un valor numérico continuo).

## 5.2 Dataset: Medical Cost Personal

| Característica | Valor |
|---|---|
| **Filas** | 1,338 asegurados |
| **Columnas** | 7 (6 features + 1 target) |
| **Features** | age, sex, bmi, children, smoker, region |
| **Valores nulos** | **NINGUNO** (0 nulos) |
| **Distribución de charges** | Asimétrica positiva (skewness > 1.5) |
| **Origen** | Kaggle, datos de seguros de EE.UU. |

## 5.3 ¿Hubo imputación de datos?

**NO.** El dataset **no tiene valores nulos** (0 nulos en las 7 columnas). No fue necesario imputar nada.

## 5.4 Valores Atípicos (Outliers)

### ¿Salieron outliers?

**SÍ, en `charges`:** la distribución es **asimétrica positiva** (skewness > 1.5). Hay personas con costos extremadamente altos ($40,000-$63,770) que están muy lejos de la mediana ($9,382).

### ¿Por qué hay outliers en charges?

Porque los **fumadores con BMI alto** tienen costos exponencialmente mayores. Un fumador obeso puede costar 5-6× más que un no fumador saludable.

### ¿Qué hicimos con los outliers?

**NO los eliminamos.** Razones:
1. **Son datos reales y válidos**: representan el riesgo real de la aseguradora
2. **Los modelos basados en árboles (Gradient Boosting, XGBoost) manejan bien los outliers**
3. **Eliminarlos sesgaría el modelo**: la aseguradora necesita predecir también los casos caros

### ¿Qué tan balanceados están los outliers con respecto a la variable objetivo?

En regresión no hay "clases", pero analizamos la distribución:
- Los outliers de `charges` están **fuertemente correlacionados con `smoker=yes`**
- Fumadores: promedio $32,050 (con muchos outliers arriba de $50,000)
- No fumadores: promedio $8,434 (pocos outliers)

**Conclusión**: los outliers no son errores, son el efecto sinérgico de fumar + BMI alto.

## 5.5 Data Desbalanceada

### ¿Hay desbalance en regresión?

**No aplica** el concepto de desbalance como en clasificación. En regresión, la variable objetivo es continua, no categórica.

Sin embargo, la distribución de `charges` es **asimétrica** (no normal):
- Media: $13,270
- Mediana: $9,382
- La media > mediana → cola larga hacia la derecha (valores altos)

**¿Hicimos algo con la asimetría?**
Consideramos aplicar `log1p(charges)` para normalizarla, pero decidimos **no hacerlo** porque:
1. Los modelos basados en árboles no asumen normalidad
2. Interpretar MAE en log-charges es menos intuitivo para el negocio
3. Gradient Boosting maneja bien distribuciones asimétricas

## 5.6 Feature Engineering

### ¿Qué es Feature Engineering?

Es el proceso de **crear nuevas variables** a partir de las existentes para que el modelo pueda capturar relaciones más complejas.

### ¿Qué feature engineering hicimos en Regresión?

**Dos features nuevas:**

1. **`bmi_smoker`** (interacción):
   ```python
   df['bmi_smoker'] = df['bmi'] * df['smoker_enc']
   ```
   **¿Por qué?** El efecto del BMI en el costo es **diferente** para fumadores y no fumadores. Un BMI alto en un fumador es mucho más peligroso que en un no fumador. Esta interacción captura ese **efecto sinérgico**.

2. **`age_sq`** (término cuadrático):
   ```python
   df['age_sq'] = df['age'] ** 2
   ```
   **¿Por qué?** El costo médico no crece linealmente con la edad. Crece **aceleradamente**: la diferencia entre 20 y 30 años es menor que entre 50 y 60. El término cuadrático captura esta **no-linealidad**.

### ¿Por qué NO hicimos feature engineering en los otros casos?

- **Clasificación**: las variables V1-V28 ya son PCA (transformadas). No sabemos qué representan originalmente.
- **Agrupamiento**: en no supervisado, no hay variable objetivo para validar si una feature nueva ayuda.

### ¿Cuál fue el impacto del feature engineering?

**`bmi_smoker` dominó con 80.67% de importancia** en el modelo Gradient Boosting. Sin esta interacción, los modelos lineales no podían capturar que el BMI afecta diferente a fumadores vs no fumadores.

## 5.7 Selección de Variables

### ¿Qué variables usamos?

| Variable | Tipo | Transformación |
|---|---|---|
| `age` | numérica | Original + `age_sq` (cuadrática) |
| `sex` | categórica | Binaria: male=1, female=0 |
| `bmi` | numérica | Original + interacción `bmi_smoker` |
| `children` | numérica | Original |
| `smoker` | categórica | Binaria: yes=1, no=0 |
| `region` | categórica | One-hot encoding (3 dummies: northwest, southeast, southwest) |

### ¿Por qué estas variables y no otras?

**Porque son TODAS las variables disponibles.** El dataset solo tiene 6 features + 1 target. No había variables adicionales para elegir.

### ¿Por qué One-Hot Encoding para region y no Label Encoding?

Porque `region` es **nominal** (no tiene orden):
- southwest, southeast, northwest, northeast → no hay un "mayor que" entre ellas
- Si usás Label Encoding (0, 1, 2, 3), el modelo pensaría que northeast (3) > southwest (0), lo cual no tiene sentido
- One-Hot Encoding crea columnas binarias independientes

### ¿Por qué encoding binario para sex y smoker?

Porque son **binarias** (solo 2 valores):
- sex: male/female → 1/0
- smoker: yes/no → 1/0
- One-Hot Encoding crearía 2 columnas redundantes (si sabés que no es male, automáticamente es female)

## 5.8 Modelos de Machine Learning

### ¿Cuáles modelos evaluamos?

| # | Modelo | ¿Lo usamos? | ¿Por qué? |
|---|--------|-------------|-----------|
| 1 | **Regresión Lineal** | ✅ Sí | Baseline. Simple, interpretable. |
| 2 | **Ridge** | ✅ Sí | Regresión lineal con regularización L2. Previene overfitting. |
| 3 | **Lasso** | ✅ Sí | Regresión lineal con regularización L1. Puede eliminar features. |
| 4 | **Random Forest** | ✅ Sí | Captura no-linealidades e interacciones automáticamente. |
| 5 | **XGBoost** | ✅ Sí | Gradient boosting optimizado. State-of-the-art. |
| 6 | **Gradient Boosting** | ✅ Sí | Similar a XGBoost, implementación scikit-learn. |

### ¿Cuáles modelos NO evaluamos y por qué?

| Modelo | ¿Por qué NO? |
|---|---|
| **SVR (Support Vector Regression)** | Muy lento, sensible a escalado, difícil de interpretar. |
| **K-Neighbors Regressor** | Pésimo en datos de baja dimensionalidad con relaciones complejas. |
| **Redes Neuronales** | Overkill para 1,338 registros. Se sobreajustarían fácilmente. |
| **ElasticNet** | Combinación de Ridge+Lasso. Redundante con los que ya evaluamos. |

### ¿Cuál modelo ganó y por qué?

**🏅 Gradient Boosting** fue el ganador:

| Modelo | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| **Gradient Boosting** | **2,495.65** | **4,440.17** | **0.8730** |
| XGBoost | 2,583.13 | 4,518.97 | 0.8685 |
| Ridge | 2,763.69 | 4,529.65 | 0.8678 |
| Regresión Lineal | 2,755.47 | 4,537.67 | 0.8674 |
| Lasso | 2,761.84 | 4,544.74 | 0.8669 |
| Random Forest | 2,506.33 | 4,666.44 | 0.8597 |

**¿Por qué Gradient Boosting ganó?**
1. **Mayor R² (0.8730)**: explica el 87.3% de la varianza del costo
2. **Menor MAE ($2,495)**: error promedio de ~$2,500 al año
3. **Menor RMSE ($4,440)**: penaliza menos los errores grandes que Random Forest
4. **Captura no-linealidades**: la interacción BMI-tabaquismo y la edad cuadrática

**¿Por qué los modelos lineales (Ridge, Lasso) fueron tan competitivos?**
Porque el feature engineering (`bmi_smoker`, `age_sq`) ya capturó las no-linealidades principales. Con esas features, la regresión lineal funciona sorprendentemente bien (R² = 0.867).

**¿Por qué Random Forest fue el peor de los ensembles?**
Porque tiende a **sobreajustar** en datasets pequeños (1,338 filas). Gradient Boosting es más conservador y generaliza mejor.

### Top 5 Features más importantes (Gradient Boosting):

| # | Variable | Importancia | ¿Qué significa? |
|---|---|---|---|
| 1 | `bmi_smoker` | **80.67%** | La interacción BMI×tabaquismo DOMINA todo |
| 2 | `age` | 7.52% | La edad importa, pero mucho menos |
| 3 | `bmi` | 4.47% | El BMI por sí solo tiene efecto moderado |
| 4 | `age_sq` | 4.36% | La no-linealidad de la edad es relevante |
| 5 | `smoker_enc` | 1.33% | Ser fumador importa, pero menos que la interacción |

**Dato clave**: `bmi_smoker` (80.67%) + `smoker_enc` (1.33%) = **82% del modelo depende del tabaquismo**.

## 5.9 Métricas de Evaluación

### ¿Por qué MAE como métrica principal?

**MAE (Mean Absolute Error)**: error promedio en dólares.

```
MAE = (1/n) × Σ|y_real - y_predicho|
```

**¿Por qué MAE y no R²?**
- **MAE es interpretable para el negocio**: "el modelo se equivoca en promedio $2,495 por año"
- **R² es estadístico**: "explica el 87.3% de la varianza" → ¿qué significa eso para una aseguradora?
- **MAE está en la misma unidad** que la variable objetivo (dólares)

### ¿Por qué también usamos RMSE y R²?

| Métrica | ¿Qué mide? | ¿Por qué usarla? |
|---|---|---|
| **MAE** | Error promedio | Interpretable en dólares |
| **RMSE** | Error cuadrático promedio | Penaliza errores grandes (útil para riesgos extremos) |
| **R²** | Bondad de ajuste | Compara modelos de forma relativa |

## 5.10 Inferencia en Regresión

### ¿Cómo funciona la inferencia?

1. El usuario ingresa: age, bmi, children, smoker, sex, region
2. La app hace el feature engineering: `bmi_smoker = bmi × smoker_enc`, `age_sq = age²`
3. Aplica One-Hot Encoding para region
4. Escala con StandardScaler
5. El modelo Gradient Boosting predice el costo
6. Se devuelve el valor en USD

### ¿Dónde se generó el modelo?

En el **notebook** `Regresion/notebook/seguro_regresion.ipynb`:

```python
joblib.dump(modelo, '../../../app/regresion_app/model/modelo_seguro.pkl')
joblib.dump(scaler, '../../../app/regresion_app/model/scaler_seguro.pkl')
joblib.dump(list(X.columns), '../../../app/regresion_app/model/feature_names.pkl')
```

Se guardan **3 archivos**: el modelo, el scaler y los nombres de features (para asegurar el orden correcto).

---

# 6. APLICACIÓN WEB Y DASHBOARD

## 6.1 Arquitectura

**Framework:** Flask (Python)
**Puerto:** 5000
**URL:** http://localhost:5000

### Estructura:

```
app/
├── app.py                    ← Servidor Flask unificado
├── templates/
│   └── index.html            ← Frontend con 3 pestañas
├── static/
│   └── style.css             ← Diseño estilo Databricks
├── clasificacion_app/
│   └── model/
│       └── modelo_fraude.pkl
├── agrupamiento_app/
│   └── model/
│       ├── kmeans_spotify.pkl
│       ├── scaler_spotify.pkl
│       └── pca_spotify.pkl
└── regresion_app/
    └── model/
        ├── modelo_seguro.pkl
        ├── scaler_seguro.pkl
        └── feature_names.pkl
```

### ¿Por qué una app unificada y no 3 apps separadas?

1. **Un solo proceso**: `python app.py` levanta todo
2. **Navegación por sidebar**: el usuario cambia entre casos sin recargar
3. **Carga bajo demanda**: los dashboards se generan solo cuando se solicitan (lazy loading)
4. **Cache**: las gráficas del dashboard se cachean para no regenerarlas
5. **Graceful degradation**: si falta un modelo, la app sigue funcionando para los otros casos

## 6.2 Dashboard

### ¿Qué gráficas tiene cada dashboard?

#### Dashboard de Fraude:
1. **Pie chart**: distribución de clases (0.17% fraude)
2. **Histograma**: distribución de Amount por clase (escala log)
3. **Bar chart**: top 10 variables correlacionadas con fraude

#### Dashboard de Spotify:
1. **PCA scatter**: clusters en 2D con varianza explicada
2. **Histograma overlay**: distribución de danceability, energy, acousticness, valence
3. **Bar chart**: top 12 géneros en el dataset

#### Dashboard de Seguro:
1. **Histograma + box**: distribución de charges con línea de media
2. **Box plot**: fumadores vs no fumadores (charges)
3. **Scatter**: real vs predicho con línea diagonal ideal

### ¿Cómo se generan las gráficas?

- **Plotly**: gráficas interactivas (zoom, hover, tooltips)
- **Lazy loading**: se generan solo cuando el usuario hace clic en "Ver Dashboard"
- **Cache**: una vez generadas, se almacenan en `_dashboard_cache` para no regenerar
- **JSON**: las gráficas se serializan a JSON y se renderizan en el frontend con Plotly.js

## 6.3 ¿Cómo ejecutar?

```bash
cd "corte3/Proyecto Final/app"
python app.py
# Abrir http://localhost:5000
```

---

# 7. PREGUNTAS FRECUENTES DEL PROFESOR

## P1: ¿Cuál es la variable objetivo de cada dataset?

| Dataset | Variable Objetivo | Tipo |
|---|---|---|
| Credit Card Fraud | `Class` (0=legítima, 1=fraude) | Categórica binaria |
| Spotify Tracks | **NO HAY** (no supervisado) | N/A |
| Medical Cost | `charges` (costo en USD) | Numérica continua |

## P2: ¿Por qué eligieron esas variables y no otras?

- **Fraude**: usamos TODAS las 30 variables (V1-V28, Time, Amount) porque las PCA ya son una reducción dimensional
- **Spotify**: elegimos 10 features de audio que capturan dimensiones perceptivas de la música. Excluimos duration, key, mode, time_signature porque no agrupan por estilo musical
- **Seguro**: usamos TODAS las 6 variables disponibles. El dataset solo tiene esas

## P3: ¿Hicieron imputación de datos?

**NO en ningún caso.** Los 3 datasets venían sin valores nulos.

## P4: ¿Salieron valores atípicos? ¿Qué hicieron con ellos?

| Caso | Outliers | Acción |
|---|---|---|
| Fraude | Amount con valores extremos | NO eliminar. Escalar con StandardScaler. Son información valiosa. |
| Spotify | Tempo y loudness extremos | NO eliminar. StandardScaler los normaliza. |
| Seguro | Charges con cola larga (skewness > 1.5) | NO eliminar. Son datos reales de alto riesgo. Gradient Boosting los maneja bien. |

## P5: ¿Qué tan balanceados están los datos?

| Caso | Desbalance | Solución |
|---|---|---|
| Fraude | **Extremo**: 0.17% fraude | **SMOTE** en training set (227K → 454K filas, 50/50) |
| Spotify | No aplica (no supervisado) | Clusters desiguales pero naturales |
| Seguro | No aplica (regresión) | Distribución asimétrica pero válida |

## P6: ¿Qué modelo de ML usaron? ¿Cuáles observaron? ¿Cuáles descartaron?

### Clasificación (Fraude):
- **Usamos**: Random Forest ✅, XGBoost ✅, Gradient Boosting ✅, Regresión Logística ✅
- **Descartamos**: SVM (lento con 284K filas), KNN (pésimo en alta dimensionalidad), Naive Bayes (asume independencia), Redes Neuronales (fuera de scope)
- **Ganador**: Random Forest (ROC-AUC 0.9841, Recall 0.81)

### Agrupamiento (Spotify):
- **Usamos**: K-Means ✅
- **Descartamos**: DBSCAN (sensible a parámetros en 10D), Agglomerative (O(n²), muy lento con 114K), GMM (más complejo)
- **Ganador**: K-Means con K=6

### Regresión (Seguro):
- **Usamos**: Gradient Boosting ✅, XGBoost ✅, Random Forest ✅, Regresión Lineal ✅, Ridge ✅, Lasso ✅
- **Descartamos**: SVR (lento, difícil de interpretar), KNN Regressor (pésimo), Redes Neuronales (overkill para 1,338 filas)
- **Ganador**: Gradient Boosting (R² 0.8730, MAE $2,495)

## P7: ¿Por qué usaron Recall en clasificación?

Porque en fraude, un **falso negativo** (no detectar un fraude) cuesta dinero real. Un **falso positivo** (bloquear una transacción legítima) solo molesta al cliente. Es peor perder dinero que molestar.

## P8: ¿Qué es Feature Engineering? ¿Cuál hicieron?

Es crear nuevas variables a partir de las existentes para capturar relaciones más complejas.

**Solo en Regresión:**
- `bmi_smoker = bmi × smoker_enc` → captura el efecto sinérgico entre obesidad y tabaquismo (80.67% de importancia)
- `age_sq = age²` → captura la no-linealidad del costo con la edad

## P9: ¿Tuvieron problemas? ¿Cómo los solucionaron?

### Problema 1: Desbalance extremo en fraude
- **Problema**: 0.17% fraude → los modelos predecían todo como "legítimo"
- **Solución**: SMOTE en el training set → dataset 50/50

### Problema 2: Rangos muy distintos en Spotify
- **Problema**: tempo (50-200) vs danceability (0-1) → tempo dominaba la distancia euclidiana
- **Solución**: StandardScaler → todas las features con media 0 y desviación 1

### Problema 3: No-linealidad en seguro médico
- **Problema**: la regresión lineal no capturaba que el BMI afecta diferente a fumadores
- **Solución**: Feature engineering con `bmi_smoker` (interacción) → R² mejoró significativamente

### Problema 4: Data leakage con SMOTE
- **Problema**: si aplicás SMOTE antes del train/test split, el modelo ve datos del test
- **Solución**: primero split 80/20, DESPUÉS SMOTE solo en training

### Problema 5: Elegir K en clustering
- **Problema**: K=2 tenía mejor Silhouette pero era demasiado grueso
- **Solución**: K=6 como balance entre granularidad e interpretabilidad

## P10: ¿Cómo y dónde generaron los modelos?

**Dónde**: En los notebooks Jupyter de cada caso:
- `Clasificacion/notebook/fraude_clasificacion.ipynb`
- `Agrupamiento/notebook/spotify_clustering.ipynb`
- `Regresion/notebook/seguro_regresion.ipynb`

**Cómo**: Con `joblib.dump()`:
```python
joblib.dump(modelo, 'ruta/al/modelo.pkl')
```

**Dónde se cargan**: En `app/app.py` con `joblib.load()` al iniciar el servidor Flask.

## P11: ¿Por qué CRISP-DM?

Porque es el estándar de la industria, es iterativo, y separa el entendimiento del negocio del modelado técnico.

## P12: ¿Por qué Flask y no Streamlit o Django?

- **Flask**: ligero, flexible, control total del routing y API
- **Streamlit**: más fácil pero menos flexible para dashboards custom
- **Django**: overkill para una app de demostración

## P13: ¿Por qué Plotly y no Matplotlib o Seaborn?

- **Plotly**: gráficas interactivas (zoom, hover, tooltips) → mejor para dashboards web
- **Matplotlib/Seaborn**: estáticos, solo para análisis exploratorio en notebooks

---

# 8. GLOSARIO DE CONCEPTOS CLAVE

## Términos que el profesor puede preguntar:

### Clasificación
- **Clasificación binaria**: predecir entre 2 clases (sí/no, fraude/legítima)
- **Verdadero Positivo (TP)**: predijimos positivo y era positivo
- **Falso Positivo (FP)**: predijimos positivo pero era negativo (falsa alarma)
- **Verdadero Negativo (TN)**: predijimos negativo y era negativo
- **Falso Negativo (FN)**: predijimos negativo pero era positivo (error costoso)

### Métricas
- **Accuracy**: (TP+TN) / Total → engañosa en datos desbalanceados
- **Precision**: TP / (TP+FP) → de los que dije "positivo", ¿cuántos eran realmente?
- **Recall**: TP / (TP+FN) → de los que ERAN positivos, ¿cuántos detecté?
- **F1-Score**: media armónica de Precision y Recall → balance entre ambas
- **ROC-AUC**: área bajo la curva ROC → poder discriminativo general (0.5=random, 1.0=perfecto)
- **MAE**: error absoluto promedio → en la misma unidad que el target
- **RMSE**: raíz del error cuadrático promedio → penaliza errores grandes
- **R²**: proporción de varianza explicada → 0=peor que la media, 1=perfecto
- **Silhouette Score**: qué tan bien separados están los clusters → -1 a 1
- **Davies-Bouldin**: qué tan similares son los clusters entre sí → menor es mejor

### Técnicas
- **SMOTE**: crea instancias sintéticas de la clase minoritaria interpolando vecinos
- **StandardScaler**: normaliza features a media 0 y desviación 1
- **One-Hot Encoding**: convierte categóricas en columnas binarias
- **PCA**: reduce dimensionalidad preservando la máxima varianza
- **Stratified Split**: divide train/test preservando la proporción de clases

### Modelos
- **Regresión Logística**: modelo lineal para clasificación binaria
- **Random Forest**: ensemble de árboles de decisión con bagging
- **XGBoost**: gradient boosting optimizado con regularización
- **Gradient Boosting**: ensemble secuencial que corrige errores del anterior
- **K-Means**: clustering que minimiza la inercia intra-cluster
- **Ridge**: regresión lineal con penalización L2 (reduce coeficientes)
- **Lasso**: regresión lineal con penalización L1 (puede eliminar features)

### Conceptos
- **Data Leakage**: cuando información del test "se filtra" al entrenamiento
- **Overfitting**: el modelo memoriza los datos de entrenamiento, no generaliza
- **Underfitting**: el modelo es demasiado simple para capturar patrones
- **Feature Engineering**: crear nuevas variables a partir de las existentes
- **Supervisado**: hay variable objetivo (etiquetas)
- **No supervisado**: no hay variable objetivo, se descubren patrones solos

---

# 📝 RESUMEN EJECUTIVO PARA MEMORIZAR

## Los 3 casos en 1 minuto:

1. **🔴 FRAUDE** (Clasificación)
   - Target: `Class` (0=legítima, 1=fraude)
   - Dataset: 284,807 filas, 0.17% fraude, SIN nulos
   - Desbalance: SMOTE en training (227K → 454K)
   - Outliers: Amount extremos → NO eliminar, escalar
   - Modelos: 4 evaluados → **Random Forest gana** (ROC-AUC 0.9841, Recall 0.81)
   - Métrica principal: **Recall** (falsos negativos = dinero perdido)

2. **🎵 SPOTIFY** (Agrupamiento)
   - Target: **NO HAY** (no supervisado)
   - Dataset: 114,000 canciones, 125 géneros, SIN nulos
   - Features: 10 de audio → StandardScaler obligatorio
   - K=6 clusters → balance entre granularidad e interpretabilidad
   - Modelo: **K-Means** (Silhouette 0.1652, Davies-Bouldin 1.62)
   - Outliers: tempo/loudness → NO eliminar, escalar

3. **🏥 SEGURO** (Regresión)
   - Target: `charges` (costo en USD)
   - Dataset: 1,338 filas, SIN nulos, distribución asimétrica
   - Feature Engineering: `bmi_smoker` (80.67% importancia) + `age_sq`
   - Outliers: charges altos → NO eliminar, son riesgo real
   - Modelos: 6 evaluados → **Gradient Boosting gana** (R² 0.8730, MAE $2,495)
   - Métrica principal: **MAE** (interpretable en dólares)

## App Flask:
- Unificada, puerto 5000, 3 pestañas
- Dashboards con Plotly (lazy loading + cache)
- Modelos cargados con joblib desde `.pkl`

---

> 💡 **Tip para la sustentación**: Si no sabés una respuesta, volvé a los fundamentos:
> - "¿Qué problema de negocio estamos resolviendo?"
> - "¿Qué métrica importa para ese negocio?"
> - "¿Qué técnica es apropiada para ese tipo de problema?"
>
> El profesor valora más el **razonamiento** que la respuesta exacta.
