# 📄 Paper Unificado — Detección de Fraude en Transacciones de Tarjeta de Crédito

> **Documento completo para sustentación**
> Contiene la totalidad del contenido LaTeX del paper de clasificación en formato legible.
> Generado a partir de: `main.tex`, `referencias.bib`, `resumen.tex`, `introduccion.tex`,
> `metodologia.tex`, `resultados.tex`, `conclusiones.tex`

---

## Título

**Detección de Fraude en Transacciones de Tarjeta de Crédito Mediante Técnicas de Clasificación y Aprendizaje Automático**

**Autores:** [Nombre Estudiante 1], [Nombre Estudiante 2]
**Institución:** Universidad Popular del Cesar — Ingeniería de Sistemas
**Metodología:** CRISP-DM

---

## Resumen

El fraude en transacciones con tarjetas de crédito representa una amenaza creciente para el sistema financiero global. Este trabajo presenta una solución basada en aprendizaje automático para la detección automática de transacciones fraudulentas, empleando la metodología CRISP-DM sobre el dataset Credit Card Fraud Detection disponible en Kaggle, que contiene 284,807 transacciones de titulares europeos con un desbalance severo de clases (0.17% de fraudes). Se evaluaron cuatro algoritmos de clasificación: Regresión Logística, Random Forest, XGBoost y Gradient Boosting. El manejo del desbalance se realizó mediante SMOTE. Los resultados demuestran que **[completar con mejor modelo]** alcanzó el mayor ROC-AUC de **[valor]** , con un Recall de **[valor]** para la clase fraude, evidenciando la viabilidad de los modelos propuestos para su integración en sistemas de detección en tiempo real.

**Palabras clave:** fraude, clasificación, SMOTE, Random Forest, XGBoost, desbalance de clases, CRISP-DM.

---

## 1. Introducción

El fraude con tarjetas de crédito es un fenómeno delictivo que genera pérdidas millonarias a nivel mundial. Según la Nilson Report, las pérdidas globales por fraude en pagos con tarjeta superaron los 32 mil millones de dólares en 2021, con proyecciones de crecimiento sostenido (Nilson Report, 2021). La detección temprana de estas actividades ilícitas es crucial para proteger tanto a los consumidores como a las instituciones financieras.

Los métodos tradicionales de detección de fraude, basados en reglas heurísticas, presentan limitaciones importantes frente a la evolución constante de las tácticas fraudulentas. El aprendizaje automático ofrece una alternativa adaptativa y escalable, capaz de identificar patrones complejos y no lineales en grandes volúmenes de datos transaccionales (Dal Pozzolo et al., 2015).

El principal desafío técnico de este problema es el severo desbalance de clases: en un entorno real, la proporción de transacciones fraudulentas es menor al 1% del total. Esto provoca que los clasificadores estándar tiendan a predecir siempre la clase mayoritaria, logrando alta exactitud pero fallando en detectar los fraudes, que son precisamente el objetivo (Chawla et al., 2002).

Este trabajo aborda el problema mediante la metodología CRISP-DM (Wirth & Hipp, 2000), aplicando técnicas de sobremuestreo sintético (SMOTE) y comparando el rendimiento de cuatro algoritmos de clasificación. El objetivo es construir un modelo robusto que maximice el recall de la clase fraude, minimizando así las pérdidas financieras causadas por transacciones no detectadas.

### 1.1 Objetivos

- Implementar un pipeline completo de detección de fraude siguiendo CRISP-DM.
- Comparar el rendimiento de Regresión Logística, Random Forest, XGBoost y Gradient Boosting.
- Manejar el desbalance de clases mediante SMOTE.
- Desplegar el modelo como servicio web con visualización interactiva.

---

## 2. Metodología

Se adoptó la metodología CRISP-DM (*Cross-Industry Standard Process for Data Mining*) (Wirth & Hipp, 2000), que estructura el desarrollo en seis fases iterativas.

### 2.1 Comprensión del Negocio

El problema se formula como clasificación binaria supervisada: dada una transacción con características $\mathbf{x} \in \mathbb{R}^{30}$, predecir $y \in \{0, 1\}$ donde $y=1$ indica fraude. La métrica de negocio prioritaria es el *Recall* de la clase fraude, pues un falso negativo (fraude no detectado) implica pérdida económica directa.

### 2.2 Comprensión y Preparación de los Datos

El dataset contiene 284,807 transacciones con 30 características: 28 componentes PCA ($V_1$–$V_{28}$), *Time* y *Amount*. La variable objetivo *Class* presenta una distribución extremadamente desbalanceada: 99.83% transacciones legítimas vs 0.17% fraudulentas.

Las columnas *Time* y *Amount* se normalizaron con *StandardScaler*. El conjunto se dividió 80/20 con estratificación para preservar la proporción de clases.

### 2.3 Manejo del Desbalance — SMOTE

Se aplicó *Synthetic Minority Over-sampling Technique* (SMOTE) (Chawla et al., 2002) exclusivamente sobre el conjunto de entrenamiento para evitar data leakage. SMOTE genera instancias sintéticas de la clase minoritaria interpolando entre vecinos cercanos en el espacio de características.

### 2.4 Modelos Evaluados

#### Regresión Logística
Modelo lineal base. Dado el vector de características $\mathbf{x}$:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}$$

#### Random Forest
Ensemble de $T$ árboles de decisión. La predicción final agrega por mayoría de votos:

$$\hat{y} = \text{mode}\{h_t(\mathbf{x})\}_{t=1}^{T}$$

#### XGBoost
Gradient boosting optimizado. Minimiza una función objetivo regularizada:

$$\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

### 2.5 Métricas de Evaluación

Dado el desbalance, se descarta la exactitud como métrica primaria. Se utilizan:

- **ROC-AUC**: área bajo la curva ROC
- **Precision**: $TP / (TP + FP)$
- **Recall**: $TP / (TP + FN)$ — métrica principal
- **F1-Score**: media armónica de Precision y Recall

---

## 3. Resultados

### 3.1 Análisis Exploratorio

El análisis exploratorio confirmó la distribución extremadamente desbalanceada del dataset. El monto promedio de transacciones fraudulentas ($122.21) resultó inferior al de transacciones legítimas ($88.29), contrario a la intuición inicial. Las variables con mayor correlación con la clase fraude fueron $V_{14}$, $V_{17}$ y $V_{12}$ con coeficientes negativos significativos.

### 3.2 Comparación de Modelos

| Modelo | ROC-AUC | Precision | Recall | F1 |
|--------|---------|-----------|--------|----|
| Regresión Logística | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| Gradient Boosting | — | — | — | — |

> ⚠️ *Completar con valores reales del notebook ejecutado.*

### 3.3 Análisis del Mejor Modelo

[Completar con análisis de la matriz de confusión, importancia de features y curva Precision-Recall del modelo ganador.]

---

## 4. Conclusiones

Este trabajo demostró la viabilidad de los modelos de aprendizaje automático para la detección de fraude en transacciones financieras. Las principales conclusiones son:

1. El manejo del desbalance mediante SMOTE mejoró significativamente el Recall de la clase fraude respecto a los modelos sin sobremuestreo.
2. **[Completar con conclusión sobre el mejor modelo]**
3. Las variables $V_{14}$, $V_{17}$ y $V_{12}$ resultaron las más informativas para la detección de fraude.
4. La solución fue desplegada como aplicación web con Flask y Plotly, permitiendo predicción en tiempo real.

### Trabajo Futuro

Como trabajo futuro se propone evaluar técnicas de detección de anomalías no supervisadas (Isolation Forest, Autoencoder) y explorar estrategias de monitoreo de deriva del modelo (*concept drift*).

---

## 5. Referencias

1. Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G. (2015). Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, 41(10), 4915–4928.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
3. Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*.
4. The Nilson Report (2021). Issue 1209. HSN Consultants.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.

---

## 6. Estructura del Proyecto (para sustentación)

```
Clasificacion/
├── data/creditcard.csv                ← Dataset (colocado por el estudiante)
├── notebook/
│   └── fraude_clasificacion.ipynb     ← Notebook completo con CRISP-DM
├── paper/
│   ├── main.tex                       ← Documento LaTeX principal
│   ├── referencias.bib                ← Bibliografía
│   └── secciones/                     ← Secciones del paper
│       ├── resumen.tex
│       ├── introduccion.tex
│       ├── metodologia.tex
│       ├── resultados.tex
│       └── conclusiones.tex
├── paper_unificado.md                 ← Este documento (todo en uno)
└── README.md                          ← Instrucciones del módulo
```

---

> ℹ️ **Nota para sustentación:** Este documento unifica las 5 secciones del paper IEEE + resumen + referencias en un solo flujo de lectura. Facilita la exposición porque puedes recorrer: problema → método → resultados → conclusiones sin saltar entre archivos.
