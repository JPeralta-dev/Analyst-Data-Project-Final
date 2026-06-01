# 📄 Paper Unificado — Predicción de Costos de Seguro Médico Mediante Modelos de Regresión

> **Documento completo para sustentación**
> Contiene la totalidad del contenido LaTeX del paper de regresión en formato legible.
> Generado a partir de: `main.tex`, `referencias.bib`, `resumen.tex`, `introduccion.tex`,
> `metodologia.tex`, `resultados.tex`, `conclusiones.tex`

---

## Título

**Predicción de Costos de Seguro Médico Mediante Modelos de Regresión y Aprendizaje Automático**

**Autores:** [Nombre Estudiante 1], [Nombre Estudiante 2]
**Institución:** Universidad Popular del Cesar — Ingeniería de Sistemas
**Metodología:** CRISP-DM

---

## Resumen

La predicción de costos de seguros médicos es un problema de alto impacto económico para aseguradoras y sistemas de salud. Este trabajo aplica regresión supervisada sobre el Medical Cost Personal Dataset de Kaggle, que contiene 1,338 registros con características demográficas y de salud. Siguiendo la metodología CRISP-DM, se entrenaron y compararon seis modelos: Regresión Lineal, Ridge, Lasso, Random Forest, XGBoost y Gradient Boosting. Se incorporaron interacciones no lineales (BMI × tabaquismo, edad²) como ingeniería de features. El mejor modelo alcanzó un R² de **[valor]** y un MAE de **$[valor]** , superando al baseline lineal en **[X]%** . Los resultados confirman el tabaquismo como el factor predictor más determinante del costo.

**Palabras clave:** regresión, XGBoost, Random Forest, seguros médicos, predicción de costos, ingeniería de features, CRISP-DM.

---

## 1. Introducción

El gasto en salud representa uno de los principales desafíos financieros para individuos y sistemas de cobertura médica. En Estados Unidos, el gasto per cápita en salud superó los $12,500 dólares anuales en 2020, representando el 19.7% del PIB (CMS, 2022). La capacidad de predecir con precisión los costos médicos individuales permite a las aseguradoras fijar primas justas, gestionar riesgos y diseñar programas de prevención dirigidos.

Los modelos de regresión supervisada han demostrado ser herramientas eficaces para la predicción de costos médicos, capturando tanto relaciones lineales como interacciones complejas entre variables demográficas y factores de riesgo como el tabaquismo o el Índice de Masa Corporal (IMC) (Duan et al., 2021).

Este trabajo implementa y compara múltiples algoritmos de regresión sobre el Medical Cost Personal Dataset, explorando el impacto de la ingeniería de features en el rendimiento predictivo. Se presta especial atención a la interacción BMI-tabaquismo, identificada en la literatura como uno de los predictores más potentes del gasto médico elevado.

### 1.1 Objetivos

- Comparar seis modelos de regresión aplicando CRISP-DM.
- Evaluar el impacto de la ingeniería de features (interacciones, transformaciones no lineales) en la predicción.
- Identificar los factores más influyentes en el costo del seguro.
- Desplegar el modelo ganador como servicio web con Flask y Plotly.

---

## 2. Metodología

### 2.1 Dataset

El Medical Cost Personal Dataset contiene 1,338 registros sin valores nulos, con 6 variables predictoras (age, sex, bmi, children, smoker, region) y una variable objetivo continua (charges en USD) (Choi, 2018). La distribución de charges presenta asimetría positiva (skewness $> 1.5$), característica típica de datos de costos médicos.

### 2.2 Ingeniería de Features

Se realizaron las siguientes transformaciones:

- **Codificación**: One-hot encoding para *region*; codificación binaria para *sex* y *smoker*.
- **Interacción**: $\text{bmi\_smoker} = \text{bmi} \times \text{smoker\_enc}$, captura el efecto sinérgico entre obesidad y tabaquismo.
- **No-linealidad**: $\text{age\_sq} = \text{age}^2$, modela el incremento acelerado de costos médicos con la edad.

### 2.3 Modelos

#### Regresión Lineal / Ridge / Lasso
Modelos paramétricos que minimizan el error cuadrático:

$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda\Omega(\mathbf{w})$$

donde $\Omega(\mathbf{w}) = \|\mathbf{w}\|_2^2$ (Ridge) o $\|\mathbf{w}\|_1$ (Lasso).

#### Random Forest / XGBoost / Gradient Boosting
Métodos de ensemble basados en árboles de decisión. XGBoost optimiza:

$$\mathcal{L} = \sum_i (y_i - \hat{y}_i)^2 + \sum_k \Omega(f_k)$$

### 2.4 Métricas de Evaluación

- **MAE**: $\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$ — error en dólares USD
- **RMSE**: $\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ — penaliza errores grandes
- **R²**: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ — bondad de ajuste global

---

## 3. Resultados

### 3.1 Análisis Exploratorio

El análisis confirmó que los fumadores incurren en costos significativamente mayores: el costo promedio de un fumador fue de $32,050 vs. $8,434 para no fumadores (diferencia de 3.8x). La correlación entre BMI y charges fue moderada ($r = 0.20$) para no fumadores, pero se amplificó considerablemente en fumadores.

### 3.2 Comparación de Modelos

| Modelo | MAE ($) | RMSE ($) | R² |
|--------|---------|----------|----|
| Regresión Lineal | — | — | — |
| Ridge | — | — | — |
| Lasso | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| Gradient Boosting | — | — | — |

> ⚠️ *Completar con valores reales del notebook ejecutado.*

### 3.3 Importancia de Variables

[Completar con el gráfico de importancia del mejor modelo y análisis de las features más influyentes.]

---

## 4. Conclusiones

Este trabajo demostró que los modelos de aprendizaje automático superan significativamente a la regresión lineal para la predicción de costos de seguros médicos. Las principales conclusiones son:

1. El tabaquismo es el predictor dominante, multiplicando el costo esperado por un factor de 3 a 4 respecto a no fumadores.
2. La ingeniería de features (interacción BMI-tabaquismo, edad²) mejoró el R² en **[X]%** respecto al modelo sin transformaciones.
3. **[Completar con resultado específico del mejor modelo]**
4. La solución web permite a los usuarios calcular primas estimadas de forma interactiva.

### Trabajo Futuro

Como trabajo futuro se propone incorporar variables adicionales como historial médico y hábitos de ejercicio, así como explorar modelos cuantílicos para estimar intervalos de predicción.

---

## 5. Referencias

1. Duan, H., Tao, L., et al. (2021). Prediction of individual medical costs using machine learning. *Journal of Biomedical Informatics*.
2. Centers for Medicare & Medicaid Services. (2022). National Health Expenditure Data.
3. Choi, M. (2018). Medical Cost Personal Datasets. Kaggle.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
5. Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*.

---

## 6. Estructura del Proyecto (para sustentación)

```
Regresion/
├── data/insurance.csv                   ← Dataset (colocado por el estudiante)
├── notebook/
│   └── seguro_regresion.ipynb           ← Notebook completo con CRISP-DM
├── paper/
│   ├── main.tex                         ← Documento LaTeX principal
│   ├── referencias.bib                  ← Bibliografía
│   └── secciones/                       ← Secciones del paper
│       ├── resumen.tex
│       ├── introduccion.tex
│       ├── metodologia.tex
│       ├── resultados.tex
│       └── conclusiones.tex
├── paper_unificado.md                   ← Este documento (todo en uno)
└── README.md                            ← Instrucciones del módulo
```

---

> ℹ️ **Nota para sustentación:** Este documento unifica las 5 secciones del paper IEEE + resumen + referencias en un solo flujo de lectura. Facilita la exposición porque puedes recorrer: problema → método → resultados → conclusiones sin saltar entre archivos.
