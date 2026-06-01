# 🏥 Caso de Regresión — Predicción de Costos de Seguro Médico

**Universidad Popular del Cesar**
**Asignatura:** Ciencia de Datos — Docente: Aimer Rivera Centeno
**Metodología:** CRISP-DM

## 📋 Descripción

Este módulo implementa un pipeline completo de regresión para predecir el costo de seguro médico (`charges`) de una persona a partir de sus características demográficas y de salud. Se entrenan y comparan 6 modelos de regresión para seleccionar el de mejor rendimiento.

## 📁 Estructura del Módulo

```
Regresion/
├── data/
│   └── insurance.csv              ← Dataset (colocar manualmente)
├── notebook/
│   └── seguro_regresion.ipynb     ← Notebook Jupyter completo (CRISP-DM)
├── paper/
│   ├── main.tex                   ← Artículo LaTeX (formato IEEE)
│   ├── referencias.bib            ← Referencias bibliográficas
│   ├── figuras/                   ← Figuras generadas (vacío inicialmente)
│   └── secciones/
│       ├── resumen.tex
│       ├── introduccion.tex
│       ├── metodologia.tex
│       ├── resultados.tex
│       └── conclusiones.tex
└── README.md                      ← Este archivo
```

## 🔗 Dataset

- **Medical Cost Personal Dataset** de Kaggle
- Descarga: https://www.kaggle.com/datasets/mirichoi0218/insurance
- Archivo: `Regresion/data/insurance.csv`
- **El estudiante debe colocar el archivo manualmente** antes de ejecutar el notebook

## 📊 Modelos Evaluados

| Modelo | Tipo |
|--------|------|
| Regresión Lineal | Lineal paramétrico |
| Ridge (L2) | Lineal regularizado |
| Lasso (L1) | Lineal regularizado |
| Random Forest | Ensemble (árboles) |
| XGBoost | Gradient Boosting optimizado |
| Gradient Boosting | Ensemble (árboles secuenciales) |

## 🚀 Cómo Ejecutar

### 1. Notebook (análisis y entrenamiento)

```bash
# Desde la carpeta Regresion/notebook/
jupyter notebook seguro_regresion.ipynb
```

El notebook:
- Realiza análisis exploratorio completo (EDA)
- Aplica ingeniería de features (interacciones, términos cuadráticos)
- Entrena y evalúa 6 modelos
- Guarda el mejor modelo en `app/regresion_app/model/`

### 2. App Web Flask

```bash
# Desde la carpeta app/regresion_app/
python app.py
```

- **Puerto:** 5003
- **Formulario:** http://localhost:5003/
- **Dashboard:** http://localhost:5003/dashboard

### 3. Paper LaTeX

```bash
# Compilar desde Regresion/paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 🧠 Ingeniería de Features Destacada

1. **`smoker_enc`**: Codificación binaria de fumador (0/1)
2. **`sex_enc`**: Codificación binaria de género (0/1)
3. **One-hot encoding** para `region` (3 dummies)
4. **`bmi_smoker`**: Interacción BMI × tabaquismo (captura sinergia)
5. **`age_sq`**: Término cuadrático de edad (modela no-linealidad)

## 📈 Métricas de Evaluación

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Error promedio en USD |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Penaliza errores grandes |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proporción de varianza explicada |

## 📁 Artefactos Generados

Los modelos entrenados se guardan en `app/regresion_app/model/`:

| Archivo | Descripción |
|---------|-------------|
| `modelo_seguro.pkl` | Modelo entrenado (mejor R²) |
| `scaler_seguro.pkl` | Scaler para normalizar inputs |
| `feature_names.pkl` | Nombres de features en orden |

## 📚 Referencias

- Duan, H. et al. (2021). *Prediction of individual medical costs using machine learning.*
- CMS (2022). *National Health Expenditure Data.*
- Choi, M. (2018). *Medical Cost Personal Datasets.* Kaggle.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* SIGKDD.
- Wirth, R. & Hipp, J. (2000). *CRISP-DM: Towards a standard process model for data mining.*
