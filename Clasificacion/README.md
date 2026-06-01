# 🔴 Caso de Clasificación — Detección de Fraude en Transacciones de Tarjeta de Crédito

## Descripción del Caso
Sistema de clasificación binaria para detectar transacciones fraudulentas en tarjetas de crédito. El modelo predice si una transacción es **fraudulenta (1)** o **legítima (0)** basándose en 30 características (28 componentes PCA, Time y Amount).

## Dataset
- **Fuente:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)
- **Archivo:** `data/creditcard.csv` (colocado por el estudiante)
- **Registros:** 284,807 transacciones
- **Clases:** 99.83% legítimas, 0.17% fraudulentas (desbalance severo)
- **Variables:** 28 componentes PCA (V1-V28), Time, Amount, Class

## Metodología
El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

| Fase | Descripción |
|------|-------------|
| 1. Comprensión del Negocio | Minimizar pérdidas detectando fraudes (Recall como métrica principal) |
| 2. Comprensión de los Datos | EDA, distribución de clases, correlaciones |
| 3. Preparación de los Datos | StandardScaler, SMOTE, train/test split |
| 4. Modelado | Regresión Logística, Random Forest, XGBoost, Gradient Boosting |
| 5. Evaluación | ROC-AUC, F1, Precision, Recall, Matriz de Confusión |
| 6. Despliegue | App Flask con formulario y dashboard interactivo |

## Estructura del Módulo

```
Clasificacion/
├── data/                              ← creditcard.csv (colocado por el estudiante)
├── notebook/
│   └── fraude_clasificacion.ipynb     ← Notebook completo (12 celdas, CRISP-DM)
├── paper/
│   ├── main.tex                       ← Paper LaTeX formato IEEE
│   ├── referencias.bib                ← Bibliografía
│   └── secciones/
│       ├── resumen.tex
│       ├── introduccion.tex
│       ├── metodologia.tex
│       ├── resultados.tex
│       └── conclusiones.tex
├── paper_unificado.md                 ← Versión markdown del paper
└── README.md                          ← Este archivo
```

## Instrucciones de Ejecución

### 1. Preparación
```bash
# Colocar el dataset en la carpeta data/
# Descargar de: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Archivo: Clasificacion/data/creditcard.csv
```

### 2. Ejecutar el Notebook
```bash
# Navegar a la carpeta del notebook
cd Clasificacion/notebook/

# Ejecutar el notebook (Jupyter)
jupyter notebook fraude_clasificacion.ipynb

# O ejecutar como script (opcional)
jupyter nbconvert --to notebook --execute fraude_clasificacion.ipynb
```

### 3. Ejecutar la App Flask
```bash
# Navegar a la carpeta de la aplicación
cd app/clasificacion_app/

# Instalar dependencias (si no están instaladas)
pip install flask pandas numpy plotly scikit-learn imbalanced-learn xgboost joblib

# Ejecutar la aplicación
python app.py

# Abrir en el navegador: http://127.0.0.1:5001
```

### 4. Compilar el Paper (opcional)
```bash
cd Clasificacion/paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Resultados Esperados
- **Mejor modelo:** XGBoost (generalmente) con ROC-AUC > 0.97
- **Recall para clase fraude:** > 0.80 (después de SMOTE)
- **Precision:** > 0.85 (balance con Recall)
- **Despliegue:** App funcional en puerto 5001

## Dependencias
- Python 3.12+
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn
- imbalanced-learn
- xgboost
- flask
- joblib

## Paper Unificado
Ver `Clasificacion/paper_unificado.md` para una versión en markdown del paper académico.

---

**Universidad Popular del Cesar**
*Ingeniería de Sistemas — Ciencia de Datos*
*Docente: Aimer Rivera Centeno*
