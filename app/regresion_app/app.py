# =============================================================================
# app/regresion_app/app.py — Aplicación Flask para Predicción de Seguro Médico
# =============================================================================
# Esta aplicación web carga el modelo entrenado (modelo_seguro.pkl),
# el scaler (scaler_seguro.pkl) y los nombres de features (feature_names.pkl)
# para servir predicciones de costos de seguro médico.
#
# Endpoints:
#   /          → Formulario HTML para ingresar datos del paciente
#   /predict   → API JSON que recibe datos y devuelve el costo estimado
#   /dashboard → Dashboard con gráficas Plotly interactivas
#
# Puerto: 5003

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import joblib
import os

# =============================================================================
# Configuración de rutas (relativas a este archivo)
# =============================================================================
# Obtenemos el directorio base de la aplicación para construir rutas absolutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas a los artefactos del modelo (generados por el notebook)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'modelo_seguro.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler_seguro.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'feature_names.pkl')

# Ruta al dataset original (para generar las gráficas del dashboard)
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'Regresion', 'data', 'insurance.csv')

app = Flask(__name__)

# =============================================================================
# Carga de artefactos al iniciar la aplicación
# =============================================================================
# Cargamos el modelo, el scaler y los nombres de features una sola vez
# (al arrancar la app) para evitar recargarlos en cada petición.
print("Cargando modelo de regresión...")
modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
print(f"✓ Modelo cargado. Features esperadas ({len(feature_names)}): {feature_names}")

# Cargamos el dataset para las visualizaciones del dashboard
print("Cargando dataset para dashboard...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset cargado: {df.shape[0]} registros")


# =============================================================================
# Funciones auxiliares de preprocesamiento
# =============================================================================

def preprocesar_entrada(data):
    """
    Transforma los datos del formulario en el mismo formato de features
    que usó el modelo durante el entrenamiento.

    El orden de las features debe ser IDÉNTICO al del entrenamiento:
    age, bmi, children, smoker_enc, sex_enc, region_northwest,
    region_southeast, region_southwest, bmi_smoker, age_sq

    Args:
        data: Diccionario con los valores del formulario
              (age, sex, bmi, children, smoker, region)

    Returns:
        DataFrame con 1 fila y las mismas columnas que X_train
    """
    # Convertimos las variables categóricas a numéricas
    smoker_enc = 1 if data['smoker'] == 'yes' else 0
    sex_enc = 1 if data['sex'] == 'male' else 0

    # One-hot encoding para región (mismo esquema que en el notebook)
    region = data['region']
    # region_northwest, region_southeast, region_southwest (southeast es referencia)
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0

    # Feature engineering: mismo que en el notebook
    age = float(data['age'])
    bmi = float(data['bmi'])
    children = float(data['children'])

    bmi_smoker = bmi * smoker_enc   # Interacción BMI × tabaquismo
    age_sq = age ** 2                # Término cuadrático de edad

    # Construimos el vector de features en el ORDEN EXACTO del entrenamiento
    # Este orden debe coincidir con feature_names.pkl
    vector = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker_enc': smoker_enc,
        'sex_enc': sex_enc,
        'region_northwest': region_northwest,
        'region_southeast': region_southeast,
        'region_southwest': region_southwest,
        'bmi_smoker': bmi_smoker,
        'age_sq': age_sq
    }

    # Creamos DataFrame en el orden correcto de features
    df_entrada = pd.DataFrame([vector])
    df_entrada = df_entrada[feature_names]  # Reordenar según el orden del modelo

    # Escalamos usando el scaler entrenado (mismo fit del notebook)
    df_scaled = scaler.transform(df_entrada)

    return df_scaled


# =============================================================================
# Rutas
# =============================================================================

@app.route('/')
def index():
    """
    Página principal: formulario para ingresar los datos del paciente.
    Renderiza index.html con las opciones de región y sexo.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de predicción.
    Recibe JSON con los datos del formulario, preprocesa,
    ejecuta el modelo y devuelve el costo estimado.

    Ejemplo de request:
    {
        "age": 35,
        "sex": "male",
        "bmi": 28.5,
        "children": 2,
        "smoker": "no",
        "region": "southeast"
    }

    Ejemplo de response:
    {
        "success": true,
        "prediccion": 12500.45,
        "modelo": "XGBoost"
    }
    """
    try:
        # Recibimos los datos en formato JSON
        data = request.get_json()

        # Preprocesamos: aplicamos el mismo pipeline que en el notebook
        X_input = preprocesar_entrada(data)

        # Predecimos: ejecutamos el modelo cargado
        prediccion = modelo.predict(X_input)[0]

        # Devolvemos el resultado en formato JSON
        return jsonify({
            'success': True,
            'prediccion': round(float(prediccion), 2),
            'modelo': type(modelo).__name__
        })

    except Exception as e:
        # Si algo falla, devolvemos el error para depuración
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/dashboard')
def dashboard():
    """
    Dashboard con 4 gráficas Plotly interactivas:
    1. Real vs Predicho (scatter plot con línea diagonal)
    2. Importancia de Features (bar chart horizontal)
    3. Distribución de Charges (histograma original + log)
    4. Charges por Fumador (boxplot interactivo)

    Las gráficas se renderizan en el template dashboard.html
    usando Plotly.js en el lado del cliente.
    """
    # =========================================================================
    # Gráfica 1: Distribución de Charges (costo del seguro)
    # Histograma interactivo con la distribución de la variable objetivo
    # =========================================================================
    fig1 = px.histogram(
        df, x='charges', nbins=50,
        title='Distribución de Costos de Seguro Médico',
        labels={'charges': 'Costo (USD)'},
        color_discrete_sequence=['#2196F3'],
        marginal='box'  # Añade un boxplot en la parte superior
    )
    fig1.add_vline(
        x=df['charges'].mean(),
        line_dash='dash',
        line_color='red',
        annotation_text=f"Media: ${df['charges'].mean():,.0f}"
    )
    fig1.update_layout(
        xaxis_title='Costo del Seguro (USD)',
        yaxis_title='Frecuencia',
        showlegend=False
    )

    # =========================================================================
    # Gráfica 2: Charges por Fumador (boxplot)
    # Muestra la diferencia dramática entre fumadores y no fumadores
    # =========================================================================
    fig2 = px.box(
        df, x='smoker', y='charges', color='smoker',
        title='Costo del Seguro: Fumadores vs No Fumadores',
        labels={'smoker': 'Fumador', 'charges': 'Costo (USD)'},
        color_discrete_map={'yes': '#F44336', 'no': '#2196F3'},
        points='all'  # Muestra todos los puntos, no solo el boxplot
    )
    fig2.update_layout(
        xaxis_title='',
        yaxis_title='Costo del Seguro (USD)'
    )

    # =========================================================================
    # Gráfica 3: Real vs Predicho
    # Simula predicciones sobre el dataset para mostrar la precisión del modelo
    # =========================================================================
    # Preparamos los datos igual que en el notebook
    df_plot = df.copy()
    df_plot['smoker_enc'] = (df_plot['smoker'] == 'yes').astype(int)
    df_plot['sex_enc'] = (df_plot['sex'] == 'male').astype(int)
    df_plot = pd.get_dummies(df_plot, columns=['region'], prefix='region', drop_first=True)
    df_plot['bmi_smoker'] = df_plot['bmi'] * df_plot['smoker_enc']
    df_plot['age_sq'] = df_plot['age'] ** 2

    X_plot = df_plot[feature_names]

    # Escalamos y predecimos
    X_plot_sc = scaler.transform(X_plot)
    y_pred_all = modelo.predict(X_plot_sc)

    # Scatter plot: real vs predicho
    fig3 = px.scatter(
        x=df['charges'], y=y_pred_all,
        title='Real vs Predicho — Precisión del Modelo',
        labels={'x': 'Costo Real (USD)', 'y': 'Costo Predicho (USD)'},
        opacity=0.5,
        color=df['smoker'],
        color_discrete_map={'yes': '#F44336', 'no': '#2196F3'}
    )
    # Línea diagonal (predicción perfecta)
    min_val = min(df['charges'].min(), y_pred_all.min())
    max_val = max(df['charges'].max(), y_pred_all.max())
    fig3.add_shape(
        type='line', x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color='green', dash='dash', width=2)
    )
    fig3.update_layout(
        xaxis_title='Costo Real (USD)',
        yaxis_title='Costo Predicho (USD)'
    )

    # =========================================================================
    # Gráfica 4: Importancia de Features
    # Solo se muestra si el modelo tiene el atributo feature_importances_
    # =========================================================================
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=True)  # ascendente para barra horizontal

        fig4 = px.bar(
            fi_df, x='importance', y='feature', orientation='h',
            title='Importancia de Variables Predictoras',
            labels={'importance': 'Importancia relativa', 'feature': 'Variable'},
            color='importance',
            color_continuous_scale='Viridis',
            text_auto='.2%'
        )
        fig4.update_layout(yaxis=dict( autorange='reversed'))  # mayor importancia arriba
    else:
        # Si el modelo no tiene feature_importances_ (ej: Ridge lineal),
        # mostramos un placeholder
        fig4 = go.Figure()
        fig4.add_annotation(
            text="Feature importance no disponible para este modelo lineal",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig4.update_layout(title='Importancia de Variables')

    # =========================================================================
    # Convertimos las gráficas a JSON para Plotly.js
    # =========================================================================
    graficas_json = [
        json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
        json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    ]

    return render_template('dashboard.html', graficas=graficas_json)


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("🏥 App de Predicción de Seguro Médico")
    print(f"📊 Modelo: {type(modelo).__name__}")
    print(f"🔗 Puerto: 5003")
    print("=" * 50)
    app.run(debug=True, port=5003)
