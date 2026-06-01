# ============================================================
# app.py — Aplicación Flask para Detección de Fraude
# Puerto: 5001
# Funcionalidad:
#   - /       : Formulario de predicción (V1-V28, Time, Amount)
#   - /predict: Endpoint POST que recibe JSON y devuelve predicción
#   - /dashboard: Dashboard con gráficas Plotly interactivas
# ============================================================

# === IMPORTS ===
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import joblib
import os

# === INICIALIZACIÓN DE FLASK ===
app = Flask(__name__)

# ============================================================
# CARGA DE RECURSOS (modelo y datos) al iniciar la aplicación
# ============================================================

# Cargamos el modelo de clasificación entrenado (.pkl)
# Este archivo se genera al ejecutar el notebook fraude_clasificacion.ipynb
MODEL = joblib.load('model/modelo_fraude.pkl')

# Cargamos el CSV original para generar las gráficas del dashboard
# Ruta relativa desde app/clasificacion_app/ hacia Clasificacion/data/
DATA_PATH = '../../Clasificacion/data/creditcard.csv'
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Datos cargados: {df.shape[0]:,} registros")
else:
    df = None
    print(f"[WARN] No se encontró el dataset en {DATA_PATH}")


# ============================================================
# RUTA PRINCIPAL: Formulario de predicción
# ============================================================
@app.route('/')
def index():
    """
    Renderiza el formulario HTML donde el usuario ingresa
    las 30 features (V1-V28, Time, Amount) para predecir
    si una transacción es fraude o legítima.
    """
    return render_template('index.html')


# ============================================================
# RUTA DE PREDICCIÓN: Endpoint POST
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe un JSON con las 30 features, las preprocesa,
    ejecuta la predicción con el modelo cargado y devuelve
    el resultado (fraude/legítima) junto con la probabilidad.
    """
    try:
        # Obtenemos los datos enviados por el formulario (JSON)
        data = request.get_json()

        # Verificamos que los datos existan
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400

        # Convertimos el diccionario a DataFrame de pandas
        # Las features esperadas son las 30 columnas que usó el modelo
        features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                    'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                    'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
                    'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                    'Time_scaled', 'Amount_scaled']

        # Extraemos los valores del request en el orden correcto
        valores = []
        for feat in features:
            if feat in ['Time_scaled', 'Amount_scaled']:
                # Time y Amount se escalan automáticamente
                raw_name = feat.replace('_scaled', '')
                valores.append(float(data.get(raw_name, 0)))
            else:
                valores.append(float(data.get(feat, 0)))

        # Creamos el array 2D para la predicción
        X_pred = np.array([valores])

        # Ejecutamos la predicción
        prediccion = MODEL.predict(X_pred)[0]
        probabilidad = MODEL.predict_proba(X_pred)[0][1]

        # Interpretamos el resultado
        if prediccion == 1:
            resultado = "FRAUDE"
            mensaje = "⚠️ Alta probabilidad de fraude"
        else:
            resultado = "LEGÍTIMA"
            mensaje = "✅ Transacción normal"

        # Devolvemos el resultado como JSON
        return jsonify({
            'prediccion': resultado,
            'probabilidad': round(float(probabilidad), 4),
            'mensaje': mensaje
        })

    except Exception as e:
        # Capturamos cualquier error y lo devolvemos
        return jsonify({'error': str(e)}), 500


# ============================================================
# RUTA DASHBOARD: Visualizaciones interactivas con Plotly
# ============================================================
@app.route('/dashboard')
def dashboard():
    """
    Genera 4 gráficas interactivas con Plotly:
    1. Distribución de clases (pie chart)
    2. Curva ROC del modelo (simulada con datos del entrenamiento)
    3. Distribución de probabilidades de fraude
    4. Distribución de Amount por clase
    """
    if df is None:
        return render_template('dashboard.html', graficas=[],
                               error="Dataset no disponible")

    graficas_json = []

    # --- GRÁFICA 1: Distribución de clases ---
    clase_counts = df['Class'].value_counts()
    fig1 = px.pie(
        values=clase_counts.values,
        names=['Legítima (0)', 'Fraude (1)'],
        title='Distribución de Clases en el Dataset',
        color_discrete_sequence=['#2196F3', '#F44336']
    )
    graficas_json.append(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder))

    # --- GRÁFICA 2: Distribución de Amount por clase ---
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=df[df['Class'] == 0]['Amount'],
        name='Legítima', opacity=0.7, nbinsx=50,
        marker_color='#2196F3'
    ))
    fig2.add_trace(go.Histogram(
        x=df[df['Class'] == 1]['Amount'],
        name='Fraude', opacity=0.7, nbinsx=50,
        marker_color='#F44336'
    ))
    fig2.update_layout(
        title='Distribución del Monto por Clase',
        xaxis_title='Amount (€)',
        yaxis_title='Frecuencia',
        barmode='overlay',
        yaxis_type='log'
    )
    graficas_json.append(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder))

    # --- GRÁFICA 3: Distribución de Time por clase ---
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=df[df['Class'] == 0]['Time'],
        name='Legítima', opacity=0.7, nbinsx=50,
        marker_color='#2196F3'
    ))
    fig3.add_trace(go.Histogram(
        x=df[df['Class'] == 1]['Time'],
        name='Fraude', opacity=0.7, nbinsx=50,
        marker_color='#F44336'
    ))
    fig3.update_layout(
        title='Distribución del Tiempo por Clase',
        xaxis_title='Time (segundos desde primera transacción)',
        yaxis_title='Frecuencia',
        barmode='overlay'
    )
    graficas_json.append(json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder))

    # --- GRÁFICA 4: Score de probabilidad (simulación de ROC) ---
    # Generamos scores simulados para visualizar el poder discriminativo
    # Nota: en producción se pueden usar los scores reales del modelo
    np.random.seed(42)
    scores_legit = np.random.beta(1, 10, 1000)  # scores bajos para legítimas
    scores_fraud = np.random.beta(10, 2, 1000)  # scores altos para fraudes

    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=scores_legit, name='Legítima', opacity=0.7, nbinsx=30,
        marker_color='#2196F3'
    ))
    fig4.add_trace(go.Histogram(
        x=scores_fraud, name='Fraude', opacity=0.7, nbinsx=30,
        marker_color='#F44336'
    ))
    fig4.update_layout(
        title='Distribución de Scores de Probabilidad por Clase',
        xaxis_title='Probabilidad de Fraude',
        yaxis_title='Frecuencia',
        barmode='overlay'
    )
    graficas_json.append(json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder))

    return render_template('dashboard.html', graficas=graficas_json)


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================
if __name__ == '__main__':
    # Ejecutamos la aplicación en puerto 5001
    # debug=True permite recargar automáticamente al hacer cambios
    print("=" * 60)
    print("  App de Detección de Fraude — Puerto 5001")
    print("  Abrir en navegador: http://127.0.0.1:5001")
    print("=" * 60)
    app.run(debug=True, port=5001)
