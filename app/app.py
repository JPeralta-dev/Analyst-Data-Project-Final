"""
=============================================================================
app.py — Proyecto Final Ciencia de Datos (UNIFICADO)
=============================================================================
Una sola API, un solo frontend con 3 pestañas:
  1. 🔴 Clasificación — Detección de Fraude
  2. 🎵 Agrupamiento — Spotify Clustering
  3. 🏥 Regresión — Predicción de Seguro Médico

Ejecutar:  python app.py
Puerto:    5000
=============================================================================
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # Proyecto Final/

app = Flask(__name__)

# Cache para dashboards (se generan una sola vez)
_dashboard_cache = {}


# ============================================================================
# CARGA DE MODELOS (con graceful degradation si faltan)
# ============================================================================
models = {}

# --- Agrupamiento (Spotify) ---
try:
    models['spotify_kmeans'] = joblib.load(
        os.path.join(BASE_DIR, 'agrupamiento_app', 'model', 'kmeans_spotify.pkl'))
    models['spotify_scaler'] = joblib.load(
        os.path.join(BASE_DIR, 'agrupamiento_app', 'model', 'scaler_spotify.pkl'))
    models['spotify_pca'] = joblib.load(
        os.path.join(BASE_DIR, 'agrupamiento_app', 'model', 'pca_spotify.pkl'))
    print('✓ Spotify: modelos cargados')
except Exception as e:
    models['spotify_kmeans'] = None
    print(f'✗ Spotify: modelos NO disponibles ({e})')

# --- Clasificación (Fraude) ---
try:
    models['fraude_modelo'] = joblib.load(
        os.path.join(BASE_DIR, 'clasificacion_app', 'model', 'modelo_fraude.pkl'))
    print('✓ Fraude: modelo cargado')
except Exception as e:
    models['fraude_modelo'] = None
    print(f'✗ Fraude: modelo NO disponible ({e})')

# --- Regresión (Seguro) ---
try:
    models['seguro_modelo'] = joblib.load(
        os.path.join(BASE_DIR, 'regresion_app', 'model', 'modelo_seguro.pkl'))
    models['seguro_scaler'] = joblib.load(
        os.path.join(BASE_DIR, 'regresion_app', 'model', 'scaler_seguro.pkl'))
    models['seguro_features'] = joblib.load(
        os.path.join(BASE_DIR, 'regresion_app', 'model', 'feature_names.pkl'))
    print('✓ Seguro: modelos cargados')
except Exception as e:
    models['seguro_modelo'] = None
    print(f'✗ Seguro: modelos NO disponibles ({e})')


# ============================================================================
# CARGA DE DATASETS (lazy, solo cuando se necesiten)
# ============================================================================
def _load_dataset(name):
    """Carga un dataset bajo demanda."""
    paths = {
        'spotify': os.path.join(PROJECT_DIR, 'Agrupamiento', 'Data', 'dataset.csv'),
        'fraude': os.path.join(PROJECT_DIR, 'Clasificacion', 'Data', 'creditcard.csv'),
        'seguro': os.path.join(PROJECT_DIR, 'Regresion', 'Data', 'insurance.csv'),
    }
    path = paths.get(name)
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ============================================================================
# PÁGINA PRINCIPAL
# ============================================================================
@app.route('/')
def index():
    """Renderiza la página unificada con 3 pestañas."""
    return render_template('index.html',
        spotify_ready=models['spotify_kmeans'] is not None,
        fraude_ready=models['fraude_modelo'] is not None,
        seguro_ready=models['seguro_modelo'] is not None)


# ============================================================================
# API: PREDICCIÓN — SPOTIFY CLUSTERING
# ============================================================================
@app.route('/api/predict/spotify', methods=['POST'])
def predict_spotify():
    data = request.get_json()
    kmeans = models['spotify_kmeans']
    scaler = models['spotify_scaler']

    if kmeans is None:
        return jsonify({'success': False, 'error': 'Modelo no disponible. Ejecutá el notebook primero.'})

    features_order = ['danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness',
                      'valence', 'tempo', 'popularity']

    try:
        X = np.array([[float(data.get(f, 0)) for f in features_order]])
        X_scaled = scaler.transform(X)
        cluster_id = int(kmeans.predict(X_scaled)[0])
        distancias = np.linalg.norm(X_scaled - kmeans.cluster_centers_, axis=1)

        nombres = {
            0: 'Energía y En Vivo', 1: 'Acústico y Relajado',
            2: 'Bailable y Positivo', 3: 'Clásico e Instrumental',
            4: 'Instrumental Enérgico', 5: 'Hablado / Podcast'
        }
        descripciones = {
            0: 'Alta energy (0.82) y liveness. Electrónica enérgica, rock en vivo, EDM.',
            1: 'Alta acousticness (0.66), baja energy (0.39). Folk, indie acústico, baladas.',
            2: 'Alta danceability (0.70), energy (0.73) y valence (0.69). Pop bailable, reggaetón, música alegre.',
            3: 'Muy alta acousticness (0.86) e instrumentalness (0.77), baja energy (0.17). Música clásica, bandas sonoras, ambient.',
            4: 'Alta instrumentalness (0.79) con energy (0.75). Jazz instrumental, post-rock, electrónica sin voz.',
            5: 'Muy alta speechiness (0.79) y liveness (0.62). Podcasts, rap, spoken word, hip-hop en vivo.'
        }

        return jsonify({
            'success': True,
            'cluster_id': cluster_id,
            'cluster_nombre': nombres.get(cluster_id, f'Cluster {cluster_id}'),
            'descripcion': descripciones.get(cluster_id, ''),
            'distancia_centroide': round(float(distancias[cluster_id]), 3)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API: PREDICCIÓN — FRAUDE
# ============================================================================
@app.route('/api/predict/fraude', methods=['POST'])
def predict_fraude():
    data = request.get_json()
    modelo = models['fraude_modelo']

    if modelo is None:
        return jsonify({'success': False, 'error': 'Modelo no disponible. Ejecutá el notebook primero.'})

    features = [f'V{i}' for i in range(1, 29)] + ['Time_scaled', 'Amount_scaled']

    try:
        valores = []
        for feat in features:
            if feat in ('Time_scaled', 'Amount_scaled'):
                raw = feat.replace('_scaled', '')
                valores.append(float(data.get(raw, 0)))
            else:
                valores.append(float(data.get(feat, 0)))

        X_pred = np.array([valores])
        pred = int(modelo.predict(X_pred)[0])
        prob = float(modelo.predict_proba(X_pred)[0][1])

        return jsonify({
            'success': True,
            'prediccion': 'FRAUDE' if pred == 1 else 'LEGÍTIMA',
            'probabilidad': round(prob, 4),
            'mensaje': '⚠️ Alta probabilidad de fraude' if pred == 1 else '✅ Transacción normal'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API: PREDICCIÓN — SEGURO MÉDICO
# ============================================================================
@app.route('/api/predict/seguro', methods=['POST'])
def predict_seguro():
    data = request.get_json()
    modelo = models['seguro_modelo']
    scaler = models['seguro_scaler']
    feature_names = models['seguro_features']

    if modelo is None:
        return jsonify({'success': False, 'error': 'Modelo no disponible. Ejecutá el notebook primero.'})

    try:
        smoker_enc = 1 if data.get('smoker') == 'yes' else 0
        sex_enc = 1 if data.get('sex') == 'male' else 0
        region = data.get('region', 'southeast')
        age = float(data.get('age', 35))
        bmi = float(data.get('bmi', 28))
        children = float(data.get('children', 0))

        vector = {
            'age': age, 'bmi': bmi, 'children': children,
            'smoker_enc': smoker_enc, 'sex_enc': sex_enc,
            'region_northwest': 1 if region == 'northwest' else 0,
            'region_southeast': 1 if region == 'southeast' else 0,
            'region_southwest': 1 if region == 'southwest' else 0,
            'bmi_smoker': bmi * smoker_enc, 'age_sq': age ** 2
        }

        df_input = pd.DataFrame([vector])
        df_input = df_input[feature_names]
        X_scaled = scaler.transform(df_input)
        pred = float(modelo.predict(X_scaled)[0])

        return jsonify({
            'success': True,
            'prediccion': round(pred, 2),
            'modelo': type(modelo).__name__
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API: DASHBOARDS (lazy-loading con cache)
# ============================================================================
@app.route('/api/dashboard/<case>')
def dashboard(case):
    """Genera gráficas Plotly para el dashboard de cada caso y las devuelve como JSON."""
    if case in _dashboard_cache:
        return jsonify(_dashboard_cache[case])

    try:
        if case == 'spotify':
            graphs = _dashboard_spotify()
        elif case == 'fraude':
            graphs = _dashboard_fraude()
        elif case == 'seguro':
            graphs = _dashboard_seguro()
        else:
            return jsonify({'error': 'Caso inválido'})

        _dashboard_cache[case] = graphs
        return jsonify(graphs)
    except Exception as e:
        return jsonify({'error': str(e)})


def _dashboard_spotify():
    df = _load_dataset('spotify')
    if df is None:
        return []

    FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']
    scaler = models['spotify_scaler']
    kmeans = models['spotify_kmeans']
    pca = models['spotify_pca']

    graphs = []

    # 1. PCA scatter
    muestra = df.sample(min(3000, len(df)), random_state=42)
    X_m = scaler.transform(muestra[FEATURES].dropna())
    X_pca = pca.transform(X_m)
    clusters = kmeans.predict(X_m)
    var = pca.explained_variance_ratio_ * 100

    fig1 = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=[f'Cluster {c}' for c in clusters],
        title=f'Clusters Spotify — PCA 2D ({var[0]:.1f}% + {var[1]:.1f}% varianza)',
        opacity=0.6, color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig1.update_layout(xaxis_title=f'PC1 ({var[0]:.1f}%)', yaxis_title=f'PC2 ({var[1]:.1f}%)',
                       width=650, height=420, margin=dict(l=20, r=20, t=50, b=20))
    graphs.append(json.loads(fig1.to_json()))

    # 2. Distribución de features
    fig2 = go.Figure()
    for feat in ['danceability', 'energy', 'acousticness', 'valence']:
        fig2.add_trace(go.Histogram(x=df[feat].dropna(), name=feat.capitalize(),
                                    opacity=0.7, nbinsx=35))
    fig2.update_layout(title='Distribución de Características de Audio', barmode='overlay',
                       width=650, height=420, margin=dict(l=20, r=20, t=50, b=20),
                       xaxis_title='Valor', yaxis_title='Frecuencia')
    graphs.append(json.loads(fig2.to_json()))

    # 3. Géneros top
    top_genres = df['track_genre'].value_counts().head(12)
    fig3 = px.bar(x=top_genres.values, y=top_genres.index, orientation='h',
                  title='Top 12 Géneros en el Dataset',
                  labels={'x': 'Canciones', 'y': 'Género'},
                  color=top_genres.values, color_continuous_scale='Viridis')
    fig3.update_layout(width=650, height=420, margin=dict(l=20, r=20, t=50, b=20))
    graphs.append(json.loads(fig3.to_json()))

    return graphs


def _dashboard_fraude():
    df = _load_dataset('fraude')
    if df is None:
        return []

    graphs = []

    # 1. Distribución de clases
    counts = df['Class'].value_counts()
    fig1 = px.pie(values=counts.values, names=['Legítima', 'Fraude'],
                  title='Distribución de Clases (0.17% fraude)',
                  color_discrete_sequence=['#2196F3', '#F44336'])
    fig1.update_layout(width=500, height=380, margin=dict(l=20, r=20, t=50, b=20))
    graphs.append(json.loads(fig1.to_json()))

    # 2. Amount por clase
    fig2 = go.Figure()
    for cls, name, color in [(0, 'Legítima', '#2196F3'), (1, 'Fraude', '#F44336')]:
        fig2.add_trace(go.Histogram(
            x=df[df['Class'] == cls]['Amount'], name=name,
            opacity=0.7, nbinsx=50, marker_color=color))
    fig2.update_layout(title='Distribución del Monto por Clase', barmode='overlay',
                       yaxis_type='log', xaxis_title='Amount (€)',
                       width=650, height=420, margin=dict(l=20, r=20, t=50, b=20))
    graphs.append(json.loads(fig2.to_json()))

    # 3. Correlaciones top con Class
    corr = df.corr()['Class'].abs().sort_values(ascending=False).head(11).drop('Class')
    fig3 = px.bar(x=corr.values, y=corr.index, orientation='h',
                  title='Top 10 Variables Correlacionadas con Fraude',
                  labels={'x': '|Correlación|', 'y': 'Variable'},
                  color=corr.values, color_continuous_scale='Reds')
    fig3.update_layout(width=650, height=420, margin=dict(l=20, r=20, t=50, b=20))
    graphs.append(json.loads(fig3.to_json()))

    return graphs


def _dashboard_seguro():
    df = _load_dataset('seguro')
    if df is None:
        return []

    graphs = []

    # 1. Distribución de charges
    fig1 = px.histogram(df, x='charges', nbins=40, title='Distribución de Costos de Seguro',
                        color_discrete_sequence=['#4CAF50'], marginal='box')
    fig1.add_vline(x=df['charges'].mean(), line_dash='dash', line_color='red',
                   annotation_text=f"Media: ${df['charges'].mean():,.0f}")
    fig1.update_layout(width=650, height=420, margin=dict(l=20, r=20, t=50, b=20),
                       xaxis_title='Costo (USD)', yaxis_title='Frecuencia')
    graphs.append(json.loads(fig1.to_json()))

    # 2. Fumadores vs No fumadores
    fig2 = px.box(df, x='smoker', y='charges', color='smoker',
                  title='Costo del Seguro: Fumadores vs No Fumadores',
                  color_discrete_map={'yes': '#F44336', 'no': '#2196F3'}, points='all')
    fig2.update_layout(width=650, height=420, margin=dict(l=20, r=20, t=50, b=20),
                       xaxis_title='', yaxis_title='Costo (USD)')
    graphs.append(json.loads(fig2.to_json()))

    # 3. Real vs Predicho (sample)
    feature_names = models['seguro_features']
    scaler = models['seguro_scaler']
    modelo = models['seguro_modelo']

    if modelo is not None and hasattr(modelo, 'predict'):
        df_plot = df.copy()
        df_plot['smoker_enc'] = (df_plot['smoker'] == 'yes').astype(int)
        df_plot['sex_enc'] = (df_plot['sex'] == 'male').astype(int)
        df_plot = pd.get_dummies(df_plot, columns=['region'], prefix='region', drop_first=True)
        df_plot['bmi_smoker'] = df_plot['bmi'] * df_plot['smoker_enc']
        df_plot['age_sq'] = df_plot['age'] ** 2

        X_plot = df_plot[feature_names]
        X_sc = scaler.transform(X_plot)
        y_pred = modelo.predict(X_sc)

        fig3 = px.scatter(x=df['charges'], y=y_pred, opacity=0.5,
                          title='Real vs Predicho — Modelo de Regresión',
                          labels={'x': 'Costo Real (USD)', 'y': 'Costo Predicho (USD)'})
        mn, mx = min(df['charges'].min(), y_pred.min()), max(df['charges'].max(), y_pred.max())
        fig3.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                       line=dict(color='green', dash='dash', width=2))
        fig3.update_layout(width=650, height=420, margin=dict(l=20, r=20, t=50, b=20))
        graphs.append(json.loads(fig3.to_json()))

    return graphs


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  🚀  Proyecto Final — Ciencia de Datos (App Unificada)')
    print('=' * 60)
    print(f'  🎵 Spotify:    {"✅ Listo" if models["spotify_kmeans"] else "⚠️  Falta modelo"}')
    print(f'  🔴 Fraude:     {"✅ Listo" if models["fraude_modelo"] else "⚠️  Falta modelo"}')
    print(f'  🏥 Seguro:     {"✅ Listo" if models["seguro_modelo"] else "⚠️  Falta modelo"}')
    print(f'  🔗 http://localhost:5000')
    print('=' * 60 + '\n')
    app.run(debug=True, port=5000)
