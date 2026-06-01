# ============================================================================
# APP FLASK — Agrupamiento Spotify
# ============================================================================
# Aplicación web para predicción de clusters musicales usando K-Means.
# Permite al usuario ingresar características de audio mediante sliders
# y visualizar a qué cluster pertenece la canción.
#
# Puerto: 5002
# Stack: Flask + Plotly + joblib + scikit-learn
# ============================================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import joblib
import json
import os

# ---------------------------------------------------------------------------
# Inicialización de la aplicación Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Carga de modelos al iniciar la aplicación
# ---------------------------------------------------------------------------
# Determinamos la ruta base donde están los modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Cargamos los 3 artefactos guardados desde el notebook
kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans_spotify.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_spotify.pkl'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca_spotify.pkl'))

print('✓ Modelos cargados exitosamente')
print(f'  - K-Means: {kmeans.n_clusters} clusters')
print(f'  - Scaler: {scaler.mean_.shape[0]} features')
print(f'  - PCA: {pca.n_components_} componentes')

# ---------------------------------------------------------------------------
# Carga del dataset para las gráficas del dashboard
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'Agrupamiento', 'data', 'dataset.csv')
df = pd.read_csv(DATA_PATH)

# Características numéricas usadas en el clustering
FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'popularity']

# Nombres descriptivos para cada cluster (ajustar según perfil real)
NOMBRES_CLUSTER = {
    0: 'Cluster 0 — Energía y Baile',
    1: 'Cluster 1 — Acústico y Relajado',
    2: 'Cluster 2 — Instrumental',
    3: 'Cluster 3 — Hablado / Podcast',
    4: 'Cluster 4 — En Vivo',
    5: 'Cluster 5 — Melódico y Positivo'
}

# Descripciones detalladas para mostrar en la predicción
DESCRIPCIONES_CLUSTER = {
    0: 'Canciones con alta danceability y energy. Ideales para fiesta, electrónica y pop bailable.',
    1: 'Canciones acústicas con baja energy. Música relajada, folk o baladas.',
    2: 'Canciones con alta instrumentalness. Bandas sonoras, música clásica o experimental.',
    3: 'Canciones con alta speechiness. Podcasts, rap o contenido hablado.',
    4: 'Canciones con alta liveness. Grabaciones en vivo o conciertos.',
    5: 'Canciones con alta valence y danceability. Música positiva y alegre.'
}


# ============================================================================
# RUTA PRINCIPAL — Formulario de predicción
# ============================================================================
@app.route('/')
def index():
    """
    Renderiza la página principal con el formulario de sliders
    para que el usuario ingrese las características de audio.
    """
    return render_template('index.html')


# ============================================================================
# RUTA DE PREDICCIÓN — Endpoint POST que recibe JSON y devuelve el cluster
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe un JSON con las 10 features de audio, las escala con el
    StandardScaler y predice el cluster usando K-Means.
    
    Retorna:
        - cluster_id: ID numérico del cluster
        - cluster_nombre: Nombre descriptivo del cluster
        - descripcion: Descripción del perfil musical
    """
    try:
        # Recibimos los datos del formulario en formato JSON
        data = request.get_json()
        
        # Extraemos las features en el orden correcto
        features = [
            float(data['danceability']),
            float(data['energy']),
            float(data['loudness']),
            float(data['speechiness']),
            float(data['acousticness']),
            float(data['instrumentalness']),
            float(data['liveness']),
            float(data['valence']),
            float(data['tempo']),
            float(data['popularity'])
        ]
        
        # Convertimos a array numpy y escalamos
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Predecimos el cluster
        cluster_id = int(kmeans.predict(X_scaled)[0])
        
        # Calculamos la distancia al centroide (qué tan típico es del cluster)
        distancias = np.linalg.norm(X_scaled - kmeans.cluster_centers_, axis=1)
        distancia_centroide = float(distancias[cluster_id].round(2))
        
        return jsonify({
            'success': True,
            'cluster_id': cluster_id,
            'cluster_nombre': NOMBRES_CLUSTER.get(cluster_id, f'Cluster {cluster_id}'),
            'descripcion': DESCRIPCIONES_CLUSTER.get(cluster_id, ''),
            'distancia_centroide': distancia_centroide,
            'n_clusters': kmeans.n_clusters
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


# ============================================================================
# RUTA DASHBOARD — Visualizaciones interactivas con Plotly
# ============================================================================
@app.route('/dashboard')
def dashboard():
    """
    Genera un dashboard con 3 gráficas Plotly:
    1. Scatter PCA 2D con los clusters coloreados
    2. Radar Chart con el perfil promedio de cada cluster
    3. Distribución de las features de audio
    """
    
    # -----------------------------------------------------------------------
    # Gráfica 1: Scatter PCA 2D
    # -----------------------------------------------------------------------
    # Tomamos una muestra para la visualización (rendimiento)
    muestra_pca = df.sample(min(5000, len(df)), random_state=42)
    X_muestra = scaler.transform(muestra_pca[FEATURES].dropna())
    X_pca = pca.transform(X_muestra)
    clusters_muestra = kmeans.predict(X_muestra)
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [NOMBRES_CLUSTER.get(c, f'Cluster {c}') for c in clusters_muestra]
    })
    
    fig_pca = px.scatter(
        df_pca, x='PC1', y='PC2', color='Cluster',
        title=f'Clusters Spotify — Proyección PCA 2D ({var_pc1:.1f}% + {var_pc2:.1f}% varianza)',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set1,
        width=700, height=500
    )
    fig_pca.update_layout(
        xaxis_title=f'Componente Principal 1 ({var_pc1:.1f}%)',
        yaxis_title=f'Componente Principal 2 ({var_pc2:.1f}%)'
    )
    grafica_pca = json.dumps(fig_pca, cls=plotly.utils.PlotlyJSONEncoder)
    
    # -----------------------------------------------------------------------
    # Gráfica 2: Radar Chart — Perfil promedio por cluster
    # -----------------------------------------------------------------------
    # Calculamos el perfil promedio para las features en escala 0-1
    features_radar = ['danceability', 'energy', 'speechiness',
                      'acousticness', 'liveness', 'valence']
    
    # Predecimos clusters para todo el dataset (o una muestra)
    X_all = scaler.transform(df[FEATURES].dropna())
    all_clusters = kmeans.predict(X_all)
    
    df_temp = df[FEATURES + ['track_genre']].dropna().copy()
    df_temp['cluster'] = all_clusters
    
    perfil = df_temp.groupby('cluster')[features_radar].mean()
    
    fig_radar = go.Figure()
    for cluster_id in range(kmeans.n_clusters):
        valores = perfil.loc[cluster_id, features_radar].tolist()
        valores += [valores[0]]  # cerramos el polígono
        fig_radar.add_trace(go.Scatterpolar(
            r=valores,
            theta=features_radar + [features_radar[0]],
            fill='toself',
            name=NOMBRES_CLUSTER.get(cluster_id, f'Cluster {cluster_id}')
        ))
    
    fig_radar.update_layout(
        title='Perfil Promedio de Audio por Cluster',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        width=700, height=500
    )
    grafica_radar = json.dumps(fig_radar, cls=plotly.utils.PlotlyJSONEncoder)
    
    # -----------------------------------------------------------------------
    # Gráfica 3: Distribución de Features de Audio
    # -----------------------------------------------------------------------
    fig_dist = go.Figure()
    for feat in ['danceability', 'energy', 'acousticness', 'valence']:
        fig_dist.add_trace(go.Histogram(
            x=df[feat].dropna(),
            name=feat.capitalize(),
            opacity=0.7,
            nbinsx=40
        ))
    
    fig_dist.update_layout(
        title='Distribución de Características de Audio',
        xaxis_title='Valor',
        yaxis_title='Frecuencia',
        barmode='overlay',
        width=700, height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    grafica_dist = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(
        'dashboard.html',
        grafica_pca=grafica_pca,
        grafica_radar=grafica_radar,
        grafica_dist=grafica_dist
    )


# ============================================================================
# Punto de entrada
# ============================================================================
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  🎵  App Agrupamiento Spotify — Puerto 5002')
    print('=' * 60)
    print('  Rutas disponibles:')
    print('    http://localhost:5002/           → Formulario de predicción')
    print('    http://localhost:5002/dashboard   → Dashboard interactivo')
    print('    http://localhost:5002/predict     → API JSON (POST)')
    print('=' * 60 + '\n')
    app.run(debug=True, port=5002)
