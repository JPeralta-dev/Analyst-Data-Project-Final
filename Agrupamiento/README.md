# 🎵 Agrupamiento Spotify — Clustering de Canciones

Módulo de **Agrupamiento (Clustering)** para el Proyecto Final de Ciencia de Datos.
Aplica K-Means sobre el Spotify Tracks Dataset para descubrir patrones musicales
basados en características de audio.

## 📁 Estructura del Módulo

```
Agrupamiento/
├── data/
│   └── dataset.csv              ← Dataset (colocar manualmente)
├── notebook/
│   └── spotify_clustering.ipynb  ← Notebook Jupyter (CRISP-DM)
├── paper/
│   ├── main.tex                  ← Documento LaTeX principal
│   ├── referencias.bib           ← Bibliografía
│   ├── secciones/
│   │   ├── resumen.tex
│   │   ├── introduccion.tex
│   │   ├── metodologia.tex
│   │   ├── resultados.tex
│   │   └── conclusiones.tex
│   └── figuras/                  ← Imágenes para el paper
└── README.md                     ← Este archivo
```

## 🚀 Flujo de Trabajo

### 1. Descargar el Dataset

Descargar desde Kaggle:
- **URL:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **Archivo:** `dataset.csv`
- **Destino:** `Agrupamiento/data/dataset.csv`

### 2. Ejecutar el Notebook

```bash
cd Agrupamiento/notebook/
jupyter notebook spotify_clustering.ipynb
```

El notebook sigue CRISP-DM:
1. **Celda 1:** Portada
2. **Celda 2:** Imports
3. **Celda 3:** Carga y EDA
4. **Celda 4:** Histogramas de features de audio
5. **Celda 5:** Mapa de correlación interactivo
6. **Celda 6:** Scatter danceability vs energy
7. **Celda 7:** Preparación de datos (StandardScaler)
8. **Celda 8:** Método del Codo y Silhouette Score
9. **Celda 9:** K-Means con K óptimo
10. **Celda 10:** Visualización PCA 2D
11. **Celda 11:** Perfil de cada cluster
12. **Celda 12:** Guardar modelos
13. **Celda 13:** Conclusiones

### 3. Compilar el Paper

```bash
cd Agrupamiento/paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 4. Ejecutar la App Flask

```bash
cd app/agrupamiento_app/
python app.py
```

La app se abrirá en: http://localhost:5002

## 📊 Dataset

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| danceability | float | 0-1 | Qué tan bailable es |
| energy | float | 0-1 | Intensidad y actividad |
| loudness | float | -60-0 | Volumen promedio (dB) |
| speechiness | float | 0-1 | Presencia de voz hablada |
| acousticness | float | 0-1 | Probabilidad acústica |
| instrumentalness | float | 0-1 | Sin voces |
| liveness | float | 0-1 | Grabación en vivo |
| valence | float | 0-1 | Positividad musical |
| tempo | float | 0-250 | BPM |
| popularity | int | 0-100 | Popularidad Spotify |

## 🧠 Algoritmo

- **K-Means** con K=6 (ajustable según método del codo)
- **StandardScaler** para normalización
- **PCA** para visualización 2D
- Métricas: Silhouette Score, Davies-Bouldin Index

## 🛠️ Stack Tecnológico

- Python 3.12
- scikit-learn (KMeans, PCA, StandardScaler)
- Flask + Plotly (App Web)
- joblib (persistencia de modelos)
- Matplotlib + Seaborn (EDA)

## 📋 Requisitos

```bash
pip install pandas numpy matplotlib seaborn plotly
pip install scikit-learn joblib flask
```

## 📝 Notas

- Ejecutar el notebook **antes** de la app Flask para generar los `.pkl`
- Ajustar `K_OPTIMO` según resultados del método del codo
- Los nombres de cluster se personalizan según el perfil observado

---
**Universidad Popular del Cesar** — Ingeniería de Sistemas
**Docente:** Aimer Rivera Centeno
