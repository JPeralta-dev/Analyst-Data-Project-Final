# 📄 Paper Unificado — Segmentación de Canciones Spotify Mediante Algoritmos de Agrupamiento

> **Documento completo para sustentación**
> Contiene la totalidad del contenido LaTeX del paper de agrupamiento en formato legible.
> Generado a partir de: `main.tex`, `referencias.bib`, `resumen.tex`, `introduccion.tex`,
> `metodologia.tex`, `resultados.tex`, `conclusiones.tex`

---

## Título

**Segmentación de Canciones Spotify Mediante Algoritmos de Agrupamiento: Un Enfoque Basado en Características de Audio**

**Autores:** [Nombre Estudiante 1], [Nombre Estudiante 2]
**Institución:** Universidad Popular del Cesar — Ingeniería de Sistemas
**Metodología:** CRISP-DM

---

## Resumen

Los servicios de streaming musical generan enormes volúmenes de datos de audio que ofrecen oportunidades para el descubrimiento automático de patrones musicales. Este trabajo aplica algoritmos de agrupamiento (*clustering*) sobre el Spotify Tracks Dataset, que contiene 114,000 canciones de 125 géneros con 10 características de audio cuantitativas. Se empleó K-Means como algoritmo principal, complementado con análisis PCA para reducción dimensional y visualización. El proceso siguió la metodología CRISP-DM. Los resultados identificaron **[K]** clusters con perfiles musicales diferenciados, obteniendo un Silhouette Score de **[valor]** , lo que evidencia agrupaciones coherentes con los géneros musicales reales. La aplicación práctica es un sistema de recomendación musical basado en similitud de cluster.

**Palabras clave:** clustering, K-Means, Spotify, características de audio, PCA, sistema de recomendación, CRISP-DM.

---

## 1. Introducción

Con más de 600 millones de usuarios activos y un catálogo superior a los 100 millones de canciones, Spotify es la plataforma de streaming musical más grande del mundo (Spotify, 2024). El éxito de su sistema de recomendación —responsable del 30% de los plays— depende en gran medida de técnicas de aprendizaje automático que identifican similitudes entre canciones y preferencias de usuarios (Schedl et al., 2018).

El agrupamiento no supervisado ofrece una aproximación natural a la organización de contenido musical: sin requerir etiquetas manuales, permite descubrir estructuras latentes en el espacio de características de audio. Variables como *danceability*, *energy* y *valence*, derivadas del análisis de señales de audio, capturan aspectos perceptivos de la música que van más allá del género declarado (Turnbull et al., 2008).

El presente trabajo aplica K-Means y análisis de componentes principales (PCA) sobre el Spotify Tracks Dataset para identificar clusters musicales coherentes. El objetivo práctico es construir la base algorítmica de un sistema de recomendación que sugiera canciones similares basándose en el cluster al que pertenecen.

### 1.1 Objetivos

- Aplicar K-Means y métricas de evaluación (Silhouette, Davies-Bouldin) para determinar el número óptimo de clusters.
- Caracterizar cada cluster mediante el perfil promedio de sus características de audio.
- Visualizar los clusters en espacio reducido mediante PCA.
- Desplegar el sistema como aplicación web con Flask y Plotly.

---

## 2. Metodología

### 2.1 Dataset

El Spotify Tracks Dataset contiene 114,000 canciones distribuidas en 125 géneros musicales (Pandya, 2023). Cada registro incluye metadatos (artista, álbum, género) y 10 características de audio cuantitativas normalizadas por Spotify mediante análisis de señal.

### 2.2 Preprocesamiento

Las características seleccionadas para el clustering fueron: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo y popularity. Se aplicó *StandardScaler* para normalizar cada feature a media cero y desviación estándar unitaria, requisito fundamental para K-Means dado su dependencia de distancias euclidianas (Jain, 2010).

### 2.3 Algoritmo K-Means

K-Means minimiza la inercia intra-cluster:

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

donde $\boldsymbol{\mu}_k$ es el centroide del cluster $C_k$.

### 2.4 Selección de K

Se evaluaron valores de $K \in \{2, \ldots, 10\}$ utilizando:

- **Método del Codo**: punto de inflexión en la curva de inercia.
- **Silhouette Score**: mide cohesión interna y separación entre clusters. Rango $[-1, 1]$; valores cercanos a 1 indican clusters bien definidos.

### 2.5 Reducción Dimensional — PCA

PCA transforma el espacio de 10 dimensiones a 2 componentes principales para visualización, preservando la máxima varianza posible:

$$\mathbf{Z} = \mathbf{X} \mathbf{W}_{2}$$

donde $\mathbf{W}_2$ contiene los dos eigenvectores de mayor eigenvalor de la matriz de covarianza $\mathbf{\Sigma}$.

---

## 3. Resultados

### 3.1 Análisis Exploratorio

El análisis de correlación reveló relaciones significativas entre las características de audio: energy y loudness presentan correlación positiva fuerte ($r > 0.7$), mientras que energy y acousticness muestran correlación negativa ($r < -0.7$), consistente con la naturaleza opuesta de música electrónica/acústica (Schedl et al., 2018).

### 3.2 Determinación del K Óptimo

| Método | Resultado |
|--------|-----------|
| Método del Codo | K = **[completar]** |
| Silhouette Score | K = **[completar]** con valor **[completar]** |

> ⚠️ *Completar con valores reales del notebook ejecutado.*

### 3.3 Caracterización de Clusters

| Cluster | Danceability | Energy | Valence | Acousticness | N canciones |
|---------|-------------|--------|---------|--------------|-------------|
| 0 | — | — | — | — | — |
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| … | — | — | — | — | — |

---

## 4. Conclusiones

La aplicación de K-Means sobre las características de audio del dataset de Spotify permitió identificar **[K]** clusters musicales con perfiles diferenciados. Las principales conclusiones son:

1. La normalización es crítica: sin StandardScaler, las variables con mayor rango (tempo, loudness) dominarían la distancia euclidiana.
2. PCA con 2 componentes explica **[X]%** de la varianza, suficiente para visualización pero no para reconstrucción fiel.
3. Los clusters identificados corresponden parcialmente a géneros musicales reales, validando la coherencia del agrupamiento.
4. **[Completar con hallazgo específico del análisis]**

### Trabajo Futuro

Como trabajo futuro se propone explorar clustering jerárquico para análisis de subgéneros, y evaluar modelos de mezcla gaussiana (GMM) como alternativa probabilística a K-Means.

---

## 5. Referencias

1. Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, 31(8), 651–666.
2. Schedl, M., Knees, P., McFee, B., & Bogdanov, D. (2018). Current challenges and visions in music recommender systems research. *International Journal of Multimedia Information Retrieval*, 7, 95–116.
3. Turnbull, D., Barrington, L., Torres, D., & Lanckriet, G. (2008). Semantic annotation and retrieval of music and sound effects. *IEEE Transactions on Audio, Speech, and Language Processing*, 16(2), 467–476.
4. Pandya, M. (2023). Spotify Tracks Dataset. Kaggle.
5. Spotify Technology S.A. (2024). Spotify for the Record — Q4 2023 Earnings.
6. Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*.

---

## 6. Estructura del Proyecto (para sustentación)

```
Agrupamiento/
├── data/dataset.csv                     ← Dataset (colocado por el estudiante)
├── notebook/
│   └── spotify_clustering.ipynb         ← Notebook completo con CRISP-DM
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
