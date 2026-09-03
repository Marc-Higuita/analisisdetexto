import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from streamlit_lottie import st_lottie
from deep_translator import GoogleTranslator
import re

# 1. Configuración de página
st.set_page_config(
    page_title="Studio NLP: Sentimiento & TF-IDF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo UI/UX minimalista en modo oscuro
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0f1117;
        color: #e2e8f0;
    }
    
    .brand-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    
    .brand-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.8rem;
    }
    
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        border: none;
    }

    .tag-pill {
        display: inline-block;
        background-color: #262b36;
        color: #93c5fd;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Animaciones Lottie directas en CDN estable
LOTTIE_HAPPY = "https://lottie.host/8040d750-6e4f-4091-a1e1-e9451996f874/4EwOnYqDpx.json"
LOTTIE_SAD = "https://lottie.host/8c066f7f-3db6-410c-9ee8-74a441e8ef0f/J1s2w9X0y1.json"
LOTTIE_NEUTRAL = "https://lottie.host/28cb2eb5-2b4a-464a-9860-915eef22f280/HInL3fLqP7.json"

# 2. Configuración del menú lateral
with st.sidebar:
    st.markdown("### Navegación NLP")
    
    opcion_menu = st.radio(
        "Módulo de trabajo:",
        ["Análisis de Sentimiento & WordCloud", "Análisis de Relevancia TF-IDF"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Configuración")
    
    if opcion_menu == "Análisis de Sentimiento & WordCloud":
        ancho_nube = st.slider("Ancho de Nube de Palabras", 400, 800, 600, 50)
        alto_nube = st.slider("Alto de Nube de Palabras", 200, 500, 300, 50)
    else:
        max_features = st.slider("Máximo de términos (Top Words)", 5, 20, 10)

# Encabezado principal
st.markdown('<div class="brand-header">Studio NLP & Text Analytics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="brand-sub"><span class="tag-pill">Módulo Activo</span> {opcion_menu}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# EJERCICIO 1: ANÁLISIS DE SENTIMIENTO & WORDCLOUD
# ---------------------------------------------------------
if opcion_menu == "Análisis de Sentimiento & WordCloud":
    st.markdown("#### Análisis de Sentimiento y Nube de Palabras")
    st.write("Ingresa una reseña, poema o texto para calcular la polaridad del sentimiento y visualizar su personaje animado:")
    
    texto_sentimiento = st.text_area(
        "Texto de entrada:",
        placeholder="Ingresa tu texto aquí...",
        height=140,
        label_visibility="collapsed"
    )
    
    if st.button("Procesar Sentimiento y WordCloud", use_container_width=True):
        if texto_sentimiento.strip() != "":
            # Traducción robusta previa al inglés con deep-translator para precisión de TextBlob
            try:
                traductor = GoogleTranslator(source='auto', target='en')
                texto_en = traductor.translate(texto_sentimiento)
            except:
                texto_en = texto_sentimiento

            blob = TextBlob(texto_en)
            polaridad = blob.sentiment.polarity
            subjetividad = blob.sentiment.subjectivity
            
            col_res, col_anim = st.columns([1, 1], gap="large")
            
            with col_res:
                st.markdown("##### Métricas del Análisis")
                st.write(f"**Polaridad (-1.0 a 1.0):** `{polaridad:.2f}`")
                st.write(f"**Subjetividad (0.0 a 1.0):** `{subjetividad:.2f}`")
                
                if polaridad > 0.05:
                    st.success("Sentimiento detectado: Positivo")
                    lottie_url = LOTTIE_HAPPY
                elif polaridad < -0.05:
                    st.error("Sentimiento detectado: Negativo")
                    lottie_url = LOTTIE_SAD
                else:
                    st.info("Sentimiento detectado: Neutral")
                    lottie_url = LOTTIE_NEUTRAL
            
            with col_anim:
                st.markdown("##### Personaje Animado")
                try:
                    st_lottie(lottie_url, height=200, key="sentiment_lottie_anim")
                except:
                    st.warning("No se pudo cargar la animación.")
            
            st.markdown("---")
            st.markdown("##### Nube de Palabras (WordCloud)")
            
            texto_limpio = re.sub(r'[^\w\s]', '', texto_sentimiento.lower())
            
            if len(texto_limpio.split()) >= 2:
                wc = WordCloud(
                    width=ancho_nube,
                    height=alto_nube,
                    background_color='#181b24',
                    colormap='Blues',
                    max_words=50
                ).generate(texto_limpio)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor('#0f1117')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("Escribe más palabras para generar la Nube de Palabras.")
        else:
            st.warning("Por favor ingresa un texto antes de presionar el botón.")


# ---------------------------------------------------------
# EJERCICIO 2: ANÁLISIS DE RELEVANCIA TF-IDF
# ---------------------------------------------------------
elif opcion_menu == "Análisis de Relevancia TF-IDF":
    st.markdown("#### Análisis TF-IDF")
    st.write("Ingresa dos o más párrafos (uno por línea) para calcular la relevancia de cada término:")
    
    texto_tfidf = st.text_area(
        "Colección de Documentos:",
        value="El desarrollo de aplicaciones web multimodales es fascinante.\nLas aplicaciones web utilizan algoritmos de inteligencia artificial.\nEl procesamiento de lenguaje natural permite analizar textos en la web.",
        height=160,
        label_visibility="collapsed"
    )
    
    if st.button("Calcular Matriz TF-IDF", use_container_width=True):
        documentos = [doc.strip() for doc in texto_tfidf.split('\n') if doc.strip() != ""]
        
        if len(documentos) >= 2:
            try:
                vectorizer = TfidfVectorizer(max_features=max_features)
                tfidf_matrix = vectorizer.fit_transform(documentos)
                
                df_tfidf = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Documento {i+1}" for i in range(len(documentos))]
                )
                
                st.markdown("##### Matriz de Relevancia TF-IDF")
                st.dataframe(df_tfidf.style.background_gradient(cmap="Blues"), use_container_width=True)
                
                st.markdown("##### Términos más relevantes globalmente")
                promedio_tfidf = df_tfidf.mean(axis=0).sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 3.5))
                fig.patch.set_facecolor('#0f1117')
                ax.set_facecolor('#181b24')
                
                promedio_tfidf.plot(kind='bar', ax=ax, color='#2563eb')
                ax.tick_params(colors='#e2e8f0', which='both')
                ax.spines['bottom'].set_color('#262b36')
                ax.spines['top'].set_color('#262b36')
                ax.spines['right'].set_color('#262b36')
                ax.spines['left'].set_color('#262b36')
                plt.xticks(rotation=45, ha='right')
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error al calcular TF-IDF: {e}")
        else:
            st.warning("Ingresa al menos dos líneas distintas de texto para comparar.")
