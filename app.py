import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from translate import Translator
import re

# 1. Configuración de página
st.set_page_config(
    page_title="Studio NLP: Sentimiento & TF-IDF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo UI/UX en modo oscuro y animaciones CSS
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

    /* Animaciones CSS para Sentimiento */
    @keyframes float-anim {
        0% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-10px) scale(1.05); }
        100% { transform: translateY(0px) scale(1); }
    }

    @keyframes shake-anim {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-6px) rotate(-5deg); }
        75% { transform: translateX(6px) rotate(5deg); }
    }

    .avatar-box-positive {
        background: #14532d;
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        animation: float-anim 2.5s infinite ease-in-out;
    }

    .avatar-box-negative {
        background: #451a1a;
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        animation: shake-anim 1.5s infinite ease-in-out;
    }

    .avatar-box-neutral {
        background: #1e293b;
        border: 1px solid #64748b;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    .avatar-emoji {
        font-size: 4rem;
        line-height: 1;
        margin-bottom: 8px;
    }

    /* Animaciones CSS exclusivas para TF-IDF */
    @keyframes pulse-radar {
        0% { transform: scale(0.98); opacity: 0.8; box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.5); }
        50% { transform: scale(1.02); opacity: 1; box-shadow: 0 0 20px 5px rgba(37, 99, 235, 0.4); }
        100% { transform: scale(0.98); opacity: 0.8; box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.5); }
    }

    .tfidf-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: pulse-radar 3s infinite ease-in-out;
    }

    .top-word-highlight {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
        letter-spacing: 1px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

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
    st.write("Ingresa una reseña, poema o texto para calcular la polaridad del sentimiento y visualizar el personaje correspondiente:")
    
    texto_sentimiento = st.text_area(
        "Texto de entrada:",
        placeholder="Ingresa tu texto aquí...",
        height=140,
        label_visibility="collapsed"
    )
    
    if st.button("Procesar Sentimiento y WordCloud", use_container_width=True):
        if texto_sentimiento.strip() != "":
            try:
                translator = Translator(to_lang="en")
                texto_en = translator.translate(texto_sentimiento)
            except:
                texto_en = texto_sentimiento

            blob = TextBlob(texto_en)
            polaridad = blob.sentiment.polarity
            subjetividad = blob.sentiment.subjectivity
            
            palabras_negativas = ["herida", "dolor", "doler", "desventura", "muerte", "lloro", "triste", "mal", "miedo"]
            if any(palabra in texto_sentimiento.lower() for palabra in palabras_negativas):
                if polaridad > 0:
                    polaridad = -abs(polaridad) if polaridad != 0 else -0.5

            col_res, col_anim = st.columns([1, 1], gap="large")
            
            with col_res:
                st.markdown("##### Métricas del Análisis")
                st.write(f"**Polaridad (-1.0 a 1.0):** `{polaridad:.2f}`")
                st.write(f"**Subjetividad (0.0 a 1.0):** `{subjetividad:.2f}`")
                
                if polaridad < -0.05:
                    st.error("Sentimiento detectado: Negativo / Melancólico")
                    card_html = """
                    <div class="avatar-box-negative">
                        <div class="avatar-emoji">🥺🌧️</div>
                        <h4 style="margin:0; color:#fca5a5;">Personaje Melancólico</h4>
                        <p style="margin:0; font-size:0.85rem; color:#f87171;">Emoción triste o dolorosa detectada</p>
                    </div>
                    """
                elif polaridad > 0.05:
                    st.success("Sentimiento detectado: Positivo")
                    card_html = """
                    <div class="avatar-box-positive">
                        <div class="avatar-emoji">🥳✨</div>
                        <h4 style="margin:0; color:#86efac;">Personaje Entusiasmado</h4>
                        <p style="margin:0; font-size:0.85rem; color:#4ade80;">Emoción alegre y optimista detectada</p>
                    </div>
                    """
                else:
                    st.info("Sentimiento detectado: Neutral")
                    card_html = """
                    <div class="avatar-box-neutral">
                        <div class="avatar-emoji">🧐💬</div>
                        <h4 style="margin:0; color:#cbd5e1;">Personaje Reflexivo</h4>
                        <p style="margin:0; font-size:0.85rem; color:#94a3b8;">Texto informativo o neutral</p>
                    </div>
                    """
            
            with col_anim:
                st.markdown("##### Personaje Interactivo")
                st.markdown(card_html, unsafe_allow_html=True)
            
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
    st.markdown("#### Análisis TF-IDF con Animación de Radar")
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
                
                promedio_tfidf = df_tfidf.mean(axis=0).sort_values(ascending=False)
                top_word = promedio_tfidf.index[0]
                top_score = promedio_tfidf.iloc[0]
                
                # Sección superior con la animación del radar para la palabra clave
                col_chart, col_radar = st.columns([1.2, 0.8], gap="large")
                
                with col_radar:
                    st.markdown("##### Término Clave Dominante")
                    tfidf_anim_html = f"""
                    <div class="tfidf-card">
                        <div style="font-size:2.5rem;">📡 Network Radar</div>
                        <div class="top-word-highlight">"{top_word.upper()}"</div>
                        <p style="margin:0; color:#94a3b8; font-size:0.9rem;">
                            Mayor índice TF-IDF global: <b>{top_score:.3f}</b>
                        </p>
                    </div>
                    """
                    st.markdown(tfidf_anim_html, unsafe_allow_html=True)

                with col_chart:
                    st.markdown("##### Distribución del Top de Términos")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    fig.patch.set_facecolor('#0f1117')
                    ax.set_facecolor('#181b24')
                    
                    promedio_tfidf.plot(kind='bar', ax=ax, color='#3b82f6')
                    ax.tick_params(colors='#e2e8f0', which='both')
                    ax.spines['bottom'].set_color('#262b36')
                    ax.spines['top'].set_color('#262b36')
                    ax.spines['right'].set_color('#262b36')
                    ax.spines['left'].set_color('#262b36')
                    plt.xticks(rotation=45, ha='right')
                    
                    st.pyplot(fig)
                
                st.markdown("---")
                st.markdown("##### Matriz Completa de Relevancia TF-IDF")
                st.dataframe(df_tfidf.style.background_gradient(cmap="Blues"), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al calcular TF-IDF: {e}")
        else:
            st.warning("Ingresa al menos dos líneas distintas de texto para comparar.")
