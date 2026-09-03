import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from translate import Translator
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
import re

# Descargar datos NLTK requeridos
nltk.download('punkt', quiet=True)

# 1. Configuración de página
st.set_page_config(
    page_title="Studio NLP: Sentimiento, TF-IDF & QA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos UI/UX en modo oscuro y animaciones CSS
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
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.8rem 1.2rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        border-color: #3b82f6;
        background-color: #334155;
        color: #ffffff;
        transform: translateY(-2px);
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

    /* Tarjetas de Ánimo */
    .mood-card-happy {
        background: linear-gradient(135deg, #065f46 0%, #022c22 100%);
        border: 1px solid #34d399;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .mood-card-sad {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        border: 1px solid #60a5fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .mood-card-neutral {
        background: linear-gradient(135deg, #334155 0%, #0f172a 100%);
        border: 1px solid #94a3b8;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
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

    /* Radar TF-IDF */
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

    /* Animaciones exclusivas para QA (Preguntas y Respuestas) */
    @keyframes qa-scan-beam {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes pulse-glow-emerald {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); border-color: #10b981; }
        50% { box-shadow: 0 0 25px 8px rgba(16, 185, 129, 0.6); border-color: #34d399; }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); border-color: #10b981; }
    }

    .qa-animated-card {
        background: linear-gradient(90deg, #064e3b 0%, #022c22 50%, #064e3b 100%);
        background-size: 200% 100%;
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        animation: qa-scan-beam 4s infinite linear, pulse-glow-emerald 2.5s infinite ease-in-out;
    }

    .qa-radar-icon {
        font-size: 2.8rem;
        animation: float-anim 2s infinite ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# 2. Configuración del menú lateral
with st.sidebar:
    st.markdown("### Navegación NLP")
    
    opcion_menu = st.radio(
        "Módulo de trabajo:",
        [
            "Análisis de Sentimiento & WordCloud",
            "Análisis de Relevancia TF-IDF",
            "Demo TF-IDF Preguntas y Respuestas"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Configuración")
    
    if opcion_menu == "Análisis de Sentimiento & WordCloud":
        ancho_nube = st.slider("Ancho de Nube de Palabras", 400, 800, 600, 50)
        alto_nube = st.slider("Alto de Nube de Palabras", 200, 500, 300, 50)
    elif opcion_menu == "Análisis de Relevancia TF-IDF":
        max_features = st.slider("Máximo de términos (Top Words)", 5, 20, 10)
    else:
        idioma_qa = st.selectbox("Idioma del motor QA:", ["Español", "Inglés"])
        usar_stemming = st.checkbox("Aplicar Stemming (Normalización)", value=True)

# Encabezado principal
st.markdown('<div class="brand-header">Studio NLP & Text Analytics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="brand-sub"><span class="tag-pill">Módulo Activo</span> {opcion_menu}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# MÓDULO 1: ANÁLISIS DE SENTIMIENTO & WORDCLOUD
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

    st.markdown("---")
    st.markdown("#### ¿Cómo te sientes hoy?")
    st.write("Selecciona una emoción para activar la experiencia interactiva:")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        if st.button("🥳 Estoy muy feliz", use_container_width=True):
            st.session_state["mood"] = "happy"
    with col_b2:
        if st.button("🥺 Estoy triste", use_container_width=True):
            st.session_state["mood"] = "sad"
    with col_b3:
        if st.button("🧐 Estoy neutral", use_container_width=True):
            st.session_state["mood"] = "neutral"

    if "mood" in st.session_state:
        mood = st.session_state["mood"]
        st.markdown("<br>", unsafe_allow_html=True)
        
        if mood == "happy":
            st.balloons()
            st.markdown("""
            <div class="mood-card-happy">
                <h3 style="margin:0; color:#6ee7b7;">🥳 Sentimiento: Positivo / Feliz</h3>
                <p style="margin:8px 0 0 0; color:#a7f3d0;">✨ ¡Mantén esa sonrisa y comparte tu buena energía todo el día!</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif mood == "sad":
            st.snow()
            st.markdown("""
            <div class="mood-card-sad">
                <h3 style="margin:0; color:#93c5fd;">🥺 Sentimiento: Melancólico / Triste</h3>
                <p style="margin:8px 0 0 0; color:#bfdbfe;">❄️ tómate un respiro. Mañana será un día mejor y lleno de nuevas oportunidades.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif mood == "neutral":
            st.toast("💡 Estado neutral activado: ¡Modo reflexivo!", icon="🧐")
            st.markdown("""
            <div class="mood-card-neutral">
                <h3 style="margin:0; color:#e2e8f0;">🧐 Sentimiento: Neutral / Calmo</h3>
                <p style="margin:8px 0 0 0; color:#94a3b8;">💬 Un día de calma y equilibrio es ideal para enfocarte y descansar.</p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------
# MÓDULO 2: ANÁLISIS DE RELEVANCIA TF-IDF
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


# ---------------------------------------------------------
# MÓDULO 3: DEMO TF-IDF PREGUNTAS Y RESPUESTAS (QA)
# ---------------------------------------------------------
elif opcion_menu == "Demo TF-IDF Preguntas y Respuestas":
    st.markdown("#### Demo de TF-IDF con Preguntas y Respuestas")
    st.write("Cada línea se trata como un **documento**. Ingresa oraciones y formula preguntas para activar el escáner semántico:")
    
    if idioma_qa == "Español":
        doc_default = "El perro ladra fuerte en el parque.\nEl gato maúlla suavemente durante la noche.\nEl perro y el gato juegan juntos en el jardín.\nLos niños corren y se divierten en el parque.\nLa música suena muy alta en la fiesta."
        preg_default = "¿Dónde juegan el perro y el gato?"
    else:
        doc_default = "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
        preg_default = "Who is playing?"

    col_input, col_sug = st.columns([1.2, 0.8], gap="large")

    with col_input:
        st.markdown("**Escribe tus documentos (uno por línea):**")
        docs_input = st.text_area("Documentos:", value=doc_default, height=150, label_visibility="collapsed")
        
        st.markdown("**Escribe tu pregunta:**")
        pregunta_input = st.text_input("Pregunta:", value=preg_default, label_visibility="collapsed")

    with col_sug:
        st.markdown("##### 💡 Preguntas Sugeridas")
        if idioma_qa == "Español":
            st.code("¿Dónde juegan el perro y el gato?\n¿Qué hacen los niños en el parque?\n¿Dónde suena la música alta?\n¿Qué animal maúlla durante la noche?", language="text")
        else:
            st.code("Who is playing?\nWhat does the dog do?\nWhen does the cat meow?", language="text")

    if st.button("🔍 Calcular TF-IDF y Buscar Respuesta", use_container_width=True):
        documentos_list = [doc.strip() for doc in docs_input.split('\n') if doc.strip() != ""]
        
        if len(documentos_list) > 0 and pregunta_input.strip() != "":
            try:
                def preprocess_stem(text_list, lang):
                    if not usar_stemming:
                        return text_list
                    stemmer = PorterStemmer() if lang == "Inglés" else SnowballStemmer("spanish")
                    processed = []
                    for doc in text_list:
                        words = re.findall(r'\w+', doc.lower())
                        stemmed_words = [stemmer.stem(w) for w in words]
                        processed.append(" ".join(stemmed_words))
                    return processed

                corpus_full = documentos_list + [pregunta_input]
                corpus_processed = preprocess_stem(corpus_full, idioma_qa)
                
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(corpus_processed)
                
                doc_vectors = tfidf_matrix[:-1]
                question_vector = tfidf_matrix[-1]
                
                similaridades = cosine_similarity(question_vector, doc_vectors).flatten()
                
                best_idx = similaridades.argmax()
                best_score = similaridades[best_idx]
                best_doc = documentos_list[best_idx]
                
                st.markdown("---")
                col_ans, col_scan = st.columns([1.1, 0.9], gap="large")
                
                with col_ans:
                    st.markdown("##### 🎯 Respuesta con Escaneo Semántico")
                    if best_score > 0:
                        qa_card_html = f"""
                        <div class="qa-animated-card">
                            <div class="qa-radar-icon">🎯📡</div>
                            <span style="background:#10b981; color:#ffffff; padding:4px 12px; border-radius:12px; font-size:0.8rem; font-weight:700;">Coincidencia Detectada</span>
                            <h3 style="margin:12px 0 6px 0; color:#a7f3d0; font-size:1.35rem;">"{best_doc}"</h3>
                            <p style="margin:0; color:#9ca3af; font-size:0.85rem;">Extraído del Documento #{best_idx + 1}</p>
                        </div>
                        """
                        st.markdown(qa_card_html, unsafe_allow_html=True)
                    else:
                        st.warning("No se encontró coincidencia relevante para esa pregunta en los documentos ingresados.")

                with col_scan:
                    st.markdown("##### 📊 Nivel de Coincidencia Vectorial")
                    st.write(f"**Puntaje Coseno TF-IDF:** `{best_score:.4f}`")
                    st.progress(float(best_score))
                    
                    df_scores = pd.DataFrame({
                        "Documento": [f"Doc {i+1}: {doc[:30]}..." for i, doc in enumerate(documentos_list)],
                        "Similitud Coseno": similaridades
                    }).sort_values(by="Similitud Coseno", ascending=False)
                    
                    st.dataframe(df_scores.style.background_gradient(cmap="Greens"), use_container_width=True)

            except Exception as e:
                st.error(f"Error al procesar la consulta QA: {e}")
        else:
            st.warning("Por favor ingresa documentos y una pregunta para realizar el análisis.")
