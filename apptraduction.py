import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
from gtts import gTTS
import tempfile
import speech_recognition as sr
import PyPDF2
import docx
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import wordnet
from fpdf import FPDF
import os
import time
import gc

# ==============================
# CONFIG STREAMLIT ET THÈME
# ==============================
st.set_page_config(
    page_title="🌍 Traduction Multilingue Intelligente",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thème sombre/clair
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.sidebar.checkbox("🌙 Mode sombre", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

# CSS personnalisé
css = """
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f4f4f4;
    color: #333;
    transition: background-color 0.3s ease;
}
[data-testid="stSidebar"] {
    background-color: #2c3e50;
    color: white;
    border-radius: 10px;
}
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #2980b9;
}
.stTextArea textarea {
    border-radius: 8px;
    border: 1px solid #ddd;
    padding: 10px;
}
.stSelectbox select, .stMultiselect select {
    border-radius: 8px;
    border: 1px solid #ddd;
}
.stAudioInput audio {
    border-radius: 8px;
}
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #2c3e50;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
.stExpander {
    border-radius: 8px;
    border: 1px solid #ddd;
}
.stProgress > div > div {
    background-color: #3498db;
}
</style>
"""
if dark_mode:
    css += """
    <style>
    body { background-color: #121212; color: white; }
    .stTextArea textarea, .stSelectbox select, .stMultiselect select { background-color: #333; color: white; border: 1px solid #555; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; }
    .stExpander { background-color: #333; border: 1px solid #555; }
    </style>
    """
st.markdown(css, unsafe_allow_html=True)

st.title("🌍 Traduction Multilingue Intelligente")
st.caption("Texte | Voix | Image | Fichier | Résumé | Synonymes | Mode Temps Réel")

# ==============================
# TESSERACT OCR
# ==============================
# ⚠️ Remplace ce chemin si différent
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\hassnae\Desktop\mini projet python\tesseract.exe"

# ==============================
# NLTK
# ==============================
nltk.download("wordnet")
nltk.download("omw-1.4")

# ==============================
# CACHE MODÈLES
# ==============================
@st.cache_resource
def charger_modele_traduction(src, tgt):
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except:
        st.sidebar.error(f"Modèle {src}-{tgt} indisponible.")
        return None, None

@st.cache_resource
def charger_outils_nlp():
    try:
        correcteur = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
        resumeur = pipeline("summarization", model="facebook/bart-large-cnn")
        return correcteur, resumeur
    except:
        st.sidebar.error("Erreur chargement outils NLP.")
        return None, None

correcteur, resumeur = charger_outils_nlp()

# ==============================
# FONCTIONS NLP
# ==============================
def detecter_langue(texte):
    try:
        return detect(texte)
    except:
        return "unknown"

def traduire_multilangue(texte, src, tgt):
    if src == tgt:
        return texte
    try:
        tok, mod = charger_modele_traduction(src, tgt)
        if tok and mod:
            inputs = tok(texte, return_tensors="pt", truncation=True)
            outputs = mod.generate(**inputs, max_length=256)
            return tok.decode(outputs[0], skip_special_tokens=True)
        else:
            # Pivot via anglais
            tok1, mod1 = charger_modele_traduction(src, "en")
            if tok1 and mod1:
                inputs = tok1(texte, return_tensors="pt", truncation=True)
                outputs = mod1.generate(**inputs, max_length=256)
                texte = tok1.decode(outputs[0], skip_special_tokens=True)
            tok2, mod2 = charger_modele_traduction("en", tgt)
            if tok2 and mod2:
                inputs = tok2(texte, return_tensors="pt", truncation=True)
                outputs = mod2.generate(**inputs, max_length=256)
                return tok2.decode(outputs[0], skip_special_tokens=True)
            return texte
    except MemoryError:
        st.error("Mémoire insuffisante. Essayez un texte plus court.")
        return texte

def corriger_texte(texte):
    if correcteur:
        try:
            return correcteur(texte, max_length=256)[0]["generated_text"]
        except MemoryError:
            st.error("Mémoire insuffisante pour la correction.")
            return texte
    return texte

def resumer_texte(texte):
    if len(texte.split()) < 40:
        return texte
    if resumeur:
        try:
            return resumeur(texte, max_length=120, min_length=40)[0]["summary_text"]
        except MemoryError:
            st.error("Mémoire insuffisante pour le résumé.")
            return texte
    return texte

def compter_mots(texte):
    return len(texte.split())

def synonymes(mot):
    try:
        syns = wordnet.synsets(mot)
        return list(set(lemma.name() for s in syns for lemma in s.lemmas()))[:10]
    except:
        return ["Aucun synonyme trouvé"]

def texte_vers_voix(texte, lang):
    try:
        tts = gTTS(text=texte, lang=lang)
        file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(file.name)
        return file.name
    except:
        st.error("Erreur génération audio.")
        return None

# ==============================
# AUDIO STREAMLIT
# ==============================
def audio_vers_texte_streamlit(audio_bytes, lang="fr-FR"):
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name
        with sr.AudioFile(path) as source:
            audio = r.record(source)
        return r.recognize_google(audio, language=lang)
    except:
        return ""

# ==============================
# LECTURE FICHIERS
# ==============================
def lire_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join(p.extract_text() for p in reader.pages if p.extract_text())
    except:
        return ""

def lire_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""

def image_vers_texte(file):
    try:
        img = Image.open(file)
        img = img.convert('L')
        if img.size[0] > 2000 or img.size[1] > 2000:
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        text = pytesseract.image_to_string(img, lang="eng+fra", config="--psm 6")
        if not text.strip():
            st.warning("⚠️ Aucun texte détecté")
        return text
    except Exception as e:
        st.error(f"❌ Erreur OCR : {e}")
        return ""

# ==============================
# TÉLÉCHARGEMENTS
# ==============================
def creer_fichier_txt(texte):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    f.write(texte)
    f.close()
    return f.name

def creer_fichier_pdf(texte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, texte)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(f.name)
    return f.name

# ==============================
# SESSION STATE
# ==============================
if "historique" not in st.session_state:
    st.session_state.historique = []

# ==============================
# INTERFACE
# ==============================
tab1, tab2, tab3 = st.tabs(["🚀 Traduction", "📜 Historique", "🔍 Outils"])

with tab1:
    st.subheader("Traduction Intelligente")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        texte = st.text_area("✏️ Texte à traduire", height=160)
    
    with col2:
        audio_input = st.audio_input("🎙️ Enregistrer votre voix")
        if audio_input:
            texte_reconnu = audio_vers_texte_streamlit(audio_input.getbuffer())
            if texte_reconnu:
                texte = texte_reconnu
                st.success("Voix reconnue !")
    
    uploaded_file = st.file_uploader(
        "📂 Importer fichier (txt/pdf/docx/png/jpg)",
        type=["txt","pdf","docx","png","jpg"]
    )
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            texte = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            texte = lire_pdf(uploaded_file)
        elif "word" in uploaded_file.type:
            texte = lire_docx(uploaded_file)
        elif uploaded_file.type.startswith("image"):
            texte = image_vers_texte(uploaded_file)

    if texte.strip():
        langue_detectee = detecter_langue(texte)
        st.info(f"🌍 Langue détectée : {langue_detectee.upper()}")

    col3, col4 = st.columns(2)
    with col3:
        langue_cible = st.selectbox("🌍 Langue cible", ["en","fr","es","de","ar","pt","ru","it","zh","ja"])
    with col4:
        options = st.multiselect("⚙️ Options", ["Correction", "Résumé"])

    if st.button("🚀 Traduire"):
        if texte.strip():
            if "Correction" in options:
                texte = corriger_texte(texte)
            if "Résumé" in options:
                texte = resumer_texte(texte)
            
            traduction = traduire_multilangue(texte, detecter_langue(texte), langue_cible)
            st.subheader("📌 Résultat")
            st.success(traduction)
            st.info(f"🧮 Mots : {compter_mots(traduction)}")

            audio_file = texte_vers_voix(traduction, langue_cible)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")

            st.session_state.historique.append({
                "src": texte,
                "tgt": traduction,
                "lang_src": detecter_langue(texte),
                "lang_tgt": langue_cible
            })

            col5, col6 = st.columns(2)
            with col5:
                txt_file = creer_fichier_txt(traduction)
                with open(txt_file,"rb") as f:
                    st.download_button("📄 Télécharger TXT", data=f, file_name="traduction.txt")
            with col6:
                pdf_file = creer_fichier_pdf(traduction)
                with open(pdf_file,"rb") as f:
                    st.download_button("📕 Télécharger PDF", data=f, file_name="traduction.pdf")

with tab2:
    st.subheader("Historique")
    if st.session_state.historique:
        search = st.text_input("🔍 Rechercher")
        filtered = [h for h in st.session_state.historique if search.lower() in h['src'].lower() or search.lower() in h['tgt'].lower()] if search else st.session_state.historique
        for h in reversed(filtered):
            with st.expander(f"Original ({h['lang_src']}): {h['src'][:50]}..."):
                st.write(f"**Traduit ({h['lang_tgt']}):** {h['tgt']}")
        if st.button("🗑️ Effacer Historique"):
            st.session_state.historique = []
            st.rerun()
    else:
        st.info("Aucun historique.")

with tab3:
    st.subheader("Outils")
    mot = st.text_input("🔎 Mot pour synonymes")
    if mot:
        st.write(", ".join(synonymes(mot)))

st.markdown('<div class="footer">Développé avec ❤️ | Version 2.0</div>', unsafe_allow_html=True)
