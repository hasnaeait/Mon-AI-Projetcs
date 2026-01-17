import streamlit as st
import numpy as np
import cv2
import pickle
import os
from PIL import Image
import plotly.express as px  # Pour des graphiques interactifs (optionnel)

# ===============================
# Config page (DOIT être la première commande Streamlit !)
# ===============================
st.set_page_config(
    page_title="Smart Trash Classification",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Charger modèle, scaler et classes (avec gestion d'erreurs)
# ===============================
@st.cache_data
def load_model():
    try:
        with open("trash_rf_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Erreur : Fichier 'trash_rf_model.pkl' introuvable. Assurez-vous qu'il est dans le même dossier.")
        st.stop()

@st.cache_data
def load_scaler():
    try:
        with open("trash_scaler.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Erreur : Fichier 'trash_scaler.pkl' introuvable.")
        st.stop()

@st.cache_data
def load_classes():
    try:
        with open("trash_classes.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Erreur : Fichier 'trash_classes.pkl' introuvable.")
        st.stop()

model = load_model()
scaler = load_scaler()
classes = load_classes()

IMG_SIZE = 64
MAX_IMG_SIZE = (2048, 2048)  # Limite pour éviter les surcharges
MAX_FILE_SIZE_MB = 5  # Taille max en MB

# ===============================
# Fonctions utilitaires
# ===============================
def get_recycling_tip(waste_class):
    tips = {
        "Cardboard": "Recyclez dans le bac jaune. Assurez-vous qu'il est propre et sec.",
        "Compost": "Jetez dans le composteur. Idéal pour les déchets organiques !",
        "Glass": "Recyclez dans le bac vert. Rincez les bouteilles avant.",
        "Metal": "Recyclez dans le bac jaune. Magnétique ou non, c'est recyclable.",
        "Paper": "Recyclez dans le bac bleu. Évitez le papier sale ou plastifié.",
        "Plastic": "Recyclez dans le bac jaune. Vérifiez le symbole de recyclage.",
        "Trash": "Ce n'est pas recyclable. Jetez dans la poubelle ordinaire."
    }
    return tips.get(waste_class, "Conseil non disponible.")

def detect_object_with_model(img_array, model, scaler, threshold=50.0):
    """
    Détection d'objet en utilisant le modèle entraîné.
    Prétraitement rapide et prédiction : si la confiance max > seuil, considère qu'un objet valide est détecté.
    """
    try:
        # Prétraitement rapide
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        flat = resized.flatten().reshape(1, -1)
        flat_scaled = scaler.transform(flat)
        
        # Prédiction
        probabilities = model.predict_proba(flat_scaled)[0]
        max_confidence = np.max(probabilities) * 100
        
        return max_confidence > threshold
    except Exception as e:
        st.error(f"Erreur lors de la détection : {e}")
        return False

# ===============================
# Initialiser session state pour l'historique
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0

# ===============================
# Sidebar pour options
# ===============================
with st.sidebar:
    st.header("⚙️ Options")
    dark_mode = st.toggle("Mode sombre", value=False, help="Basculez entre thème clair et sombre.")
    show_details = st.checkbox("Afficher les détails des probabilités", value=True, help="Montre un graphique des probabilités.")
    show_history = st.checkbox("Afficher l'historique", value=False, help="Affiche les dernières prédictions.")
    auto_capture = st.toggle("Mode Auto-Capture", value=False, help="Active la capture automatique si un déchet est détecté via le modèle.")
    
    # Métriques
    st.metric("Images traitées", st.session_state.total_processed)
    
    # Boutons
    if st.button("🔄 Réinitialiser l'application", help="Efface l'historique et recommence."):
        st.session_state.history = []
        st.session_state.total_processed = 0
        st.rerun()
    
    if st.button("🗑️ Vider l'historique", help="Supprime l'historique des prédictions."):
        st.session_state.history = []
        st.success("Historique vidé !")
    
    # Aide
    with st.expander("ℹ️ Aide & Instructions"):
        st.write("""
        - **Upload** : Choisissez des images ou utilisez la caméra.
        - **Auto-Capture** : Activez pour détecter automatiquement un déchet et traiter.
        - **Prédiction** : L'IA analyse et classe le déchet.
        - **Conseils** : Suivez les recommandations de recyclage.
        - **Historique** : Consultez vos prédictions précédentes.
        """)

# ===============================
# CSS (Design ultra-moderne avec animations)
# ===============================
if dark_mode:
    bg_color = "#1e1e1e"
    text_color = "#ffffff"
    card_bg = "#2d2d2d"
    accent_color = "#bb86fc"
else:
    bg_color = "#f9f9f9"
    text_color = "#0D1B2A"
    card_bg = "#ffffff"
    accent_color = "#667eea"

st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}
.header {{
    background: linear-gradient(135deg, {accent_color} 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    animation: fadeIn 1s ease-in;
}}
.header h1 {{
    color: white;
    font-size: 40px;
    margin: 0;
}}
.card {{
    background-color: {card_bg};
    border-radius: 20px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}}
.card:hover {{
    transform: translateY(-5px);
}}
.prediction {{
    font-size: 28px;
    font-weight: bold;
    color: {accent_color};
    text-align: center;
    animation: bounce 0.5s ease;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}
@keyframes bounce {{
    0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
    40% {{ transform: translateY(-10px); }}
    60% {{ transform: translateY(-5px); }}
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown("""
<div class="header">
    <h1>🗑️ Smart Trash Classification</h1>
</div>
<p style="text-align:center; font-style:italic; color:{text_color}; font-size:18px;">
AI-based waste classification system for smarter recycling ♻️
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# Description des classes avec couleurs
# ===============================
st.markdown(f"""
<div class="card">
<h3>♻️ Supported Waste Classes</h3>
<ul>
<li style="color: #8B4513;">📦 Cardboard</li>
<li style="color: #228B22;">🌱 Compost</li>
<li style="color: #00CED1;">🥤 Glass</li>
<li style="color: #FFD700;">🔧 Metal</li>
<li style="color: #4169E1;">📄 Paper</li>
<li style="color: #FF6347;">🛍️ Plastic</li>
<li style="color: #696969;">🗑️ Trash</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ===============================
# Upload d'images et caméra
# ===============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Images or Capture")

# Caméra en temps réel
camera_image = st.camera_input("Take a photo with your camera", help="Utilisez votre caméra pour capturer un déchet en direct.")

# Uploader de fichiers
uploaded_files = st.file_uploader(
    "Or choose files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Sélectionnez une ou plusieurs images (max 5MB chacune)."
)

# Combiner les sources
all_images = []
if camera_image:
    all_images.append(("Camera Capture", camera_image))
if uploaded_files:
    for uf in uploaded_files:
        all_images.append((uf.name, uf))

if all_images:
    for img_name, img_data in all_images:
        with st.spinner(f"Traitement de {img_name}..."):
            # Validation de l'image
            try:
                img = Image.open(img_data).convert("RGB")
                if img.size[0] > MAX_IMG_SIZE[0] or img.size[1] > MAX_IMG_SIZE[1]:
                    st.warning(f"Image {img_name} trop grande. Redimensionnement automatique.")
                    img.thumbnail(MAX_IMG_SIZE)
                file_size_mb = len(img_data.getvalue()) / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"Fichier {img_name} trop volumineux ({file_size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB).")
                    continue
                img_array = np.array(img)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(f"Erreur lors du chargement de {img_name} : {e}")
                continue
            
            # Détection automatique avec le modèle si activée
            if auto_capture and img_name == "Camera Capture":
                if not detect_object_with_model(img_array, model, scaler):
                    st.warning("Aucun déchet détecté dans l'image (confiance trop faible). Essayez de repositionner ou désactivez le mode auto.")
                    continue  # Passe à la suivante sans traiter
            
            # Afficher l'image avec colonnes
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption=f"Image: {img_name}", use_container_width=True)
            
            # Prétraitement
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            flat = resized.flatten().reshape(1, -1)
            flat_scaled = scaler.transform(flat)
            
            # Prédiction
            @st.cache_data
            def predict_waste(features):
                pred = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                return pred, probs
            
            prediction, probabilities = predict_waste(flat_scaled)
            label = classes[prediction]
            confidence = np.max(probabilities) * 100
            
            # Mettre à jour l'historique et les métriques
            st.session_state.history.append({"name": img_name, "class": label, "confidence": confidence})
            st.session_state.total_processed += 1
            
            # Résultats
            with col2:
                st.markdown(
                    f"<p class='prediction'>🧠 Detected Waste: <b>{label}</b></p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='text-align:center; font-size:20px;'>Confidence: <b>{confidence:.2f}%</b></p>",
                    unsafe_allow_html=True
                )
                
                # Conseil de recyclage
                tip = get_recycling_tip(label)
                st.info(f"💡 Conseil : {tip}")
                
                if show_details:
                    st.subheader("📊 Probabilities for All Classes")
                    try:
                        fig = px.bar(
                            x=classes,
                            y=probabilities * 100,
                            labels={'x': 'Waste Class', 'y': 'Probability (%)'},
                            title="Prediction Probabilities",
                            color=probabilities,
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.bar_chart({cls: prob * 100 for cls, prob in zip(classes, probabilities)})
        
        st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Historique (si activé)
# ===============================
if show_history and st.session_state.history:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📜 Prediction History")
    for entry in reversed(st.session_state.history[-10:]):  # Derniers 10
        st.write(f"- **{entry['name']}** : {entry['class']} ({entry['confidence']:.2f}%)")
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown(f"""
<p style="text-align:center; margin-top:40px; color:gray; font-size:16px;">
Developed for Smart Waste Sorting System ♻️<br>
Machine Learning • Random Forest • Image Processing<br>
<a href="https://github.com/your-repo" target="_blank">View on GitHub</a> | <a href="https://streamlit.io" target="_blank">Powered by Streamlit</a> | <a href="mailto:feedback@example.com" target="_blank">Feedback</a>
</p>
""", unsafe_allow_html=True)
