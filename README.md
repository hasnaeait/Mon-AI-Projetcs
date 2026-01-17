# 🌍 Traduction Multilingue Intelligente

Une application Streamlit qui permet de traduire du texte, de la voix, des images et des fichiers (PDF, DOCX, TXT) dans plusieurs langues. Elle inclut des fonctionnalités avancées telles que la correction grammaticale, le résumé automatique, la détection de langue, la génération audio et la recherche de synonymes.

---

## 🚀 Fonctionnalités

- Traduction multilingue intelligente avec pivot via l'anglais si la paire de langues directe n’est pas disponible.
- Détection automatique de la langue du texte.
- Entrée texte, audio ou fichiers (PDF, DOCX, TXT, images).
- Correction grammaticale automatique du texte.
- Résumé automatique du texte.
- Conversion du texte traduit en audio (TTS) avec lecture intégrée.
- Historique des traductions avec option de recherche et suppression.
- Recherche de synonymes pour les mots.
- Thème clair/sombre avec interface moderne.

---

## 🛠️ Technologies utilisées

- Python 3.10+
- Streamlit
- Transformers (Hugging Face)
- Torch / PyTorch
- LangDetect
- gTTS (Google Text-to-Speech)
- SpeechRecognition
- PyPDF2
- python-docx
- pytesseract (OCR pour images)
- PIL / Pillow
- NLTK (WordNet)
- FPDF (pour créer des fichiers PDF)

---

## ⚡ Installation

1. **Cloner le dépôt :**
```bash
git clone https://github.com/TON_UTILISATEUR/nom_du_projet.git
cd nom_du_projet
