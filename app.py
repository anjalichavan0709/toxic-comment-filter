import streamlit as st
import pickle
import re
from fpdf import FPDF
from datetime import datetime
import base64
from PIL import Image
import streamlit.components.v1 as components

# ============================
#       PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Comment Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
#       LOAD MODELS
# ============================
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/toxicity_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================
#         STYLES (NEON)
# ============================
st.markdown("""
<style>
body { background-color: #000000; }

section.main { background-color: #000000; }

input, textarea {
    color: #ffffff !important;
    font-weight: 600 !important;
}

.stTextInput>div>div>input {
    background-color: #111111 !important;
    border: 2px solid #0aff9d !important;
    color: white !important;
}

.stButton>button {
    background-color: #00ff66 !important;
    color: black !important;
    font-weight: 800 !important;
    border-radius: 6px;
    border: 2px solid white !important;
}

.title-text {
    font-size: 48px;
    font-weight: 900;
    color: #00ffcc;
    text-align: center;
}

.result-text {
    font-size: 32px;
    font-weight: 800;
    color: #00ffcc;
}

.info-note {
    color: #aaaaaa;
    font-size: 14px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ============================
#      HELPER FUNCTIONS
# ============================

extra_bad_words = {
    "fuck": "please calm down",
    "fucker": "rude person",
    "fucking": "very",
    "bitch": "unkind person",
    "slut": "person",
    "skank": "offensive term",
    "cocksucker": "offensive insult",
    "asshole": "rude individual",
    "stupid": "not thoughtful",
    "hate": "dislike",
    "idiot": "uninformed person",
}

def clean_sentence(sentence):
    words = sentence.split()
    cleaned = []
    for w in words:
        if w.lower() in extra_bad_words:
            cleaned.append(extra_bad_words[w.lower()])
        else:
            cleaned.append(w)
    return " ".join(cleaned)

def remove_toxic_words(sentence):
    words = sentence.split()
    cleaned = [w for w in words if w.lower() not in extra_bad_words]
    return " ".join(cleaned)

def generate_ticket(history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "AI SAFETY WARNING TICKET", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Issued: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detected Toxic Messages:", ln=True)

    pdf.set_font("Arial", "", 12)
    for msg in history:
        pdf.cell(0, 8, f"- {msg}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 8, "This ticket is auto-generated for safety review.", ln=True)

    return pdf.output(dest="S").encode("latin-1", "ignore")

# ============================
#      AVATARS
# ============================
red_bot = "assets/red bot.png"
blue_bot = "assets/blue bot.png"
yellow_bot = "assets/yellow bot.png"

def show_avatar(path, width=220):
    img = Image.open(path)
    st.image(img, width=width)

# ============================
#         TABS
# ============================
tab1, tab2, tab3 = st.tabs(["🤖 Chatbot", "📚 Toxic Word Dictionary", "🏓 Toxic Pong"])

# =======================================================
#                  TAB 1 – CHATBOT
# =======================================================
with tab1:
    st.markdown("<div class='title-text'>AI Comment Detector</div>", unsafe_allow_html=True)

    if "toxic_count" not in st.session_state:
        st.session_state.toxic_count = 0
    if "toxic_history" not in st.session_state:
        st.session_state.toxic_history = []

    user_input = st.text_input("Type your message:", "")

    if st.button("Analyze"):
        if user_input.strip() != "":
            vector = vectorizer.transform([user_input])
            pred = model.predict(vector)[0]

            if pred == 1:
                st.session_state.toxic_count += 1
                st.session_state.toxic_history.append(user_input)

                show_avatar(red_bot)
                st.markdown("<div class='result-text'>Toxic</div>", unsafe_allow_html=True)
                st.write("(Note: This result is based on dataset patterns and may not reflect actual intention.)")

                if st.session_state.toxic_count >= 3:
                    st.warning("3 Toxic Messages Detected — Issuing Warning Ticket...")

                    pdf_data = generate_ticket(st.session_state.toxic_history)
                    b64_pdf = base64.b64encode(pdf_data).decode()

                    download_html = f"""
                        <html>
                            <body>
                                <a id="auto_ticket"
                                   href="data:application/pdf;base64,{b64_pdf}"
                                   download="AI_Warning_Ticket.pdf"></a>

                                <script>
                                    document.getElementById('auto_ticket').click();
                                </script>
                            </body>
                        </html>
                    """

                    components.html(download_html, height=0)

                    st.session_state.toxic_count = 0
                    st.session_state.toxic_history = []

            else:
                show_avatar(blue_bot)
                st.markdown("<div class='result-text'>Safe Message</div>", unsafe_allow_html=True)
                st.write("(Based on dataset patterns, no harmful language detected.)")

# =======================================================
#      TAB 2 – TOXIC WORD DICTIONARY (NOW ML + RULE-BASED)
# =======================================================
with tab2:
    st.markdown("<div class='title-text'>Toxic Word Dictionary</div>", unsafe_allow_html=True)

    sentence = st.text_input("Enter a sentence:", "")
    st.markdown("<div class='info-note'>(ML detection + polite rewrite)</div>", unsafe_allow_html=True)

    if st.button("Rewrite Politely") and sentence.strip() != "":
        vector = vectorizer.transform([sentence])
        pred = model.predict(vector)[0]

        if pred == 1:
            polite_version = clean_sentence(sentence)
            st.error("This sentence was flagged toxic by the ML model.")
            st.success(f"Polite Rewrite: {polite_version}")
            st.write("(Note: Rewrite is based on dataset patterns + dictionary.)")
        else:
            st.success("This sentence is safe according to the ML model.")
            st.info("No rewrite needed.")

# =======================================================
#            TAB 3 – TOXIC PONG (NOW ML + RULE-BASED)
# =======================================================
with tab3:
    st.markdown("<div class='title-text'>Toxic Pong</div>", unsafe_allow_html=True)

    pong_sentence = st.text_input("Enter a sentence to clean:", "")
    st.markdown("<div class='info-note'>(ML detection + toxic-word removal)</div>", unsafe_allow_html=True)

    if st.button("Start Cleaning") and pong_sentence.strip() != "":
        vector = vectorizer.transform([pong_sentence])
        pred = model.predict(vector)[0]

        if pred == 1:
            cleaned = remove_toxic_words(pong_sentence)
            st.error("Toxic sentence detected by ML model.")
            st.success(f"Cleaned Sentence: {cleaned}")
        else:
            st.success("Sentence is already clean (ML model).")
            st.info("No cleaning needed.")
