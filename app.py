import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="QuoteGenie",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------
# Custom CSS (UI Upgrade)
# ------------------------------
st.markdown("""
<style>

.main-title{
    font-size:45px;
    font-weight:700;
    text-align:center;
    background: linear-gradient(90deg,#667eea,#764ba2);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.quote-card{
    background: #f8f9fa;
    padding:25px;
    border-radius:15px;
    border-left:6px solid #667eea;
    font-size:22px;
    font-style:italic;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

.copy-btn{
    background:#667eea;
    color:white;
    padding:10px 18px;
    border-radius:8px;
    text-decoration:none;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load resources
# ------------------------------
@st.cache_resource
def load_resources():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl","rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl","rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# ------------------------------
# Predict next word
# ------------------------------
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding="pre")

    preds = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(preds, axis=-1)[0]

    predicted_word = tokenizer.index_word.get(predicted_index, "")

    return predicted_word


# ------------------------------
# Generate quote
# ------------------------------
def generate_quote(seed_text, num_words):

    generated = seed_text

    for _ in range(num_words):

        next_word = predict_next_word(generated)

        if next_word == "":
            break

        generated += " " + next_word

    return generated


# ------------------------------
# UI
# ------------------------------

st.markdown('<p class="main-title">🧠 QuoteGenie</p>', unsafe_allow_html=True)

st.write(
"Start a sentence and let AI generate a **full inspirational quote**."
)

with st.form("quote_form"):

    user_input = st.text_input(
        "✍️ Start your quote",
        placeholder="Example: Life is"
    )

    num_words = st.slider(
        "Number of words to generate",
        1, 25, 12
    )

    submit = st.form_submit_button("✨ Generate Quote")

import time
import streamlit.components.v1 as components

if submit:

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("AI is thinking..."):
            quote = generate_quote(user_input, num_words)

        quote_placeholder = st.empty()

        typed_text = ""

        # Typing animation
        for char in quote:
            typed_text += char

            quote_placeholder.markdown(f"""
            <div style="
                background:#f8f9fa;
                padding:25px;
                border-radius:15px;
                border-left:6px solid #667eea;
                box-shadow:0px 4px 15px rgba(0,0,0,0.1);
                font-size:22px;
                font-style:italic;
            ">
            {typed_text}
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.02)

        # Copy button (real clipboard)
        components.html(f"""
        <div style="text-align:center;margin-top:15px;">
        <button onclick="copyText()" style="
            background:#667eea;
            color:white;
            border:none;
            padding:10px 18px;
            border-radius:8px;
            cursor:pointer;
            font-weight:600;
        ">
        📋 Copy Quote
        </button>
        </div>

        <script>
        function copyText() {{
            navigator.clipboard.writeText(`{quote}`);
            alert("Quote copied to clipboard!");
        }}
        </script>
        """, height=80)


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Built with ❤️ using LSTM + Streamlit")