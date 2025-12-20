import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.set_page_config(page_title="Health Myth vs Fact", layout="centered")

st.title("🩺 Health Myth vs Fact Detector")

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter a health-related statement")

if st.button("Verify"):
    if text.strip() == "":
        st.warning("Please enter a statement.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item() * 100

        if label == 1:
            st.success(f"✅ FACT ({confidence:.2f}%)")
        else:
            st.error(f"❌ MYTH ({confidence:.2f}%)")
