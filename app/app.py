import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.title("🩺 Health Myth vs Fact Detector")

tokenizer = DistilBertTokenizerFast.from_pretrained("../model/saved_model")
model = DistilBertForSequenceClassification.from_pretrained("../model/saved_model")

text = st.text_area("Enter health statement")

if st.button("Check"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()

    if label == 1:
        st.success(f"FACT ✅ ({confidence*100:.2f}%)")
    else:
        st.error(f"MYTH ❌ ({confidence*100:.2f}%)")
