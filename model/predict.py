import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("./saved_model")
model = DistilBertForSequenceClassification.from_pretrained("./saved_model")

def classify(statement):
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()

    return ("Fact" if label == 1 else "Myth"), round(confidence * 100, 2)

text = "Drinking hot water kills coronavirus"
result, conf = classify(text)

print("Prediction:", result)
print("Confidence:", conf, "%")
