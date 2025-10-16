from flask import Flask, render_template, request
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

# ---------- Load Model and Tokenizer ----------
MODEL_PATH = "D:\MY DATA\Desktop\projects\DL project"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # set to evaluation mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------- Prediction Function ----------
def predict_toxicity(text):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

    # Define class labels (Jigsaw has 6)
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # Combine labels with probabilities
    results = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # Pick all toxic categories where probability > 0.5
    toxic_labels = [lbl for lbl, p in results.items() if p > 0.5]

    if toxic_labels:
        summary = f"Toxic → {', '.join(toxic_labels)}"
        color = "red"
    else:
        summary = "Not Toxic"
        color = "green"

    return summary, results, color


# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    details = None
    color = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("text", "")
        if user_input.strip():
            prediction, details, color = predict_toxicity(user_input)

    return render_template("index.html",
                           prediction=prediction,
                           details=details,
                           color=color,
                           user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
