
# pip install transformers datasets torch scikit-learn pandas

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configurations
MODEL_NAME = 'roberta-base'
BATCH_SIZE = 42
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128

# Download dataset
print("Downloading Jigsaw Toxic Comment Classification Challenge dataset...")
path = kagglehub.dataset_download("julian3833/jigsaw-toxic-comment-classification-challenge")
print("Path to dataset files:", path)

# Load the dataset
train_file = os.path.join(path, "train.csv")
test_file = os.path.join(path, "test.csv")

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

print("\nTrain data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

dataset= train_data

# Basic preprocessing
def clean_text(text):
    return text.lower()

# Initialize tokenizer
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Basic preprocessing
dataset['comment_text'] = dataset['comment_text'].str.lower()

# Tokenize
tokenized_inputs = tokenizer(
    list(dataset['comment_text']),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN
)

# Labels
labels = list(dataset['toxic'])

# Split the dataset texts and labels first
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset['comment_text'].tolist(),
    dataset['toxic'].tolist(),
    test_size=0.1,
    random_state=42
)

# Tokenize separately
train_encodings = tokenizer(
    train_texts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN
)
val_encodings = tokenizer(
    val_texts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN
)

# Define Dataset class properly
class ToxicCommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Build datasets
train_dataset = ToxicCommentDataset(train_encodings, train_labels)
val_dataset = ToxicCommentDataset(val_encodings, val_labels)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model = model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

# Training function
def train_epoch(model, dataloader, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0
    preds = []
    labels = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits.squeeze()
        loss = loss_fn(logits, batch['labels'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(batch['labels'].detach().cpu().numpy())

    preds = np.array(preds) > 0.5
    accuracy = accuracy_score(labels, preds)
    return total_loss / len(dataloader), accuracy

# Validation function
def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits.squeeze()
            loss = loss_fn(logits, batch['labels'])

            total_loss += loss.item()
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    preds = np.array(preds) > 0.5
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return total_loss / len(dataloader), accuracy, f1

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# ------------------ TRAINING LOOP ------------------
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")

    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, lr_scheduler, device)
    val_loss, val_accuracy, val_f1 = eval_model(model, val_loader, device)

    # Save history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}")
    print(f"Val Loss   = {val_loss:.4f}, Val Acc   = {val_accuracy:.4f}, Val F1 = {val_f1:.4f}")

# ------------------ PLOTTING SECTION ------------------

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, EPOCHS+1))
plt.legend()
plt.grid()
plt.savefig('training_validation_loss.png')
plt.close()
print("\nTraining and validation loss plot saved to 'training_validation_loss.png'")

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS+1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, EPOCHS+1), val_accuracies, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(1, EPOCHS+1))
plt.legend()
plt.grid()
plt.savefig('training_validation_accuracy.png')
plt.close()
print("Training and validation accuracy plot saved to 'training_validation_accuracy.png'")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Collect predictions
model.eval()
true_labels = []
pred_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits.squeeze()
        preds = torch.sigmoid(logits) > 0.5
        true_labels.extend(batch['labels'].cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())


# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
print("\nConfusion matrix plot saved to 'confusion_matrix.png'")

# Save model
model.save_pretrained("./toxicity-roberta")
tokenizer.save_pretrained("./toxicity-roberta")

print("Training completed and model saved!")









