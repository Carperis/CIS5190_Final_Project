import os
import torch
import numpy as np
import glob
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from bert_model import TextClassificationDataset, BERTClassifier, get_device, load_news_data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report

def train_ensemble(models, train_dataloaders, optimizers, schedulers, device, writer, epoch):
    for model in models:
        model.train()
    avg_loss = 0
    for dataloader, model, optimizer, scheduler in zip(train_dataloaders, models, optimizers, schedulers):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (step + 1))
            writer.add_scalar(f'Training Loss/Model_{models.index(model)}', loss.item(), epoch * len(dataloader) + step)
        avg_loss += total_loss / len(dataloader)
    return avg_loss / len(models)

def evaluate_ensemble(models, dataloader, device):
    for model in models:
        model.eval()
    all_predictions = []
    actual_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = [model(input_ids=input_ids, attention_mask=attention_mask) for model in models]
            
            # Average ensemble outputs
            avg_outputs = torch.mean(torch.stack(outputs), dim=0)
            loss = nn.CrossEntropyLoss()(avg_outputs, labels)
            total_loss += loss.item()
            _, avg_preds = torch.max(avg_outputs, dim=1)
            
            # Majority voting ensemble outputs
            majority_preds = torch.mode(torch.stack([torch.max(output, dim=1)[1] for output in outputs]), dim=0)[0]
            
            all_predictions.extend(majority_preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(actual_labels, all_predictions)
    report = classification_report(actual_labels, all_predictions)
    return avg_loss, accuracy, report

def get_latest_checkpoints(checkpoint_dir, num_models):
    latest_checkpoints = []
    for i in range(num_models):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"bert_classifier_model_{i}_epoch_*.pth"))
        if not checkpoint_files:
            latest_checkpoints.append(None)
        else:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            latest_checkpoints.append(latest_checkpoint)
    return latest_checkpoints

def clean_previous_checkpoints(checkpoint_dir, num_models, keep_num):
    for i in range(num_models):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"bert_classifier_model_{i}_epoch_*.pth"))
        if len(checkpoint_files) > keep_num:
            checkpoint_files = sorted(checkpoint_files, key=os.path.getctime)
            for checkpoint_file in checkpoint_files[:-keep_num]:
                os.remove(checkpoint_file)

data_file = "./data/data20000.csv"
texts, labels = load_news_data(data_file)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

bert_model_name = "bert-base-uncased"
num_classes = 2
max_length = 32
batch_size = 16
num_epochs = 100
learning_rate = 2e-5
num_models = 5

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)

device = get_device()
models = [BERTClassifier(bert_model_name, num_classes).to(device) for _ in range(num_models)]
optimizers = [Adam(model.parameters(), lr=learning_rate) for model in models]
total_steps = len(train_dataset) * num_epochs // num_models
schedulers = [get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) for optimizer in optimizers]

train_dataloaders = [DataLoader(Subset(train_dataset, np.random.choice(len(train_dataset), len(train_dataset) // num_models, replace=True)), batch_size=batch_size, shuffle=True) for _ in range(num_models)]
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

checkpoint_dir = f"bert_{num_models}bagging_checkpoints/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
latest_checkpoints = get_latest_checkpoints(checkpoint_dir, num_models)
if all(checkpoint is not None for checkpoint in latest_checkpoints):
    for i, checkpoint in enumerate(latest_checkpoints):
        models[i].load_state_dict(torch.load(checkpoint))
        print(f"Loaded model {i} from {checkpoint}")
    epoch = int(latest_checkpoints[0].split("_")[-1].split(".")[0])
else:
    print("No checkpoint found, training from scratch")
    epoch = 0

from datetime import datetime
now = datetime.now()
writer = SummaryWriter(log_dir=f"runs/bert_{num_models}bagging_training_{now.strftime('%Y%m%d%H%M%S')}")

while epoch < num_epochs:
    clean_previous_checkpoints(checkpoint_dir, num_models, 3)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_ensemble(models, train_dataloaders, optimizers, schedulers, device, writer, epoch)
    val_loss, accuracy, report = evaluate_ensemble(models, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(report)
    writer.add_scalar("Train Epoch Loss", train_loss, epoch)
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"{checkpoint_dir}/bert_classifier_model_{i}_epoch_{epoch+1}.pth")
    epoch += 1

writer.close()
