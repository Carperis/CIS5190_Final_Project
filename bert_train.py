# reference: https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b

import os
import torch
import glob
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from bert_model import TextClassificationDataset, BERTClassifier, get_device, load_news_data, get_latest_checkpoint, evaluate
from torch.utils.tensorboard import SummaryWriter

def train(model, data_loader, optimizer, scheduler, device, writer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training")
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
        writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + step)
    return total_loss / len(data_loader)

def clean_previous_checkpoints(checkpoint_dir, keep_num):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if len(checkpoint_files) > keep_num:
        checkpoint_files = sorted(checkpoint_files, key=os.path.getctime)
        for checkpoint_file in checkpoint_files[:-keep_num]:
            os.remove(checkpoint_file)

bert_model_name = "bert-base-uncased"
num_classes = 2
max_length = 32
batch_size = 16
num_epochs = 20
learning_rate = 2e-5

# data_file = "./data/data.csv"
data_file = "./data/data20000.csv"
texts, labels = load_news_data(data_file)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = get_device()
model = BERTClassifier(bert_model_name, num_classes).to(device)

checkpoint_dir = "bert_checkpoints/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint = get_latest_checkpoint(checkpoint_dir)
if checkpoint is not None:
    model.load_state_dict(torch.load(checkpoint))
    epoch = int(checkpoint.split("_")[-1].split(".")[0])
    print(f"Loaded model from {checkpoint}")
else:
    print("No checkpoint found, training from scratch")
    epoch = 0

from datetime import datetime
now = datetime.now()
writer = SummaryWriter(log_dir=f"runs/bert_training_{now.strftime('%Y%m%d%H%M%S')}")

optimizer = Adam(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

print(f"Using device: {device}")
while epoch < num_epochs:
    clean_previous_checkpoints(checkpoint_dir, 3)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_dataloader, optimizer, scheduler, device, writer, epoch)
    val_loss, accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(report)
    writer.add_scalar("Training Epoch Loss", train_loss, epoch)
    writer.add_scalar("Validation Epoch Loss", val_loss, epoch)
    writer.add_scalar("Validation Accuracy", accuracy, epoch)
    torch.save(model.state_dict(), f"{checkpoint_dir}/bert_classifier_epoch_{epoch+1}.pth")
    epoch += 1

writer.close()
