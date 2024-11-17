import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from bert_model import TextClassificationDataset, BERTClassifier, get_device, load_news_data, get_latest_checkpoint, evaluate

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
    return "fox" if preds.item() == 1 else "nbc"

data_file = "./data/data.csv"
texts, labels = load_news_data(data_file)

bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 32
batch_size = 16

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = get_device()
model = BERTClassifier(bert_model_name, num_classes).to(device)

checkpoint_dir = "bert_checkpoints/"
checkpoint = get_latest_checkpoint(checkpoint_dir)
if checkpoint is not None:
    model.load_state_dict(torch.load(checkpoint))
    print(f"Loaded model from {checkpoint}")

accuracy, report = evaluate(model, train_dataloader, device)
print(f"Validation Accuracy: {accuracy:.4f}")
print(report)
