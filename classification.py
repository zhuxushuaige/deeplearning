import os
import ssl
import tqdm
import torch.nn as nn
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CURL_CA_BUNDLE"] = ""  # 禁用 cert 验证
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import BertTokenizer
model_id = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_id)
import glob
import re
from collections import Counter
import numpy as np
def load_20newsgroups_manual(data_home, subset="train", categories=None):
    base = os.path.join(data_home, "20news-bydate", f"20news-bydate-{subset}")
    all_files = []
    all_targets = []
    target_names = categories if categories else sorted(os.listdir(base))
    
    for i, cat in enumerate(target_names):
        cat_path = os.path.join(base, cat)
        if not os.path.isdir(cat_path):
            continue
        for filepath in glob.glob(os.path.join(cat_path, "*")):
            with open(filepath, "rb") as f:
                content = f.read()
            # 尝试解码（原始数据是 latin-1 / iso-8859-1）
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1")
            all_files.append(text)
            all_targets.append(i)
    
    return {
        "data": all_files,
        "target": all_targets,
        "target_names": target_names
    }

# 使用
data_folder = "d:\\deeplearning\\scikit_learn_data"
categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

train_data_raw = load_20newsgroups_manual(data_folder, subset="train", categories=categories)
test_data_raw = load_20newsgroups_manual(data_folder, subset="test", categories=categories)


def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9(),.!?\\'`]", " ", text)
    text = re.sub(r"([,.!?\\'`])", r" \\1 ", text)
    tokens = text.strip().split()
    return tokens

# 计算每篇文档的词元数量
train_text_lengths = [len(basic_tokenize(text)) for text in train_data_raw["data"]]

word_counts = Counter()
for text in train_data_raw["data"]:
    word_counts.update(basic_tokenize(text))
    
##制作字典
def build_vocab(word_counts, min_freq=5):
    vocab = {"<pad>":0, "<unk>":1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(word_counts, min_freq=5)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from torch.utils.data import Dataset
class BertTextClassificationDataset(Dataset):
    def __init__(self,text,labels,tokenizer,max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text = text
        self.labels = labels
        self.processed_data = []
        for i in tqdm.tqdm(range(len(self.text)), desc="Processing data"):
            encoding = self.tokenizer(self.text[i],add_special_tokens=True,truncation=False)
            input_ids = encoding["input_ids"]
            if len(input_ids) < self.max_length:
                self.processed_data.append({"input_ids":input_ids,"label":labels[i]})
            else:
                stride = int(self.max_length * 0.8)
                for start in range(0, len(input_ids), stride):
                    end = start + self.max_length
                    self.processed_data.append({"input_ids":input_ids[start:end],"label":labels[i]})
                    if end >= len(input_ids):
                        break
    def __len__(self):
        return len(self.processed_data)
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
def bert_collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    batch_input_ids,attention_mask,batch_labels = [],[],[]
    for item in batch:
        input_ids = item["input_ids"]
        padding_length = max_len - len(input_ids)
        batch_input_ids.append(input_ids + [tokenizer.pad_token_id]*padding_length)
        attention_mask.append([1]*len(input_ids) + [0]*padding_length)
        batch_labels.append(item["label"])
    return {
        "input_ids":torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask":torch.tensor(attention_mask, dtype=torch.long),
        "labels":torch.tensor(batch_labels, dtype=torch.long)
    }
    
from torch.utils.data import DataLoader
train_dataset = BertTextClassificationDataset(train_data_raw["data"], train_data_raw["target"], tokenizer, max_length=128)
test_dataset = BertTextClassificationDataset(test_data_raw["data"], test_data_raw["target"], tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=bert_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=bert_collate_fn)

    

from transformers import BertModel
class BertTextClassifier(nn.Module):
    def __init__(self,model_id,num_classes,freeze_bert=False):
        super(BertTextClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(model_id)
        self.classifier = nn.Linear(self.bert.config.hidden_size,num_classes)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    def forward(self,input_ids,attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs_pooled = outputs.pooler_output
        logits = self.classifier(outputs_pooled)
        return logits
    
import os
import json

class Trainer:
    def __init__(self,model,optimizer,criterion,train_loader,test_loader,device,save_path):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_path = save_path
        self.model.to(self.device)
        self.best_accuracy = 0.0
        os.makedirs(self.save_path, exist_ok=True)
        # 用于记录历史数据
        self.train_losses = []
        self.val_accuracies = []
    def _run_epoch(self,epoch):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm.tqdm(self.train_loader,desc=f"Epoch {epoch+1}[Training]"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    def _evaluate(self,epoch):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_loader,desc=f"Epoch {epoch+1}[Evaluating]"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        return accuracy
    
    def _save_checkpoint(self, epoch, val_accuracy):
        self.best_accuracy = val_accuracy
        save_path = os.path.join(self.save_path, "best_model")
        self.model.bert.save_pretrained(save_path)  # 使用 Hugging Face 的保存方法
        classifier_path = os.path.join(save_path, "classifier.pth")
        torch.save(self.model.classifier.state_dict(), classifier_path)  # 保存分类器权重
        print(f"新最佳模型已保存! Epoch: {epoch+1}, 验证集准确率: {val_accuracy:.4f}")
    def train(self, num_epochs,tokenizer,label_map):
        for epoch in range(num_epochs):
            self.train_losses = []
            self.val_accuracies = []
            avg_loss = self._run_epoch(epoch)
            accuracy = self._evaluate(epoch)
            self.train_losses.append(avg_loss)
            self.val_accuracies.append(accuracy)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Accuracy={accuracy:.4f}")
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self._save_checkpoint(epoch, accuracy)
        vocab_path = os.path.join(self.save_path, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
           json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=4)
           
        labels_path = os.path.join(self.save_path, 'label_map.json')
        with open(labels_path, 'w', encoding='utf-8') as f:
           json.dump(label_map, f, ensure_ascii=False, indent=4)
        print(f"词典 ({vocab_path}) 和标签映射 ({labels_path}) 已保存。")
        return self.train_losses,self.val_accuracies
    
h_params = {
    "model_id": model_id,
    "vocab_size": len(tokenizer.vocab),
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_classes": len(train_data_raw["target_names"]),
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "save_path": "./text_classifier_checkpoints_bert",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "num_layers": 1,
    "dropout": 0.3,
    "bidirectional": False
}

model = BertTextClassifier(h_params["model_id"], h_params["num_classes"]).to(h_params["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=h_params["learning_rate"])
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, h_params["device"], h_params["save_path"])
label_map = {idx: name for idx, name in enumerate(train_data_raw["target_names"])}
train_losses, val_accuracies = trainer.train(h_params["num_epochs"],tokenizer,label_map)
import matplotlib.pyplot as plt
        
def plot_history(train_losses, val_accuracies, title_prefix=""):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制训练损失曲线
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss')
    ax1.set_title(f'{title_prefix} Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制验证集准确率曲线
    ax2.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    ax2.set_title(f'{title_prefix} Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    plt.suptitle(f'{title_prefix} Training and Validation Metrics', fontsize=16)
    plt.show()

# 调用绘图函数
plot_history(train_losses, val_accuracies, title_prefix="BERT Classifier")           