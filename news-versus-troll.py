import re
import random
import numpy as np
import torch
import torch.nn as nn
import time
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

MAX_LEN = 200
BATCH_SIZE = 64
EMB_DIM = 100
HID_DIM = 128
NUM_CLASSES = 2
DEVICE = "cuda"
SEEDS = [213, 141, 45, 180, 12]
N_PER_CLASS = 2500
TOKENIZER_REGEX = re.compile(r"[A-Za-z']+")

# load dataset
ds = load_dataset("zkpbeats/reddit_ds_260222", split="train")
ds = ds.filter(lambda example: example["communityName"].lower() in {"r/news", "r/shitposting"})
ds = ds.map(lambda example: {"label": int(example["communityName"].lower()=="r/news")})
ds = ds.filter(lambda example: len(TOKENIZER_REGEX.findall(example["text"])) > 50)
news_full  = ds.filter(lambda ex: ex["label"]==1)
troll_full = ds.filter(lambda ex: ex["label"]==0)

# balanced sample
num_samples = min(len(news_full), len(troll_full), N_PER_CLASS)
news = news_full.shuffle(seed=88).select(range(num_samples))
troll = troll_full.shuffle(seed=88).select(range(num_samples))

small = concatenate_datasets([news, troll]).shuffle(seed=88)
split = small.train_test_split(test_size=0.2, seed=88)
train_raw, test_raw = split["train"], split["test"]

# tokenize and create vocab
def tokenize(text):

    return TOKENIZER_REGEX.findall(text.lower())

counter = Counter()
for example in train_raw:
    counter.update(tokenize(example["text"]))

vocab = {"<PAD>":0, "<UNK>":1}
for word, freq in counter.items():
    if freq >= 5:
        vocab[word] = len(vocab)

# generate glove embeddings
emb_index = {}
with open("glove.6B.100d.txt","r",encoding="utf8") as file:
    for line in file:
        parts = line.strip().split()
        emb_index[parts[0]] = np.array(parts[1:], dtype=np.float32)

matrix = np.random.normal(0,0.6,(len(vocab), EMB_DIM)).astype(np.float32)
matrix[vocab["<PAD>"]] = np.zeros(EMB_DIM, dtype=np.float32)
for word, index in vocab.items():
    if word in emb_index:
        matrix[index] = emb_index[word]
embedding_matrix = torch.tensor(matrix)

# pytorch dataset
class NewsTrollsDS(Dataset):
    def __init__(self, split):
        self.data = split
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text  = self.data[idx]["text"]
        label = self.data[idx]["label"]
        tokens = tokenize(text)[:MAX_LEN]
        indices = [vocab.get(token, 1) for token in tokens]
        indices += [0]*(MAX_LEN - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(NewsTrollsDS(train_raw), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(NewsTrollsDS(test_raw),  batch_size=BATCH_SIZE)

# actual models
class BaseLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding  = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(HID_DIM*2, NUM_CLASSES)
    def forward(self, inputs):
        inputs = inputs.to(DEVICE)
        embeddings = self.embedding(inputs)
        _,(hidden,_) = self.lstm(embeddings)
        hidden_concat = torch.cat((hidden[0], hidden[1]), dim=1)
        return self.fc(hidden_concat)

class GloveLSTM(BaseLSTM):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

class CNNBiLSTM(nn.Module):
    def __init__(self, kernel_sizes=[3,5], num_filters=100):
        super().__init__()
        self.embedding   = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(EMB_DIM, num_filters, kernel_size, padding=kernel_size//2)
            for kernel_size in kernel_sizes
        ])
        self.lstm  = nn.LSTM(num_filters*len(kernel_sizes),
                              HID_DIM, bidirectional=True, batch_first=True)
        self.fc    = nn.Linear(HID_DIM*2, NUM_CLASSES)
    def forward(self, inputs):
        inputs = inputs.to(DEVICE)
        embeddings = self.embedding(inputs).transpose(1,2)
        conv_outputs = [torch.relu(conv(embeddings)) for conv in self.convs]
        concat_conv = torch.cat(conv_outputs, 1).transpose(1,2)
        outputs, _ = self.lstm(concat_conv)
        representation = outputs.mean(1)
        return self.fc(representation)

class GloveCNNBiLSTM(CNNBiLSTM):
    def __init__(self, kernel_sizes=[3,5], num_filters=100):
        super().__init__(kernel_sizes, num_filters)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

class BiLSTMMHA(nn.Module):
    def __init__(self, heads=4, dropout=0.0):
        super().__init__()
        self.embedding  = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, bidirectional=True, batch_first=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=HID_DIM*2, num_heads=heads,
            dropout=dropout, batch_first=True
        )
        self.fc   = nn.Linear(HID_DIM*2, NUM_CLASSES)
    def forward(self, inputs):
        inputs = inputs.to(DEVICE)
        lstm_outputs, _ = self.lstm(self.embedding(inputs))
        padding_mask = inputs.eq(0)
        attn_output, _ = self.attn(lstm_outputs, lstm_outputs, lstm_outputs, key_padding_mask=padding_mask)
        representation = attn_output.mean(1)
        return self.fc(representation)

class GloveBiLSTMMHA(BiLSTMMHA):
    def __init__(self, heads=4, dropout=0.0):
        super().__init__(heads, dropout)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

class CNNBiLSTMMHA(nn.Module):
    def __init__(self, kernel_sizes=[3,5], num_filters=100, heads=4, dropout=0.0):
        super().__init__()
        self.embedding   = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(EMB_DIM, num_filters, kernel_size, padding=kernel_size//2)
            for kernel_size in kernel_sizes
        ])
        self.lstm  = nn.LSTM(num_filters*len(kernel_sizes),
                              HID_DIM, bidirectional=True, batch_first=True)
        self.attn  = nn.MultiheadAttention(
            embed_dim=HID_DIM*2, num_heads=heads,
            dropout=dropout, batch_first=True
        )
        self.fc    = nn.Linear(HID_DIM*2, NUM_CLASSES)
    def forward(self, inputs):
        inputs = inputs.to(DEVICE)
        embeddings = self.embedding(inputs).transpose(1,2)
        conv_outputs = [torch.relu(conv(embeddings)) for conv in self.convs]
        concat_conv = torch.cat(conv_outputs,1).transpose(1,2)
        lstm_outputs, _ = self.lstm(concat_conv)
        padding_mask = inputs.eq(0)
        attn_output, _ = self.attn(lstm_outputs, lstm_outputs, lstm_outputs, key_padding_mask=padding_mask)
        representation = attn_output.mean(1)
        return self.fc(representation)

class GloveCNNBiLSTMMHA(CNNBiLSTMMHA):
    def __init__(self, kernel_sizes=[3,5], num_filters=100, heads=4, dropout=0.0):
        super().__init__(kernel_sizes, num_filters, heads, dropout)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

# baseline models
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        self.rnn = nn.RNN(EMB_DIM, HID_DIM, batch_first=True)
        self.fc  = nn.Linear(HID_DIM, NUM_CLASSES)
    
    def forward(self, inputs):
        inputs = inputs.to(DEVICE)
        embeddings = self.embedding(inputs)
        _, hidden = self.rnn(embeddings)
        return self.fc(hidden.squeeze(0))

def train_baseline():
    # Simple RNN
    print("\nSimple RNN")
    start_time = time.time()

    rnn_runs = []
    for seed in SEEDS:
        set_seed(seed)
        rnn_model = SimpleRNN()
        train(rnn_model, train_loader, epochs=3, lr=1e-3)
        rnn_acc, rnn_f1 = evaluate(rnn_model, test_loader)
        rnn_runs.append((rnn_acc, rnn_f1))
        print(f" run seed={seed}:  Acc={rnn_acc:.4f}, F1={rnn_f1:.4f}")
        
    avg_rnn_acc = sum(acc for acc,_ in rnn_runs)/len(rnn_runs)
    avg_rnn_f1 = sum(f1 for _,f1 in rnn_runs)/len(rnn_runs)
    rnn_time = time.time() - start_time
    print(f"Average: Acc={avg_rnn_acc:.4f}, F1={avg_rnn_f1:.4f}")
    print(f"Training and evaluation time: {rnn_time:.2f} seconds")
    
    return [avg_rnn_acc, avg_rnn_f1, rnn_time]

# training
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train(model, loader, epochs=3, lr=1e-3):
    model.to(DEVICE).train()
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels.to(DEVICE))
            loss.backward()
            optimizer.step()

def train_mha(model, loader, epochs=8, lr=1e-4):
    model.to(DEVICE).train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = len(loader)*epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                num_warmup_steps=int(0.1*total_steps),
                num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

# evaluation
def evaluate(model, loader):
    model.to(DEVICE).eval()
    predictions, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs).argmax(1).cpu()
            predictions.extend(outputs.tolist()); labels_list.extend(labels.tolist())
    return accuracy_score(labels_list, predictions), f1_score(labels_list, predictions)


if __name__ == "__main__":
    # first, run the baseline models
    rnn_res = train_baseline()
    
    # main experiments
    experiments = [
        ("Base LSTM", BaseLSTM),
        ("BiLSTM + GloVe", GloveLSTM),
        ("CNN + BiLSTM", CNNBiLSTM),  
        ("CNN + BiLSTM + GloVe", GloveCNNBiLSTM),
        ("BiLSTM + MHA", BiLSTMMHA),
        ("BiLSTM + MHA + GloVe", GloveBiLSTMMHA),
        ("CNN + BiLSTM + MHA", CNNBiLSTMMHA),
        ("CNN + BiLSTM + MHA + GloVe", GloveCNNBiLSTMMHA),
    ]

    results = {}
    
    for name, Model in experiments:
        print(f"\n--{name}--")
        start_time = time.time()
        runs = []
        
        for seed in SEEDS:
            set_seed(seed)
            model = Model()
            if "MHA" in name:
                train_mha(model, train_loader)
            else:
                train(model, train_loader)
            accuracy, f1_score_val = evaluate(model, test_loader)
            runs.append((accuracy, f1_score_val))
            print(f"run seed={seed}: Acc={accuracy:.4f}, F1={f1_score_val:.4f}")
            
        avg_acc = sum(acc for acc,_ in runs)/len(runs)
        avg_f1 = sum(f1 for _,f1 in runs)/len(runs)
        total_time = time.time() - start_time
        print(f"Average: Acc={avg_acc:.4f}, F1={avg_f1:.4f}")
        print(f"Training and evaluation time: {total_time:.2f} seconds")
        
        results[name] = {"Accuracy": avg_acc, "F1-Score": avg_f1, "Time": total_time}
    
    print("\n\nResult Summary")
    print("Model - Accuracy - F1-Score - Time(s)")
    
    # baselines
    metrics = rnn_res
    print(f"Simple RNN                    | {metrics[0]:.4f} | {metrics[1]:.4f} | {metrics[2]:.2f}")
    
    # neural models
    for model, metrics in results.items():
        print(f"{model:28} | {metrics['Accuracy']:.4f} | {metrics['F1-Score']:.4f} | {metrics['Time']:.2f}")