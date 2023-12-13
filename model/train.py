import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pegasus_x import PegasusXModel
from transformers import AutoTokenizer
import numpy as np
import random
import os
import json
from tqdm import tqdm

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = './results'
text_path = 'model/data/C4_small/texts.txt'
label_path = 'model/data/C4_small/labels.txt'
num_epochs = 450

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_data(text_path, label_path):
    with open(text_path, 'r', encoding='utf-8') as fp:
        train_texts = fp.read().split('\n')
    with open(label_path, 'r', encoding='utf-8') as fp:
        train_labels = fp.read().split('\n')
    return train_texts, train_labels

class PegasusDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.encodings.input_ids)
    
def train_PegasusX(model, tokenizer, criterion, optimizer, config):
    train_texts, train_labels = load_data(text_path, label_path)
    inputs = tokenizer(train_texts, return_tensors='pt', padding='max_length', max_length=16384, truncation=True)
    labels = tokenizer(train_labels, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
    dataset = PegasusDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    for _ in range(config['epochs']):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
            
            loss = criterion(outputs)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")
    src_vocab_size = len(tokenizer)
    tgt_vocab_size = src_vocab_size
    config = json.load(open("model/config/configPEGASUS_X.json"))
    set_seed(config['seed'])
    pegasus_x = PegasusXModel(src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size, d_model = 768,
                               num_heads = 12, src_num_layers = 12, tgt_num_layers = 12, 
                 block_size = 512, num_global_tokens = 128, d_ff = 3072, src_padded_seq_len = 16384, 
                 tgt_padded_seq_len = 256, dropout = 0.1, masked_prediction=False)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(pegasus_x.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    train_PegasusX(pegasus_x, tokenizer, criterion, optimizer, config)
