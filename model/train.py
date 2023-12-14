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
text_path = 'summary_page/model/data/C4_small/texts.txt'
label_path = 'summary_page/model/data/C4_small/labels.txt'

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
        item['tgt_attention_mask'] = torch.tensor(self.labels['attention_mask'][idx])
        return item
    def __len__(self):
        return len(self.encodings.input_ids)
    
def generate_mask(src_attn_mask, tgt_attn_mask):
        src_attn_mask = src_attn_mask.to(dtype=torch.float32)
        mask_min_value = torch.finfo(torch.float32).min
        src_attn_mask = 1.0 - src_attn_mask
        src_attn_mask = src_attn_mask.masked_fill(
            src_attn_mask.to(torch.bool),
            mask_min_value,
        )

        tgt_attn_mask = torch.Tensor([[1,1,1,0,0]]).int()
        tgt_seq_length = tgt_attn_mask.size(1)
        tgt_attn_mask = tgt_attn_mask[:,None,:].expand(-1,-1,tgt_seq_length)
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_length, tgt_seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(torch_device)
        tgt_attn_mask = tgt_attn_mask & nopeak_mask
        tgt_attn_mask = 1.0 - tgt_attn_mask
        tgt_attn_mask = tgt_attn_mask.masked_fill(
            tgt_attn_mask.to(torch.bool),
            mask_min_value,
        )
        return src_attn_mask, tgt_attn_mask
    
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
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            src_attention_mask = batch['attention_mask'].to(device)
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)
            src_attention_mask, tgt_attention_mask = generate_mask(src_attention_mask, tgt_attention_mask)
            labels = batch['labels'].to(device)
            # process
            _, outputs = model(input_ids,labels, src_attention_mask, tgt_attention_mask)
            loss = criterion(outputs[:,:-1,:].permute(0,2,1).contiguous(), labels[:,1:])
            print(loss)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")
    src_vocab_size = len(tokenizer)
    tgt_vocab_size = src_vocab_size
    config = json.load(open("summary_page/model/config/configPEGASUS_X.json"))
    set_seed(config['seed'])
    pegasus_x = PegasusXModel(src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size, 
                              d_model = config["d_model"], num_heads = config["num_heads"], 
                              src_num_layers = config["src_num_layers"], tgt_num_layers = config["tgt_num_layers"], 
                              block_size = config["block_size"], num_global_tokens = config["num_global_tokens"], 
                              d_ff = config["decoder_ff"], dropout = config["dropout"], 
                              src_padded_seq_len = config["src_padded_seq_len"], 
                              tgt_padded_seq_len = config["tgt_padded_seq_len"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(pegasus_x.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    train_PegasusX(pegasus_x, tokenizer, criterion, optimizer, config)