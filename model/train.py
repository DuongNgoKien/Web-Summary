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
import argparse

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_dir = 'summary_page/model/checkpoint'
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

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PEGASUS_X hyperparameters")
    parser.add_argument("--src_len", help="src padded sequence length")
    parser.add_argument("--tgt_len", help="tgt padded sequence length")
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    return parser.parse_args()

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
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx].clone().detach()
        item['tgt_attention_mask'] = self.labels['attention_mask'][idx].clone().detach()
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

def _save_checkpoint(epoch, model, optimizer, config):
    checkpoint = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    checkpoint['model'] = model.state_dict()
    filename = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
    if epoch > 1:
        os.remove(os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch - 1}.pth'))
    torch.save(checkpoint, filename)

def _resume_checkpoint(resume_path, model, optimizer):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    epoch = checkpoint['epoch'] + 1
    print('Starting at epoch: ' + str(epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return epoch
    
def train_PegasusX(start_epoch, model, tokenizer, criterion, optimizer, config, args):
    train_texts, train_labels = load_data(text_path, label_path)
    inputs = tokenizer(train_texts, return_tensors='pt', padding='max_length', max_length=int(args.src_len), truncation=True)
    labels = tokenizer(train_labels, return_tensors='pt', padding='max_length', max_length=int(args.tgt_len), truncation=True)
    dataset = PegasusDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model.train()

    for epoch in range(start_epoch, start_epoch + config['epochs']):
        loop = tqdm(loader, leave=True)
        i = 0
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(torch_device)
            src_attention_mask = batch['attention_mask'].to(torch_device)
            tgt_attention_mask = batch['tgt_attention_mask'].to(torch_device)
            src_attention_mask, tgt_attention_mask = generate_mask(src_attention_mask, tgt_attention_mask)
            labels = batch['labels'].to(torch_device)
            # process
            _, outputs = model(input_ids,labels, src_attention_mask, tgt_attention_mask)
            loss = criterion(outputs[:,:-1,:].permute(0,2,1).contiguous(), labels[:,1:])
            loss.backward()
            optimizer.step()
            i += 1
            if i == 500:
              break
        _save_checkpoint(epoch, model, optimizer, config)

if __name__ == "__main__":
    args = get_arguments()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
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
                              src_padded_seq_len = int(args.src_len), 
                              tgt_padded_seq_len = int(args.tgt_len))
    pegasus_x.to(torch_device)
    
    optimizer = optim.Adam(pegasus_x.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if args.resume:
        start_epoch = _resume_checkpoint(args.resume, pegasus_x, optimizer)
    else:
        start_epoch = 1
    
    train_PegasusX(start_epoch, pegasus_x, tokenizer, criterion, optimizer, config, args)