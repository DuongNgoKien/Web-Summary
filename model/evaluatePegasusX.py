import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_dataset
from pegasus_x import PegasusXModel
from train import PegasusDataset, generate_mask, _resume_checkpoint
from tqdm import tqdm
import json
import evaluate
import argparse

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_checkpoint = 'model/checkpoint/finetune/checkpoint-PubMed.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PEGASUS_X hyperparameters")
    parser.add_argument("--start_idx", help="start id of test dataset")
    parser.add_argument("--end_idx", help="start id of test dataset")
    return parser.parse_args()


def generate_predictions(model, input, tokenizer, start_token, end_token, src_attn_mask, max_length=256, temperature=1.0):
    model.eval()
    
    src_mask, _ = generate_mask(src_attn_mask, tgt_attn_mask=None)

    # Initial target sequence with the start token
    target_sequence = torch.zeros((1, max_length), dtype=torch.int).to(torch_device)
    target_sequence[0,0] = start_token  # Assuming start_token is defined
    tgt_attn_mask = torch.zeros((1, max_length), dtype=torch.int).to(torch_device)

    with torch.no_grad():
        for index in range(max_length):

            # Generate attention mask for the target sequence
            tgt_attn_mask[0, index] = 1
            _, tgt_mask = generate_mask(src_attn_mask = None, tgt_attn_mask = tgt_attn_mask)

            # Make the prediction for the next token
            _, output = model(input, target_sequence, src_mask, tgt_mask)

            # Apply temperature to the logits for diversity
            logits = output[:, -1, :] / temperature

            # Sample the next token using the logits
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)

            # Check for the end token to stop generation
            if next_token.item() == end_token:  # Assuming end_token is defined
                break

            # Append the next token to the target sequence
            if index < max_length-1:
              target_sequence[0,index+1] = next_token

    # Convert the predicted indices back to text (replace with your own logic)
    predicted_text = tokenizer.decode(target_sequence[0,1:index+1].tolist())
    return predicted_text

if __name__ == "__main__":
    config = json.load(open("model/config/configPEGASUS_X.json"))
    args = get_arguments()

    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")
    start_token = 0
    end_token = 1
    src_vocab_size = len(tokenizer)
    tgt_vocab_size = src_vocab_size
    dataset = load_dataset("ccdv/pubmed-summarization", streaming=True)
    test_texts = []
    test_labels = []
    for index, sample in enumerate(dataset['test']):
        if index >= int(args.start_idx) and index < int(args.end_idx):
            test_texts.append(sample['article'])
            test_labels.append(sample['abstract'])
    
    max_length_input = 6400
    max_length_output = 256

    pegasus_x = PegasusXModel(src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size, 
                              d_model = config["d_model"], num_heads = config["num_heads"], 
                              src_num_layers = config["src_num_layers"], tgt_num_layers = config["tgt_num_layers"], 
                              block_size = config["block_size"], num_global_tokens = config["num_global_tokens"], 
                              d_ff = config["decoder_ff"], dropout = config["dropout"], 
                              src_padded_seq_len = int(max_length_input))
    
    inputs = tokenizer(test_texts, return_tensors='pt', padding='max_length', max_length=max_length_input, truncation=True)
    labels = tokenizer(test_labels, return_tensors='pt', padding='max_length', max_length=max_length_output, truncation=True)
    dataset = PegasusDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    pegasus_x.to(torch_device)
    _resume_checkpoint(path_checkpoint, pegasus_x, optimizer=None)
    
    predictions = []
    rouge = evaluate.load('rouge')
    loop = tqdm(loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(torch_device)
        src_attention_mask = batch['attention_mask'].to(torch_device)
        # process
        outputs = generate_predictions(pegasus_x, input_ids, tokenizer, start_token, end_token, src_attention_mask, max_length=max_length_output) 
        predictions.append(outputs)
    print(test_labels[1])
    rouge_score = rouge.compute(predictions=predictions, references=test_labels)
    print(rouge_score)

        