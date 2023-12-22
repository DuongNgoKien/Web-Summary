import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from web_scraping import web_scrapping
from transformers import AutoTokenizer
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import json
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from model.pegasus_x import PegasusXModel
from model.evaluatePegasusX import generate_predictions
from model.train import _resume_checkpoint

class Item(BaseModel):
    url: str

ml_model = {}
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    load_dotenv()
    config = json.load(open("model/config/configPEGASUS_X.json"))
    MODEL_PATH = os.getenv('MODEL_PATH')
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")
    src_vocab_size = len(tokenizer)
    tgt_vocab_size = src_vocab_size
    max_length_input = 6400
    model = PegasusXModel(src_vocab_size = src_vocab_size, tgt_vocab_size = tgt_vocab_size, 
                              d_model = config["d_model"], num_heads = config["num_heads"], 
                              src_num_layers = config["src_num_layers"], tgt_num_layers = config["tgt_num_layers"], 
                              block_size = config["block_size"], num_global_tokens = config["num_global_tokens"], 
                              d_ff = config["decoder_ff"], dropout = config["dropout"], 
                              src_padded_seq_len = int(max_length_input))
    model.to(torch_device)
    _resume_checkpoint(MODEL_PATH, model, optimizer=None)

    ml_model['model'] = model
    ml_model['tokenizer'] = tokenizer
    yield
    # Clean up the ML models and release the resources
    ml_model.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/get_summary")
async def get_summary(item: Item):
    text = web_scrapping(item.url)
    if text is not None:
        inputs = ml_model['tokenizer']([text], max_length=6400, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(torch_device)
        src_attention_mask = inputs['attention_mask'].to(torch_device)
        summary_ids = generate_predictions(ml_model['model'], input_ids, ml_model['tokenizer'], 0, 1, src_attention_mask, max_length=256) 

        message = ml_model['tokenizer'].batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        message = "Can't scrape this website."
    return {"message": message}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)