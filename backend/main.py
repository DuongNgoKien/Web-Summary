import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from web_scraping import web_scrapping
from transformers import AutoTokenizer, PegasusXForConditionalGeneration
from contextlib import asynccontextmanager

class Item(BaseModel):
    url: str

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\DuongNgoKien\\Downloads\\results\\results\\checkpoint-376500", use_safetensors=True)
    model = PegasusXForConditionalGeneration.from_pretrained("C:\\Users\\DuongNgoKien\\Downloads\\results\\results\\checkpoint-376500", use_safetensors=True)
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
        inputs = ml_model['tokenizer'](text, max_length=1024, truncation=True, return_tensors='pt')
        summary_ids = ml_model['model'].generate(inputs["input_ids"], num_beams=4, max_length=50)
        message = ml_model['tokenizer'].batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        message = "Can't scrape this website."
    return {"message": message}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)