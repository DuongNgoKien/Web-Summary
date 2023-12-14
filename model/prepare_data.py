import torch
import evaluate
import nltk
from tqdm import tqdm
from datasets import load_dataset

text_path = '/kaggle/working/texts.txt'
label_path = '/kaggle/working/labels.txt'

def process_data(rouge, document):
    labels = ""
    l_sentences = []
    for s in document.split("\n"):
        s = nltk.sent_tokenize(s)
        l_sentences.extend(s)
    len_doc = len(l_sentences)
    if len_doc >= 40: return None, None
    n_mask = int(len_doc / 5)
    if n_mask == 0: n_mask =1
    # Create a matrix to save rouge-f1 score between each sentence and the rest of the document
    ma_rouge = torch.zeros(size=(len_doc, len_doc)) 
    for i in range(len_doc):
        for j in range(i+1, len_doc):
            m = rouge.compute(predictions=[l_sentences[i]], references=[[l_sentences[j]]])
            ma_rouge[i,j] = m['rouge1']
            ma_rouge[j,i] = m['rouge1']
    mean_rouge = torch.mean(ma_rouge, 1) # calculate mean rouge-f1 score between a sentence and the reset
    _, indexes = torch.topk(mean_rouge, n_mask)
    indexes = torch.sort(indexes, 0).values
    for i in range(indexes.size()[0]):
        masked_sentence = l_sentences[indexes[i]]
        document = document.replace(masked_sentence, "<mask_1>")
        labels += masked_sentence + " "
    return document.replace('\n', ' '), labels

if __name__ == "__main__":
    nltk.download('punkt')
    rouge = evaluate.load('rouge')
    dataset = load_dataset('c4', 'en', streaming=True)
    lis_doc = []
    for index, sample in enumerate(dataset['train']):
        if index == 40000: break
        lis_doc.append(sample['text'])
    f_text = open(text_path, "a", encoding='utf-8')
    f_label = open(label_path, "a", encoding='utf-8')
    print('start_processing')
    for index in range(0, 40000):
        doc, label = process_data(rouge, lis_doc[index])
        if index % 100 == 0:
            print(index)
        if label is not None:
            f_text.write(doc + "\n")
            f_label.write(label + "\n")
    f_text.close()
    f_label.close()