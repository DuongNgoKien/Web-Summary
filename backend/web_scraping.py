from urllib.request import urlopen
from bs4 import BeautifulSoup
import nltk

def web_scrapping(url):
    try:
        page = urlopen(url)
    except Exception as ex:
        page = None
    else:
        html_bytes = page.read()
        html = html_bytes.decode("utf-8")
        
        soup = BeautifulSoup(html, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        
        list_chunks = []
        for chunk in chunks:
            sentences = nltk.sent_tokenize(chunk)
            if len(sentences) > 1:
                list_chunks.extend(sentences)
        text = ' '.join(chunk for chunk in list_chunks if chunk)    
    finally:
        if page is not None:
             page.close() 
    return text