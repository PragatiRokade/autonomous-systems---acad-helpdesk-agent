import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from loguru import logger
#pdf parsing
def pdf_parse(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []

    for pg_num in range(len(doc)):
        page = doc.load_page(pg_num)
        page_text = page.get_text()

        
        for para in page_text.split("\n\n"):
            chunks.append({
                "text": para.strip(),
                "metadata": {
                    "source": "CMU LSTM Notes",
                    "loc": pg_num + 1#location of chunk 
                }
            })
    return chunks

#html parsing
def html_parse(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    chunks = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
        text = tag.get_text().strip()
        if text:
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "Chris Olah Blog",
                    "loc": tag.name
                }
            })
    return chunks
def parse_all(pdf_path,url):
    chunk_pdf=pdf_parse(pdf_path=pdf_path)
    chunk_html=html_parse(url=url)
    chunk=chunk_pdf+chunk_html#all chunk together 
    for idx, ch in enumerate(chunk):
        ch["id"] = f"chunk{idx}"
    logger.debug("chunking complete")
    logger.debug(f"{len(chunk)}no. of chunk formed ")
    return chunk