#!/usr/bin/env python
# 03_index_vector.py – build Chroma vector-store (Hugging Face version)
# ---------------------------------------------------------------
from pathlib import Path
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ➜ Embedding HuggingFace (gratuito, CPU-friendly)
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="jinaai/jinaai/jina-embeddings-v4",   # questo è il v4
    encode_kwargs={"normalize_embeddings": True}
)

# ➜ Nuovo import Chroma (pacchetto esterno)
from langchain_chroma import Chroma
 
OCR_DIR = Path("pitch/interim/ocr_json")
VDB_DIR = Path("pitch/interim/vector_test")

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
vdb = Chroma(
    persist_directory=str(VDB_DIR),
    collection_name="pitch_test",
    embedding_function=emb
)

files = list(OCR_DIR.glob("*.json"))
if not files:
    raise SystemExit(" No JSON in pitch/interim/ocr_json – esegui prima 02_ocr.py")

n_chunks = 0
for js in tqdm(files, desc="Indicizzo"):
    cid = js.stem
    pages = json.load(open(js, encoding="utf-8"))
    for p in pages:
        for chunk in splitter.split_text(p["text"]):
            vdb.add_documents([Document(
                page_content=chunk,
                metadata={"company_id": cid,
                          "page_name": p["page_name"],
                          "ocr": p["ocr"]})])
            n_chunks += 1


print(f"Indicizzati {n_chunks:,} chunk in {VDB_DIR.resolve()}")
