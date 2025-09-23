#!/usr/bin/env python
# coding: utf-8
"""
Estrae ogni pagina PDF come PNG (300 dpi) da tutte le aziende in data/raw/test.
Salva sotto pitch/interim/slides_png/<company_id>/<pdf_stem>_p<idx>.png
"""

from pathlib import Path
from pdf2image import convert_from_path
import hashlib, multiprocessing as mp

ROOT_RAW = Path("pitch/raw/test")
OUT_DIR  = Path("pitch/interim/slides_png")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _hash_path(p: Path) -> str:
    return hashlib.sha1(str(p).encode()).hexdigest()[:6]

def pdf_to_png(pdf: Path):
    company = pdf.parent.name
    pages   = convert_from_path(pdf, dpi=300, fmt="png")
    for i, img in enumerate(pages, 1):
        out_sub = OUT_DIR / company
        out_sub.mkdir(parents=True, exist_ok=True)
        out = out_sub / f"{pdf.stem}_{_hash_path(pdf)}_p{i}.png"
        img.save(out, "PNG")

if __name__ == "__main__":
    pdfs = list(ROOT_RAW.rglob("*.pdf"))
    with mp.Pool() as pool:
        pool.map(pdf_to_png, pdfs)

    print(f"Extracted {len(list(OUT_DIR.rglob('*.png')))} PNG slides")
