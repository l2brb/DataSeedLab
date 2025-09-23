#!/usr/bin/env python
# 02_ocr.py  ─ OCR + text extraction
# =========================================
"""
Converte le slide PNG in testo via PaddleOCR e unisce eventuale
testo nativo estratto dai PDF corrispondenti. Output JSON per azienda.

Output directory:
    pitch/interim/ocr_json/<company_id>.json
"""

from pathlib import Path
import json
import multiprocessing as mp
from pypdf import PdfReader
from paddleocr import PaddleOCR

# ----- path -----
RAW_DIR   = Path("pitch/raw/test")             # PDF originali
PNG_DIR   = Path("pitch/interim/slides_png")   # PNG da step 01
OUT_DIR   = Path("pitch/interim/ocr_json")     # output JSON
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Download and verify model
# ------------------------------------------------------------------

print("Checking/Downloading PaddleOCR models (first run only)…")
PaddleOCR(use_angle_cls=True, lang="en")   # blocca finché i pesi non ci sono
print("Done")

# ------------------------------------------------------------------
# Lazy-loader in ogni worker
# ------------------------------------------------------------------
_ocr = None
def get_ocr():
    """Istanzia PaddleOCR la prima volta in ciascun processo."""
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr

# ------------------------------------------------------------------
# 3️⃣ Funzione che processa UNA azienda
def process_company(company_dir: Path) -> str:
    cid = company_dir.name
    out_file = OUT_DIR / f"{cid}.json"
    if out_file.exists():
        return f"{cid}: già presente → skip"

    records = []

    # ---- A) OCR su tutte le PNG ----
    for png in sorted(company_dir.glob("*.png")):
        txt = "\n".join(line[1][0] for line in get_ocr().ocr(str(png), cls=True)[0])
        records.append({"page_name": png.name, "text": txt, "ocr": True})

    # ---- B) testo nativo dal PDF (se disponibile) ----
    pdf_folder = RAW_DIR / cid
    for pdf in pdf_folder.glob("*.pdf"):
        reader = PdfReader(str(pdf))
        for idx, page in enumerate(reader.pages, 1):
            txt = page.extract_text() or ""
            if len(txt.strip()) >= 40:        # soglia minima caratteri
                records.append({
                    "page_name": f"{pdf.stem}_p{idx}",
                    "text":      txt,
                    "ocr":       False
                })

    # ---- C) salva JSON ----
    json.dump(records, open(out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return f"{cid}: {len(records)} pagine salvate"

# ------------------------------------------------------------------
# 4️⃣ Main parallelo (puoi mettere processes=1 se vuoi farlo in serie)
if __name__ == "__main__":
    company_dirs = [d for d in PNG_DIR.iterdir() if d.is_dir()]
    with mp.Pool(processes=min(6, len(company_dirs))) as pool:
        for msg in pool.imap_unordered(process_company, company_dirs):
            print("•", msg)

    print(f"\n🎉 Finito. JSON salvati in {OUT_DIR.resolve()}")