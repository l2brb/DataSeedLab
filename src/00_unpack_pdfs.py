# 00_unpack_pdfs.py

import zipfile
import shutil
import tempfile
import hashlib
from pathlib import Path

RAW_DIR      = Path("pitch/raw")        # i 6 zip master
STAGING_DIR  = Path("pitch/staging")    # 1 cartella per azienda
SAFE_CHARS   = "-_.() "

def safe(name): return ''.join(c for c in name if c.isalnum() or c in SAFE_CHARS)

def sha256(fp):                     # deduplica PDF identici
    h = hashlib.sha256()
    while chunk := fp.read(1 << 20):
        h.update(chunk)
    fp.seek(0); return h.hexdigest()

def recursive_unzip(zip_path: Path, out: Path):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    for inner in out.rglob("*.zip"):
        tmp = inner.with_suffix("")     # cartella col nome dello zip
        tmp.mkdir(exist_ok=True)
        with zipfile.ZipFile(inner) as z2:
            z2.extractall(tmp)
        inner.unlink()
        recursive_unzip(tmp, tmp)

def run():
    for master in RAW_DIR.glob("*.zip"):
        with tempfile.TemporaryDirectory() as tmp:
            recursive_unzip(master, Path(tmp))
            for pdf in Path(tmp).rglob("*.pdf"):
                cid = safe(pdf.parts[-2].lower())   # cartella immediatamente sopra = company
                dest_dir = STAGING_DIR / cid
                dest_dir.mkdir(parents=True, exist_ok=True)

                # deduplica: nome = hash.pdf
                with open(pdf, "rb") as f:
                    h = sha256(f)
                dest = dest_dir / f"{h}.pdf"
                if not dest.exists():
                    shutil.move(str(pdf), dest)

if __name__ == "__main__":
    run()
    print(" Unzip completed")
