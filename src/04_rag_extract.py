#!/usr/bin/env python
# 04_rag_extract.py  ─ mini-pilot di estrazione campi via RAG
# ==========================================================
"""
Per ogni company_id indicizzato in Chroma:
  1. Recupera i chunk più pertinenti (k=8, MMR).
  2. Passa il contesto a un LLM con prompt + PydanticOutputParser.
  3. Raccoglie i campi (schema demo) e li salva in Parquet.

Output:
    data/interim/rag_extract.parquet
"""

from pathlib import Path
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# ------------------------- SCEGLI LLM ---------------------------------
USE_OPENAI = True          # False ==> usa un piccolo modello locale HF

if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    # LLM locale (CPU/GPU). Richiede transformers + un modello da HF Hub.
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline

    model_name = "HuggingFaceH4/zephyr-7b-beta"     # 4-8 GB VRAM in 4-bit
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype="auto",
                                                 device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tok,
                    max_new_tokens=512)
    llm  = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0})



# ------------------------- VECTOR-STORE -------------------------------
VDB_DIR = Path("pitch/interim/vector_test")
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings   # già importato sopra

emb = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        encode_kwargs={"normalize_embeddings": True}
)

vs = Chroma(
    persist_directory=str(VDB_DIR),
    collection_name="pitch_test",
    embedding_function=emb        # ← aggiungi questa riga
)

retriever = vs.as_retriever(search_type="mmr", k=8)


# *********************************************************************
# ------------------------- SCHEMA PROMPT 
# *********************************************************************

"""class PitchFlat(BaseModel):
    business_model: str  | None = Field(..., description="es. SaaS, marketplace")
    funding_stage:  str  | None = Field(..., description="Seed, Series A …")
    tam_usd:        float| None = Field(..., description="TAM in USD") #Total Addressable Market
    mrr_usd:        float| None = Field(..., description="MRR in USD") #Monthly Recurring Revenue
    founders_count: int  | None = Field(..., description="Numero fondatori")
    usp:            str  | None = Field(..., description="Unique Selling Proposition")"""

class PitchFlat(BaseModel):
    business_model: str  | None = None
    funding_stage:  str  | None = None
    tam_usd:        float| None = None # Total Addressable Market
    mrr_usd:        float| None = None # Monthly Recurring Revenue
    founders_count: int  | None = None
    usp:            str  | None = None # Unique Selling Proposition
    survival_probability: int | None = None
    probability_reason: str | None = None 
    revenue_24m_usd: float | None = None 
    notes:        str  | None = None # Note aggiuntive (opzionale)

parser = PydanticOutputParser(pydantic_object=PitchFlat)

example_json = """
{{
  "business_model": "SaaS",
  "funding_stage":  "Seed",
  "tam_usd":        500000000,
  "mrr_usd":        12000,
  "founders_count": 2,
  "usp":            "AI-powered marketplace",
  "survival_probability": 72,                // percentuale intera 0-100
  "probability_reason": "Early traction and large TAM, but highly competitive market.",
  "revenue_24m_usd":      3500000,
  "notes":               "Solida trazione iniziale, TAM ampio e pochi competitor diretti."  
}}
"""

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Venture Capital analyst. Respond **ONLY** with a ```json``` block "
        "containing exactly these 8 fields:\n"
        "business_model, funding_stage, tam_usd, mrr_usd, founders_count, usp, "
        "survival_probability (0-100), probability_reason (1-2 sentences)., revenue_24m_usd, notes\n"
        "Ignore any knowledge not present in the context (do NOT use data after 31-Dec-2021)."
    ),
    (
        "system",
        f"Example:\n```json\n{example_json}\n```"
    ),
    (
        "user",
        "Context:\n{ctx}\n\nReturn only the requested JSON block."
    ),
])

"""class CompanyInternal(BaseModel):
    founders_avg_age:        float | None = Field(..., description="Età media fondatori")
    founders_count:          int   | None = Field(..., description="Numero fondatori")
    founders_education:      str   | None = Field(..., description="Formazione (es. PhD, MBA)")
    team_size:               int   | None = Field(..., description="Dipendenti totali")
    team_avg_experience_yrs: float | None = Field(..., description="Esperienza media (anni)")
    business_model:          str   | None = Field(..., description="Es. SaaS, marketplace")
    monetization_channels:   str   | None = Field(..., description="Es. subscription, ads")
    equity_structure:        str   | None = Field(..., description="Cap table sintetica")

class CompanyExternal(BaseModel):
    tam_usd:         float | None = Field(..., description="Total Addressable Market")
    sam_usd:         float | None = Field(..., description="Serviceable Available Market")
    som_usd:         float | None = Field(..., description="Serviceable Obtainable Market")
    customer_segment:str   | None = Field(..., description="Target clienti")
    competitors_count:int  | None = Field(..., description="N° competitor principali")
    usp:             str   | None = Field(..., description="Unique Selling Proposition")
    users_customers: int   | None = Field(..., description="Utenti o clienti attivi")
    mrr_usd:         float | None = Field(..., description="Monthly Recurring Revenue")
    engagement_metrics:str | None = Field(..., description="KPI retention, DAU/MAU …")
    partnerships_count:int | None = Field(..., description="Numero partnership")
    partnerships_type:str  | None = Field(..., description="Tipologia partnership")

class FundingInvestors(BaseModel):
    investor_profile:       str   | None = Field(..., description="Es. VC, Angel, CVC")
    funding_stage:          str   | None = Field(..., description="Pre-seed, Seed, Series A …")
    capital_raised_usd:     float | None = Field(..., description="Capitale raccolto finora")
    premoney_valuation_usd: float | None = Field(..., description="Valutazione pre-money")
    postmoney_valuation_usd:float | None = Field(..., description="Valutazione post-money")

class Technology(BaseModel):
    innovation_degree:  str   | None = Field(..., description="Radicale, incrementale …")
    development_stage:  str   | None = Field(..., description="PoC, MVP, beta, scalabile")
    ip_patent_count:    int   | None = Field(..., description="Brevetti depositati")
    proprietary_knowhow:bool  | None = Field(..., description="Know-how proprietario?")

class PitchInfo(BaseModel):
    company_internal:   CompanyInternal
    company_external:   CompanyExternal
    funding_investors:  FundingInvestors
    technology:         Technology
# --------------------------------------------------------------------

parser = PydanticOutputParser(pydantic_object=PitchInfo)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sei un analista VC. Estrarrai i campi richiesti e risponderai "
        "SOLO con JSON perfettamente valido secondo lo schema."
    ),
    (
        "user",
        "{fi}\n\nContesto:\n{ctx}\n\nDomanda: compila tutti i campi dello schema."
    ),
])"""

"""# *********************************************************************
# ------------------------- LOOP 
# *********************************************************************
OCR_DIR = Path("pitch/interim/ocr_json")   # già usato nello script 03
cids = sorted(f.stem for f in OCR_DIR.glob("*.json"))
rows = []
for cid in cids:
    docs = retriever.invoke("startup info",
                            search_kwargs={"filter": {"company_id": cid}})
    context = "\n\n".join(d.page_content for d in docs)
    res = (prompt | llm | parser).invoke({
        "fi":  parser.get_format_instructions(),
        "ctx": context
    })
    rows.append({"company_id": cid, **res.model_dump()})

# ------------------------- SALVA --------------------------------------
out_path = Path("pitch/interim/rag_extract.parquet")
pd.DataFrame(rows).to_parquet(out_path, index=False)
print(f" Salvato {len(rows)} record in {out_path.resolve()}")"""

# *********************************************************************
# ------------------------- STANDALONE 
# *********************************************************************
target_cid = "1Control"        

docs = retriever.invoke(
    "startup info",
    search_kwargs={"filter": {"company_id": target_cid}}
)
context = "\n\n".join(d.page_content for d in docs)

res = (prompt | llm | parser).invoke({
    "fi":  parser.get_format_instructions(),
    "ctx": context
})

df = pd.DataFrame([{"company_id": target_cid, **res.model_dump()}])
df.to_parquet("pitch/interim/rag_extract.parquet", index=False)
df.to_csv("pitch/interim/rag_extract.csv", index=False)
print(f"One record extracted for {target_cid}")
