import json
from ollama import chat
from ollama import ChatResponse
import pdfplumber
import spacy

# Carica il modello NLP (italiano, puoi aggiungere anche l'inglese se vuoi gestire anche quella lingua)
nlp = spacy.load("it_core_news_lg")

def processa_testo_nlp(testo):
    """
    Processa il testo con spaCy e restituisce:
    - le frasi segmentate
    - le entità riconosciute (opzionale, qui non le utilizziamo nel prompt)
    """
    doc = nlp(testo)
    frasi = [sent.text.strip() for sent in doc.sents]
    # Per ora non usiamo le entità in questo passaggio
    return frasi

def estrai_e_processa_pdf(percorso_pdf):
    """
    Estrae il testo dal PDF e ritorna un dizionario:
    { "pagina_1": {"raw_text": <testo>, "sentences": [<frase1>, <frase2>, ...] }, ... }
    """
    risultati = {}
    try:
        with pdfplumber.open(percorso_pdf) as pdf:
            for indice, pagina in enumerate(pdf.pages):
                testo = pagina.extract_text()
                if testo:
                    frasi = processa_testo_nlp(testo)
                    risultati[f"pagina_{indice + 1}"] = {
                        "raw_text": testo,
                        "sentences": frasi
                    }
        return risultati
    except Exception as e:
        print("Errore durante l'estrazione o l'elaborazione del testo:", e)
        return None

# Percorso al file PDF
percorso_file = r"C:\Users\lucab\Documents\main\StartupWalkover\extractor_agent\pitch\Docutique.pdf"
risultati = estrai_e_processa_pdf(percorso_file)

if not risultati:
    print("Nessun testo elaborato.")
    exit(1)

# Concateniamo il testo estratto: qui possiamo decidere se usare il raw_text o le frasi segmentate
# In questo esempio uso le frasi segmentate per cercare di avere un testo più "pulito"
pitch_text = ""
for pagina, dati in risultati.items():
    pitch_text += " ".join(dati["sentences"]) + "\n"

# Costruiamo il prompt per Ollama
prompt = f"""
Leggi il seguente pitch e restituisci esclusivamente in formato JSON le seguenti informazioni:
- startup_name: Nome della startup
- industry: Settore o area di business
- product_description: Descrizione del prodotto/servizio
- team_info: Informazioni sul team e background dei fondatori
- development_stage: Stadio di sviluppo (idea, prototipo, scaling, ecc.)
- financials: Metriche e dati finanziari rilevanti (fatturato, crescita, ecc.)
- business_model: Descrizione del modello di business
- investment_request: Informazioni sulla richiesta di investimento e utilizzo dei fondi

Non fornire ulteriori commenti o informazioni extra, solo il JSON.
Ecco il pitch:
'''{pitch_text}'''
"""

# Chiamata a Ollama; modifica il nome del modello se necessario (qui uso "llama3.2" come da esempio)
response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
])

# Stampa il JSON restituito
print("JSON strutturato estratto dal pitch:")
print(response['message']['content'])
# Se desideri anche elaborare il JSON in Python:
try:
    output_json = json.loads(response['message']['content'])
    print("\nJSON parsificato:")
    print(json.dumps(output_json, indent=4))
except Exception as e:
    print("Errore nel parsing del JSON:", e)




exit()

# from llama_cpp import Llama

# # Specifica il percorso completo al file del modello, ad es.:
# model_path = r"C:\Users\lucab\.llama\checkpoints\Llama-2-7b\ggml-Llama-2-7B-chat.Q4_0.bin"

# # Inizializza l'istanza del modello
# llama_instance = Llama(model_path=model_path, n_ctx=1024)

# # Esempio di prompt per generare testo
# prompt = "Ciao, dammi una breve descrizione di una startup innovativa."
# response = llama_instance(prompt, max_tokens=150, temperature=0.0)
# print(response["choices"][0]["text"].strip())

from ollama import chat
from ollama import ChatResponse


response: ChatResponse = chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)