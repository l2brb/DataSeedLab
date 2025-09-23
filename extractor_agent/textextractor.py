import pdfplumber
import spacy


# Carica il modello NLP in italiano
nlp = spacy.load("it_core_news_lg")



# Processing del testo con spaCy
def processa_testo_nlp(testo):
    """
    Processa il testo con spaCy e restituisce:
    - le frasi segmentate
    - le entità riconosciute
    """
    doc = nlp(testo)
    frasi = [sent.text.strip() for sent in doc.sents]
    entita = [(ent.text, ent.label_) for ent in doc.ents]
    return frasi, entita

# Estrazione con pdfplumber e processamento con spaCy
def estrai_e_processa_pdf(percorso_pdf):
    risultati = {}
    try:
        with pdfplumber.open(percorso_pdf) as pdf:
            for indice, pagina in enumerate(pdf.pages):
                # Estrazione del testo dalla pagina
                testo = pagina.extract_text()
                if testo:

                    # Processa il testo con spaCy
                    frasi, entita = processa_testo_nlp(testo)
                    risultati[f"pagina_{indice + 1}"] = {
                        "raw_text": testo,
                        "sentences": frasi,
                        "entities": entita
                    }
        return risultati
    except Exception as e:
        print("Error during text extraction:", e)
        return None




############################################################################################### USAGE

percorso_file = r"C:\Users\lucab\Documents\main\StartupWalkover\extractor_agent\pitch\20Seconds.pdf"
risultati = estrai_e_processa_pdf(percorso_file)

if risultati:
    for pagina, dati in risultati.items():
        print(f"\n{pagina}:\n{'-'*40}")
        # print("Testo grezzo:")
        # print(dati["raw_text"])
        print("\nFrasi segmentate:")
        for frase in dati["sentences"]:
            print("-", frase)
            print("\nEntità riconosciute:")
        if dati["entities"]:
            for ent in dati["entities"]:
                print("-", ent)
        else:
             print("Nessuna entità riconosciuta.")
else:
    print("Nessun testo elaborato.")
