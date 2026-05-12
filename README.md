# RAG con Ollama + Phi-3 + Streamlit

Un'applicazione di **Retrieval-Augmented Generation (RAG)** completamente locale che permette di interrogare documenti PDF tramite linguaggio naturale, senza inviare dati al cloud. Utilizza **Phi-3** (Microsoft) via **Ollama**, **ChromaDB** come vector store e **Streamlit** per l'interfaccia utente.

---

## Indice

- [Cos'è il RAG? (Teoria)](#-cosè-il-rag-teoria)
- [Architettura del Progetto](#️-architettura-del-progetto)
- [Stack Tecnologico](#-stack-tecnologico)
- [Prerequisiti](#-prerequisiti)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Struttura del Codice](#-struttura-del-codice)
- [Come Funziona (Passo per Passo)](#-come-funziona-passo-per-passo)
- [Limitazioni e Possibili Miglioramenti](#-limitazioni-e-possibili-miglioramenti)

---

## Cos'è il RAG? (Teoria)

### Il Problema dei LLM "Puri"

I Large Language Models (LLM) come Phi-3 hanno una **conoscenza statica** limitata alla loro data di addestramento. Non sanno nulla di:
- Documenti aziendali privati
- Dati aggiornati dopo il training cutoff
- Informazioni specifiche del tuo dominio

### La Soluzione: RAG

Il **Retrieval-Augmented Generation** è un pattern architetturale che arricchisce un LLM con una **base di conoscenza esterna e dinamica**, combinando due fasi:

```
[Documento] → [Chunking] → [Embedding] → [Vector DB]
                                                ↓
[Domanda Utente] → [Embedding] → [Ricerca Similarità] → [Chunk Rilevanti]
                                                                 ↓
                                              [LLM] → [Risposta Contestualizzata]
```

### I 5 Componenti Chiave

**1. Document Loader** — Legge il documento sorgente (PDF, DOCX, ecc.) e lo converte in testo grezzo.

**2. Text Splitter / Chunking** — Divide il testo in frammenti più piccoli (_chunk_). Necessario perché:
- I modelli hanno un _context window_ limitata
- La ricerca per similarità funziona meglio su testi corti e specifici
- Si evita di passare all'LLM tutto il documento (costoso e rumoroso)

**3. Embedding Model** — Converte ogni chunk (e poi le domande dell'utente) in vettori numerici ad alta dimensionalità che rappresentano il **significato semantico** del testo. Testi con significato simile producono vettori vicini nello spazio vettoriale.

**4. Vector Store** — Un database specializzato (ChromaDB, FAISS, Pinecone...) che indicizza i vettori e permette ricerche efficienti per similarità coseno o euclidea. In questo progetto si usa **ChromaDB** in memoria.

**5. Retriever + LLM Chain** — Quando arriva una domanda:
1. La domanda viene convertita in embedding
2. Il retriever trova i _k_ chunk più simili nel vector store
3. I chunk vengono iniettati nel prompt come contesto
4. L'LLM genera una risposta basandosi su quel contesto

### Perché "in locale" con Ollama?

Ollama permette di eseguire LLM di qualità sul proprio hardware senza mandare dati a servizi esterni (OpenAI, Anthropic, ecc.). Ideale per:
- **Privacy**: documenti sensibili restano sul tuo PC
- **Costo zero**: nessuna API key, nessun consumo di token a pagamento
- **Offline**: funziona senza connessione internet

---

## Architettura del Progetto

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI (Browser)              │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │         main.py             │
          │                             │
          │  1. PyPDFLoader             │  ← Legge il PDF
          │  2. RecursiveTextSplitter   │  ← Divide in chunk
          │  3. OllamaEmbeddings(phi3)  │  ← Genera vettori
          │  4. ChromaDB (in-memory)    │  ← Salva vettori
          │  5. RetrievalQA Chain       │  ← Recupera + Risponde
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     Ollama (localhost:11434) │
          │     Modello: phi3            │
          └─────────────────────────────┘
```

---

## Stack Tecnologico

| Componente | Tecnologia | Ruolo |
|---|---|---|
| **LLM** | [Phi-3 (Microsoft)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) via Ollama | Generazione testo |
| **Embedding** | Phi-3 via OllamaEmbeddings | Vettorizzazione testo |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Database vettoriale in-memory |
| **Framework AI** | [LangChain](https://www.langchain.com/) | Orchestrazione pipeline RAG |
| **PDF Parser** | PyPDF | Estrazione testo da PDF |
| **UI** | [Streamlit](https://streamlit.io/) | Interfaccia web |
| **Runtime LLM** | [Ollama](https://ollama.com/) | Esecuzione locale LLM |

---

## Prerequisiti

- **Python** 3.9 o superiore
- **Ollama** installato e in esecuzione
- **Phi-3** scaricato via Ollama
- **PyCharm** (o qualsiasi IDE Python)

### Installare Ollama su Windows (CLI)

```powershell
# Scarica e installa Ollama da https://ollama.com/download
# Oppure via winget:
irm https://ollama.com/install.ps1 | iex

# Verifica l'installazione
ollama --version
```

### Scaricare ed Eseguire Phi-3

```powershell
# Scarica il modello Phi-3 (circa 2.3 GB)
ollama pull phi3

# Verifica che il modello sia disponibile
ollama list

# (Opzionale) Testa il modello da terminale
ollama run phi3
```

> **Nota:** Ollama deve restare in esecuzione in background durante l'uso dell'app. Verifica che il server sia attivo su `http://localhost:11434`.

---

## Installazione

### 1. Clona la Repository

```bash
git clone https://github.com/Matteea/Progetto-RAG-Ollama.git
cd Progetto-RAG-Ollama
```

### 2. Crea un Ambiente Virtuale (consigliato)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Installa le Dipendenze

```bash
pip install langchain langchain-community langchain-classic chromadb streamlit pypdf ollama
```

Oppure, se è presente un `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Riepilogo Pacchetti

| Pacchetto | Descrizione |
|---|---|
| `langchain` | Core framework per pipeline LLM |
| `langchain-community` | Integrazioni community (Ollama, Chroma, PyPDF...) |
| `langchain-classic` | Chain classiche come `RetrievalQA` |
| `chromadb` | Database vettoriale locale |
| `streamlit` | Framework per UI web in Python |
| `pypdf` | Lettura e parsing di file PDF |
| `ollama` | Client Python per Ollama |

---

## Utilizzo

### 1. Assicurati che Ollama sia in esecuzione

```powershell
# Controlla che il server risponda
curl http://localhost:11434
# Oppure apri il browser su http://localhost:11434
```

### 2. Avvia l'Applicazione Streamlit

```bash
streamlit run main.py
```

Il browser si aprirà automaticamente su `http://localhost:8501`.

### 3. Usa l'App

1. **Carica un PDF** tramite il file uploader
2. Attendi il messaggio *"Documento pronto!"* (l'elaborazione può richiedere qualche secondo)
3. **Scrivi la tua domanda** nel campo di testo
4. Leggi la risposta generata dal modello Phi-3

---

## Struttura del Codice

```
Progetto-RAG-Ollama/
│
├── main.py           # Applicazione principale Streamlit
├── requirements.txt  # Dipendenze necessarie
└── README.md         # Questa documentazione
```

### `main.py` — Spiegazione Dettagliata

```python
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
import os

st.title("RAG Ollama phi3 lettura pdf")

# 1. Caricamento del file PDF
uploaded_file = st.file_uploader("Carica un documento PDF", type="pdf")

if uploaded_file:
    # Salva temporaneamente il file per leggerlo
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    st.info("Elaborazione del documento in corso...")

    # 2. Caricamento e Chunking (divisione del testo)
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3. Creazione degli Embedding e salvataggio nel Vector DB locale
    embeddings = OllamaEmbeddings(model="phi3")
    vector_store = Chroma.from_documents(chunks, embeddings)

    # 4. Configurazione del Retriever e del LLM
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="phi3")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    st.success("Documento pronto! Fai la tua domanda qui sotto.")

    # 5. Interfaccia di Chat
    user_question = st.text_input("Cosa vuoi sapere dal documento?")
    if user_question:
        with st.spinner("L'LLM sta pensando..."):
            response = qa_chain.run(user_question)
            st.write("### Risposta:")
            st.write(response)

    # Pulizia del file temporaneo
    os.remove("temp.pdf")
```

---

## Come Funziona

### Step 1 — Caricamento PDF
`PyPDFLoader` legge il file PDF caricato dall'utente e lo converte in una lista di oggetti `Document` di LangChain, uno per ogni pagina.

### Step 2 — Chunking
`RecursiveCharacterTextSplitter` divide il testo in chunk da **500 caratteri** con un overlap di **50 caratteri**. L'overlap garantisce che il contesto a cavallo tra due chunk non vada perso.

```
|------- chunk 1 (500 char) -------|
                             |-- overlap (50) --|
                             |------- chunk 2 (500 char) -------|
```

### Step 3 — Embedding e Vector Store
`OllamaEmbeddings` usa Phi-3 per convertire ogni chunk in un vettore numerico. I vettori vengono salvati in **ChromaDB in-memory** (non persiste al riavvio).

### Step 4 — Retriever
Il retriever, configurato con `k=3`, recupera i **3 chunk semanticamente più vicini** alla domanda dell'utente usando la similarità coseno.

### Step 5 — RetrievalQA Chain
La chain di tipo `"stuff"` concatena i 3 chunk recuperati in un unico prompt e li passa a Phi-3, che genera la risposta finale basandosi solo su quel contesto.

```
Prompt finale inviato a Phi-3:
┌────────────────────────────────────────────┐
│ Usa il seguente contesto per rispondere:   │
│                                            │
│ [Chunk 1 più rilevante]                    │
│ [Chunk 2 più rilevante]                    │
│ [Chunk 3 più rilevante]                    │
│                                            │
│ Domanda: <domanda dell'utente>             │
└────────────────────────────────────────────┘
```

---

## Limitazioni e Possibili Miglioramenti

### Limitazioni Attuali

- **ChromaDB in-memory**: il vector store viene ricreato ad ogni upload. Non c'è persistenza tra sessioni.
- **File temporaneo**: il PDF viene salvato come `temp.pdf` e poi eliminato; potrebbero verificarsi conflitti in uso concorrente.
- **Nessuna cronologia chat**: ogni domanda è indipendente, non c'è memoria della conversazione.
- **Un documento alla volta**: non è possibile caricare più PDF contemporaneamente.

### Possibili Miglioramenti

- **Persistenza ChromaDB**: usare `persist_directory` per salvare gli embedding su disco e riutilizzarli senza rielaborare il PDF
- **Memoria conversazionale**: sostituire `RetrievalQA` con `ConversationalRetrievalChain` per domande di follow-up
- **Multi-documento**: indicizzare più PDF in un unico vector store
- **Chunk size dinamico**: regolare la dimensione dei chunk in base alla lunghezza del documento
- **Modello selezionabile**: permettere all'utente di scegliere il modello Ollama dalla UI
- **Sorgenti nelle risposte**: mostrare da quale pagina/sezione del PDF viene la risposta (`return_source_documents=True`)
