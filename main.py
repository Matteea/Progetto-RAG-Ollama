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
    # Usiamo lo stesso Ollama per generare gli embedding in locale
    embeddings = OllamaEmbeddings(model="phi3")
    vector_store = Chroma.from_documents(chunks, embeddings)

    # 4. Configurazione del Retriever e del LLM
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Prende i 3 frammenti più rilevanti
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