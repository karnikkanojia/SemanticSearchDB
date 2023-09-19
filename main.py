from fastapi import FastAPI, HTTPException

import torch
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

import logging
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

@app.get('/')
def hello():
    return {"message": "Hello World"}

@app.on_event("startup")
async def startup_event():
    """
    Load all the necessary models and data once the server starts.
    """
    app.directory = './archive/data/data'
    app.documents = load_docs(app.directory)
    app.docs = split_docs(app.documents)

    app.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    app.persist_directory = "chroma_db"

    app.vectordb = Chroma.from_documents(
        documents=app.docs,
        embedding=app.embeddings,
        persist_directory=app.persist_directory
    )
    logging.info("Persisting the vector database")
    app.vectordb.persist()

    logging.info("Loading the language model")
    app.model_name = "google/flan-t5-base"
    app.llm = HuggingFaceHub(repo_id=app.model_name, model_kwargs={
        "device_map": "auto", "load_in_8bit": True, "max_length": 512, "temperature": 0, "torch_dtype":torch.bfloat16,
    })

    logging.info("Loading the QA chain")
    app.db = Chroma.from_documents(app.docs, app.embeddings)
    app.chain = load_qa_chain(app.llm, chain_type="stuff", verbose=True)


@app.get("/query/{question}")
async def query_chain(question: str):
    """
    Queries the model with a given question and returns the answer.
    """
    matching_docs_score = app.db.similarity_search_with_score(question)
    if len(matching_docs_score) == 0:
        raise HTTPException(status_code=404, detail="No matching documents found")

    matching_docs = [doc for doc, score in matching_docs_score]
    answer = app.chain.run(input_documents=matching_docs, question=question)

    # Prepare the sources
    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in matching_docs_score]

    return {"answer": answer, "sources": sources}


def load_docs(directory: str, loader_cls = PyMuPDFLoader):
    """
    Load documents from the given directory.
    """
    loader = DirectoryLoader(directory, glob='**/*.pdf', loader_cls=loader_cls, show_progress=True, use_multithreading=True)
    documents = loader.load()

    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs