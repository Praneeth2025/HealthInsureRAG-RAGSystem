import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

# --- PDF & Text Handling ---
import fitz  # from PyMuPDF
import pdfplumber
import re
import numpy as np
import os
from huggingface_hub import InferenceClient
# --- LangChain: Document Loading, Splitting, Embedding ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from pydantic import BaseModel

# --- ChromaDB Vector Store ---
import chromadb
# from chromadb import Client
# from chromadb.config import Settings
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- LangChain: LLMs, Retrieval, Compression ---
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# --- LangChain: Agent & Tools ---
from langchain_core.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor

# --- Caching ---
from diskcache import Cache
import os

# Set token directly (less secure)
import os
os.environ["CHROMA_API_IMPL"] = "chromadb.api.local.LocalAPI"



#TEXT PROCESSING:

#Extracting Letters with larger font size
def extract_headings(pdf_path, size_threshold=9):
    """
    Extracts headings from a PDF by selecting lines with font size greater than a threshold.

    Args:
        pdf_path (str): Path to the PDF file.
        size_threshold (float): Font size above which a line is considered a heading.

    Returns:
        list[str]: List of extracted heading texts.
    """
    doc = fitz.open(pdf_path)
    headings = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = " ".join(span["text"] for span in line["spans"])
                    max_font_size = max(span["size"] for span in line["spans"])
                    if max_font_size > size_threshold:
                        headings.append(line_text)

    return headings



#Extracting all the bold Sentences
def extract_bold_phrases(pdf_path):
    bold_phrases = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["fontname"])
            current_phrase = []

            for word in words:
                if "Bold" in word["fontname"]:
                    current_phrase.append(word["text"])
                else:
                    if current_phrase:
                        bold_phrases.append(" ".join(current_phrase))
                        current_phrase = []

            # End of page
            if current_phrase:
                bold_phrases.append(" ".join(current_phrase))

    return bold_phrases



def smart_contains(needle, haystack):
    """
    Match `needle` in `haystack`, treating only `\n` as space, preserving all other spacing.
    """
    # Replace \n with space in both
    needle_flat = needle.replace('\n', ' ').strip().lower()
    haystack_flat = haystack.replace('\n', ' ').lower()

    return needle_flat in haystack_flat


def prepend_latest_heading_and_bold(docs, headings, bolds):
    updated_docs = []
    heading_idx = 0
    bold_idx = 0
    last_heading = ""
    last_bold = ""

    for doc in docs:
        content = doc.page_content.strip()

        # Try to match as many headings as possible in order
        while heading_idx < len(headings):
            h = headings[heading_idx]
            if smart_contains(h, content):
                last_heading = h
                heading_idx += 1
            else:
                break  # stop at first that doesn't match

        # Try to match as many bolds as possible in order
        while bold_idx < len(bolds):
            b = bolds[bold_idx]
            if smart_contains(b, content):
                last_bold = b
                bold_idx += 1
            else:
                break  # stop at first that doesn't match

        # Prepend latest seen heading/bold
        prefix = []
        if last_heading:
            prefix.append(last_heading)
        if last_bold:
            prefix.append(last_bold)

        doc.page_content = "\n".join(prefix + [content])
        updated_docs.append(doc)

    return updated_docs


def build_vectorstore_from_pdf(file_path):
    """
    Builds a FAISS vectorstore from a PDF file with heading/bold context.

    Args:
        file_path (str): Path to the input PDF file.

    Returns:
        FAISS: Vector store containing embedded document chunks.
    """

    # Step 1: Load pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # Step 3: Extract context
    headings = extract_headings(file_path)
    bolds = extract_bold_phrases(file_path)

    # Step 4: Enrich each chunk with latest heading/bold context
    docs_with_context = prepend_latest_heading_and_bold(docs, headings, bolds)

    # Step 5: Embed the enriched chunks
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 6: Create and return Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name="My_Data_Chunk",
        persist_directory="chroma_store"
    )
    return vectorstore



#CHROMA STORE:

# #Upoading Embeddings to Chroma Store
# def upload_to_chroma_cloud(local_dir, collection_name, chroma_host, api_key):
#     """
#     Upload a local Chroma collection to Chroma Cloud.

#     Args:
#         local_dir (str): Directory of local Chroma DB (e.g., "chroma_store").
#         collection_name (str): The target collection name on Chroma Cloud.
#         chroma_host (str): URL of the remote Chroma host (e.g., "https://api.trychroma.com").
#         api_key (str): Your Chroma API key.

#     Returns:
#         Remote collection handle (Collection)
#     """
#     # Step 1: Load local collection
#     from langchain.vectorstores import Chroma
#     from langchain.embeddings import HuggingFaceEmbeddings

#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     print(f"Loading local Chroma store from {local_dir}...")
#     local_vectorstore = Chroma(
#         persist_directory=local_dir,
#         collection_name=collection_name,
#         embedding_function=embedding_model,
#     )

#     local_docs = local_vectorstore.get(include=["documents", "embeddings", "metadatas"])

#     print(f"Loaded {len(local_docs['documents'])} documents from local store.")

#     # Step 2: Connect to remote Chroma client
#     print("Connecting to remote Chroma client...")
#     client = chromadb.HttpClient(
#         Settings(
#             chroma_api_impl="rest",
#             chroma_server_host=chroma_host.replace("https://", "").replace("http://", ""),
#             chroma_server_http_port=443,
#             chroma_server_ssl_enabled=True,
#             anonymized_telemetry=False,
#             chroma_api_key=api_key,
#         )
#     )

#     # Step 3: Create or get collection on remote
#     print(f"Creating or retrieving collection '{collection_name}' on remote Chroma...")
#     collection = client.get_or_create_collection(name=collection_name)

#     # Step 4: Upload documents
#     print("Uploading documents to remote collection...")

#     collection.add(
#         documents=local_docs["documents"],
#         embeddings=local_docs["embeddings"],
#         metadatas=local_docs["metadatas"],
#         ids=[f"id-{i}" for i in range(len(local_docs["documents"]))],
#     )

#     print(f"Uploaded {len(local_docs['documents'])} documents to remote collection '{collection_name}'.")
#     return collection


# #Loading Embeddings from Chroma Store

# def load_chroma_cloud_collection(collection_name, chroma_host, api_key):
#     """
#     Connects to Chroma Cloud and retrieves an existing collection.

#     Args:
#         collection_name (str): The name of the collection to retrieve.
#         chroma_host (str): The Chroma Cloud host URL (e.g., "https://api.trychroma.com").
#         api_key (str): Your Chroma Cloud API key.

#     Returns:
#         chromadb.api.models.Collection.Collection: The retrieved collection object.
#     """
#     # Step 1: Connect to the remote Chroma client
#     client = chromadb.HttpClient(
#         Settings(
#             chroma_api_impl="rest",
#             chroma_server_host=chroma_host.replace("https://", "").replace("http://", ""),
#             chroma_server_http_port=443,
#             chroma_server_ssl_enabled=True,
#             anonymized_telemetry=False,
#             chroma_api_key=api_key,
#         )
#     )

#     # Step 2: Try to get the collection
#     try:
#         collection = client.get_collection(name=collection_name)
#         print(f"✅ Successfully retrieved collection: '{collection_name}'")
#         return collection
#     except Exception as e:
#         print(f"❌ Error retrieving collection: {e}")
#         return None
    



# RETRIEVAL:

def standard_retriever(vectorstore):
    """Creates a basic retriever from the vectorstore."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})


def get_relevant_texts(vectorstore, query):
    """
    Uses a standard vector store retriever to fetch relevant text chunks for a query.

    Args:
        vectorstore: The vector store to search in.
        query (str): The user's search query.

    Returns:
        list[str]: A list of relevant text snippets.
    """
    if vectorstore is None:
        return ["No vector store provided."]
    
    retriever = standard_retriever(vectorstore)
    results = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in results]


def retrieve_docs(query: str, vector_store) -> list[str]:
    """
    Retrieves documents based on the provided query, utilizing a cache for efficiency.

    This function first checks if the results for the given query are already cached. 
    If cached results are found, they are returned immediately. 
    If not, it retrieves the results from the vector_store using semantic similarity search,
    caches them, and returns the results.

    Args:
        query (str): The query string used to search for relevant documents.
        vector_store: The vector store (e.g., Chroma) containing embedded document chunks.

    Returns:
        list[str]: A list of relevant text snippets (strings) matching the query.
    """
    # Initialize disk-based cache
    cache = Cache("./cache")

    # Check if the query result exists in cache
    if cache.get(query) is not None:
        return cache.get(query)  # Return cached results

    # Retrieve results from the vector store
    results = get_relevant_texts(vector_store, query)

    # Cache the results with a 10-minute expiration
    cache.set(query, results, expire=600)

    # Return list of document strings
    return results




from duckduckgo_search import DDGS

def internet_search_tool(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        if results:
            return results[0]['body']  # Or results[0]['href'] to get the URL
        return "No results found."


def document_search_tool(query,vector_store):
    """Searches the document for general information.

    Args:
        query (str): The query string to search on the document.

    Returns:
        str: The result of the document search.
    """
    document_tool=retrieve_docs(query,vector_store)
    return document_tool



# Main Code:



def generate_insurance_answer(query: str, vector_store) -> str:
    """
    Combines document and internet search results to generate a response using LLaMA 3.1.

    Args:
        query (str): The user's question.
        vector_store: The internal document vector store.

    Returns:
        str: The model's response based on the combined context.
    """
    # Step 1: Internal document search
    doc_result = document_search_tool(query, vector_store)
    print("Document result:", doc_result)

    # Step 2: Internet search
    internet_result = internet_search_tool(query)
    print("Internet result:", internet_result)

    # Step 3: Combine context
    combined_context = f"""
You are an expert in insurance. A user has asked the following question:

Question: "{query}"

Here is what we found in internal documents:
{doc_result}

Here is what we found on the internet:
{internet_result}

Do consider the internal document as the main source of information unless the question is asking for some kind of definition.

Based on both sources, provide a helpful and comprehensive answer.
"""

    # Step 4: Initialize client (e.g. Fireworks.ai)
    api_key = os.environ.get("HF_TOKEN")
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=api_key
    )

    # Step 5: Call the model
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": combined_context}]
    )

    # Step 6: Return result
    return completion.choices[0].message.content
