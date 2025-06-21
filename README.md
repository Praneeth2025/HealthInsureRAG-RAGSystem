
![Document](https://github.com/user-attachments/assets/69cb9798-dc7e-4172-b5a1-30aeda6ae742)

---

# 📄 Insurance Policy QA Chatbot with LangChain, ChromaDB & Streamlit

An AI-powered document-based **Question Answering (QA)** and **Summarization** chatbot built for **insurance policy PDFs**. It uses advanced language models and retrieval techniques to help users understand dense policy documents via natural language questions and concise summaries.



## 🚀 Features

* 📄 **PDF Parsing with Context Awareness**\
  Extracts structured data from PDFs using `pdfplumber` and `PyMuPDF`, retaining formatting like **headings** and **bolded phrases**.

* ✂️ **Chunking with Contextual Embedding**\
  Text is split using `RecursiveCharacterTextSplitter` and prefixed with its **parent heading and highlights**, improving semantic understanding.

* 🧠 **Embeddings via SentenceTransformers**\
  Text chunks are embedded using `all-MiniLM-L6-v2` for high-quality semantic similarity matching.

* 📾 **ChromaDB Vector Store**\
  Stores vector representations of document chunks persistently using `Chroma`.

* 🔍 **Contextual Compression Retriever**\
  A 2-stage retrieval process using:

  * **Similarity-based retriever**\
  * **LLM compressor** for relevance filtering

* 🧠 **Mistral-7B QA LLM**\
  Uses `mistralai/Mistral-7B-Instruct-v0.1` to generate accurate, explainable answers based on compressed context.

* 🌐 **Web Search Tool**\
  Falls back to DuckDuckGo for external search when the document lacks enough information.

* 💡 **Summary Generation**\
  Users can generate a **document summary** highlighting key coverage points, exclusions, and important clauses using the LLM — helpful for quick overviews.

* 💾 **Disk-Based Caching**\
  Query responses are cached using `diskcache` to improve performance on repeated searches.

* 🖥️ **Streamlit Interface**\
  A polished web interface built with Streamlit supports PDF uploads, QA, and summary generation in real-time.

---

## 🛠️ Tech Stack

| Layer           | Tool/Library                               |
| --------------- | ------------------------------------------ |
| Text Extraction | `pdfplumber`, `fitz` (PyMuPDF)             |
| Chunking        | `LangChain.text_splitter`                  |
| Embeddings      | `sentence-transformers`                    |
| Vector Store    | `ChromaDB`                                 |
| LLM             | `Mistral-7B-Instruct` via `HuggingFaceHub` |
| Retrieval       | `LangChain` retrievers + compression       |
| Web Search      | `DuckDuckGoSearchRun`                      |
| Caching         | `diskcache`                                |
| UI              | `Streamlit`                                |

---

## 📁 Folder Structure

```
.
├── final_project.ipynb             # Core QA & summary pipeline
├── app.py                          # Streamlit web application
├── chroma_store/                   # Persistent Chroma vector store
├── cache/                          # Disk-based cache storage
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/insurance-qa-chatbot.git
cd insurance-qa-chatbot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🧪 Notebook-Based Usage

```python
query = "What are the exclusions under the critical illness plan?"
results = document_search_tool(query, vectorstore)
print(results)
```

```python
# To generate a summary
summary = generate_summary(vectorstore)
print(summary)
```

### 🖥️ Streamlit App

```bash
streamlit run app.py
```

* 📤 Upload any insurance PDF
* 💬 Ask questions interactively
* 📌 Click “Generate Summary” to get a concise explanation of the entire policy



