# Simple RAG with Ollama

A simple Retrieval-Augmented Generation (RAG) system using Ollama models for document-based question answering.

## Quick Start

### 1. Install Ollama
```bash
# Download from https://ollama.ai/download
# Then pull required models:
ollama pull llama2
ollama pull all-minilm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Documents
```bash
mkdir documents
# Add your PDF, DOCX, or TXT files to the documents folder
```

### 4. Run the Demo
```bash
python main.py
```

## Supported File Types
- PDF (`.pdf`)
- Word (`.docx`) 
- Text (`.txt`)
- CSV (`.csv`)
- Markdown (`.md`)
- PowerPoint (`.pptx`)

## API Usage

```python
from rag import SimpleRAG

# Initialize
rag = SimpleRAG()

# Load and process documents
documents = rag.load_documents("./documents", file_type="pdf")
chunks = rag.split_documents(documents)
rag.create_vector_store(chunks)
rag.setup_chain()

# Ask questions
result = rag.ask("What is this about?")
print(result["answer"])
```

## Configuration

```python
rag = SimpleRAG(
    embedding_model="all-minilm",
    llm_model="llama2", 
    temperature=0.0
)
```

## License

MIT License