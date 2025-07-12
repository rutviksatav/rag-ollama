import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader, Docx2txtLoader,
    CSVLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
)
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleRAG:
    """
    A simple RAG (Retrieval-Augmented Generation) system using Ollama models.
    
    This class provides functionality to:
    - Load documents from various file formats
    - Split documents into chunks
    - Create and manage vector stores
    - Setup retrieval chains
    - Ask questions and get answers with sources
    """
    
    SUPPORTED_FILE_TYPES = {
        "docx": {"loader": Docx2txtLoader, "glob": "**/*.docx"},
        "pdf": {"loader": PyPDFLoader, "glob": "**/*.pdf"},
        "txt": {"loader": TextLoader, "glob": "**/*.txt"},
        "csv": {"loader": CSVLoader, "glob": "**/*.csv"},
        "md": {"loader": UnstructuredMarkdownLoader, "glob": "**/*.md"},
        "pptx": {"loader": UnstructuredPowerPointLoader, "glob": "**/*.pptx"},
    }
    
    def __init__(self, 
                 embedding_model: str = "all-minilm", 
                 llm_model: str = "llama2",
                 temperature: float = 0.0,
                 base_url: Optional[str] = None):
        """
        Initialize the RAG system with Ollama models.
        
        Args:
            embedding_model: Name of the embedding model to use
            llm_model: Name of the LLM model to use
            temperature: Temperature for the LLM (0.0 for deterministic responses)
            base_url: Base URL for Ollama server (defaults to localhost:11434)
        """
        try:
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=base_url or "http://localhost:11434"
            )
            self.llm = ChatOllama(
                model=llm_model,
                temperature=temperature,
                base_url=base_url or "http://localhost:11434"
            )
            self.vector_store = None
            self.chain = None
            logger.info(f"RAG system initialized with embedding model: {embedding_model}, LLM: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise
        
    def load_documents(self, 
                      directory_path: Union[str, Path], 
                      file_type: str = "docx",
                      recursive: bool = True) -> List[Document]:
        """
        Load documents from directory based on file type.
        
        Args:
            directory_path: Path to the directory containing documents
            file_type: Type of files to load (docx, pdf, txt, csv, md, pptx)
            recursive: Whether to search recursively in subdirectories
            
        Returns:
            List of loaded documents
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        if file_type not in self.SUPPORTED_FILE_TYPES:
            supported_types = ", ".join(self.SUPPORTED_FILE_TYPES.keys())
            logger.error(f"Unsupported file type: {file_type}. Supported types: {supported_types}")
            return []
        
        try:
            file_config = self.SUPPORTED_FILE_TYPES[file_type]
            glob_pattern = file_config["glob"] if recursive else f"*.{file_type}"
            
            loader = DirectoryLoader(
                path=str(directory_path),
                glob=glob_pattern,
                loader_cls=file_config["loader"]
            )
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {directory_path}: {str(e)}")
            return []
    
    def split_documents(self, 
                       documents: List[Document], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200,
                       splitter_type: str = "character") -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            splitter_type: Type of splitter ("character" or "recursive")
            
        Returns:
            List of document chunks
        """
        try:
            if splitter_type == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
            else:
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            
            chunks = splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return []
    
    def create_vector_store(self, chunks: List[Document], index_name: str = "faiss_index"):
        """
        Create FAISS vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            index_name: Name for the vector store index
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for vector store creation")
                return
            
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Vector store created successfully with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vector_store(self, path: Union[str, Path]):
        """Save vector store to disk"""
        if not self.vector_store:
            logger.warning("No vector store to save")
            return
        
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(path))
            logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, path: Union[str, Path]):
        """Load vector store from disk"""
        try:
            path = Path(path)
            if not path.exists():
                logger.error(f"Vector store path does not exist: {path}")
                return
            
            self.vector_store = FAISS.load_local(
                str(path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def setup_chain(self, k: int = 4, system_prompt: Optional[str] = None):
        """
        Setup the retrieval chain.
        
        Args:
            k: Number of documents to retrieve
            system_prompt: Custom system prompt (optional)
        """
        if not self.vector_store:
            logger.error("Vector store not initialized. Please create or load a vector store first.")
            return
        
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            # Create prompt template
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Keep the answer concise and accurate.\n\n"
                    "{context}"
                )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Create the question-answer chain
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # Create the retrieval chain
            self.chain = create_retrieval_chain(retriever, question_answer_chain)
            
            logger.info("RAG chain setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up chain: {str(e)}")
            raise
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source information
        """
        if not self.chain:
            return {"error": "Chain not initialized. Please setup chain first."}
        
        try:
            result = self.chain.invoke({"input": question})
            
            response = {
                "question": question,
                "answer": result["answer"],
                "sources": []
            }
            
            # Extract source information
            for doc in result["context"]:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                response["sources"].append(source_info)
            
            logger.info(f"Successfully processed question: {question[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {"error": f"Error processing question: {str(e)}"}
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} similar documents for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store"""
        if not self.vector_store:
            return {"error": "No vector store initialized"}
        
        try:
            # Get basic info about the vector store
            info = {
                "index_type": "FAISS",
                "embedding_model": self.embeddings.model,
                "llm_model": self.llm.model,
                "chain_initialized": self.chain is not None
            }
            
            # Try to get document count (this might not be available for all vector stores)
            try:
                if hasattr(self.vector_store, 'index'):
                    info["document_count"] = self.vector_store.index.ntotal
            except:
                info["document_count"] = "Unknown"
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting vector store info: {str(e)}")
            return {"error": f"Error getting vector store info: {str(e)}"}
