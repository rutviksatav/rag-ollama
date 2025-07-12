#!/usr/bin/env python3
"""
Simple RAG Demo - Basic usage of SimpleRAG class

This script demonstrates the basic RAG flow:
1. Initialize RAG system
2. Load documents
3. Split documents into chunks
4. Create vector store
5. Setup retrieval chain
6. Ask questions and get answers
"""

from rag import SimpleRAG


def main():
    """Demonstrate basic RAG functionality"""
    print("Simple RAG Demo")
    print("=" * 40)
    
    # Step 1: Initialize RAG system
    print("1. Initializing RAG system...")
    rag = SimpleRAG(
        embedding_model="all-minilm",
        llm_model="llama2",
        temperature=0.0
    )
    print("RAG system initialized successfully")
    
    # Step 2: Load documents
    print("\n2. Loading documents...")
    documents = rag.load_documents("./documents", file_type="pdf")
    
    if not documents:
        print("ERROR: No documents found in ./documents folder")
        print("Please add some PDF files to the documents folder")
        return
    
    print(f"Loaded {len(documents)} documents successfully")
    
    # Step 3: Split documents into chunks
    print("\n3. Splitting documents into chunks...")
    chunks = rag.split_documents(
        documents=documents,
        chunk_size=1000,
        chunk_overlap=200
    )
    print(f"Split into {len(chunks)} chunks successfully")
    
    # Step 4: Create vector store
    print("\n4. Creating vector store...")
    rag.create_vector_store(chunks)
    print("Vector store created successfully")
    
    # Step 5: Setup retrieval chain
    print("\n5. Setting up retrieval chain...")
    rag.setup_chain(k=4)
    print("Retrieval chain ready")
    
    # Step 6: Interactive question asking
    print("\n6. Ready to answer questions!")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        try:
            # Get user input
            question = input("\nEnter your question: ").strip()
            
            # Check for exit command
            if question.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            # Skip empty questions
            if not question:
                continue
            
            # Ask the question
            print(f"\nProcessing: {question}")
            result = rag.ask(question)
            
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"Answer: {result['answer']}")
                print(f"Sources: {len(result['sources'])} documents")
                
                # Show source information
                if result['sources']:
                    print("\nSource documents:")
                    for i, source in enumerate(result['sources'], 1):
                        content = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                        print(f"  {i}. {content}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main() 