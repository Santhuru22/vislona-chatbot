import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re


# LangChain imports
try:
    from langchain.docstore.document import Document
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LangChainSplitter
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain libraries loaded successfully")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"❌ LangChain not available: {e}")
    print("Install with: pip install langchain langchain-community faiss-cpu")

class VislonaVectorStoreProcessor:
    """
    Process saved Vislona chunks and create FAISS vector store for semantic search using Ollama
    """
    
    def __init__(self, 
                 chunks_directory: str = r"D:\vislona\chatbot\dataset\chunks",
                 ollama_model: str = "nomic-embed-text",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the vector store processor
        
        Args:
            chunks_directory: Directory containing saved chunk files
            ollama_model: Ollama embedding model to use (default: nomic-embed-text)
            ollama_base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.chunks_directory = Path(chunks_directory)
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        
        print(f"🦙 Using Ollama model: {ollama_model}")
        print(f"🌐 Ollama server: {ollama_base_url}")
    
    def load_chunks_from_files(self) -> List[Dict[str, Any]]:
        """
        Load all chunk files from the chunks directory
        
        Returns:
            List of dictionaries containing chunk data
        """
        if not self.chunks_directory.exists():
            print(f"❌ Chunks directory not found: {self.chunks_directory}")
            return []
        
        chunk_files = list(self.chunks_directory.glob("chunk_*.txt"))
        chunk_files.sort()  # Sort by filename for consistent ordering
        
        chunks_data = []
        
        print(f"📁 Loading {len(chunk_files)} chunk files...")
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse chunk metadata from file header
                metadata = self._parse_chunk_metadata(content)
                
                # Extract actual content (remove header)
                actual_content = self._extract_chunk_content(content)
                
                if actual_content.strip():  # Only add non-empty chunks
                    chunk_data = {
                        "content": actual_content,
                        "file_path": str(chunk_file),
                        "file_name": chunk_file.name,
                        **metadata
                    }
                    chunks_data.append(chunk_data)
                    
            except Exception as e:
                print(f"❌ Error loading {chunk_file}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(chunks_data)} chunks")
        return chunks_data
    
    def _parse_chunk_metadata(self, content: str) -> Dict[str, Any]:
        """Parse metadata from chunk file header"""
        metadata = {}
        
        # Extract chunk ID
        chunk_id_match = re.search(r'# Chunk (\d+)', content)
        if chunk_id_match:
            metadata['chunk_id'] = int(chunk_id_match.group(1))
        
        # Extract source
        source_match = re.search(r'# Source: (.+)', content)
        if source_match:
            metadata['source'] = source_match.group(1).strip()
        
        # Extract position
        position_match = re.search(r'# Position: (\d+)-(\d+)', content)
        if position_match:
            metadata['start_index'] = int(position_match.group(1))
            metadata['end_index'] = int(position_match.group(2))
        
        # Extract length
        length_match = re.search(r'# Length: (\d+) characters', content)
        if length_match:
            metadata['chunk_size'] = int(length_match.group(1))
        
        # Add default metadata
        metadata.update({
            "source_type": "vislona_chatbot_dataset",
            "document_type": "chatbot_training_data",
            "company": "Vislona AI Platform",
            "processing_date": "2025-09-02"
        })
        
        return metadata
    
    def _extract_chunk_content(self, content: str) -> str:
        """Extract actual content from chunk file (remove header)"""
        # Find the end of metadata header (empty line after last # comment)
        lines = content.split('\n')
        content_start = 0
        
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.strip() == '':
                content_start = i + 1
                break
            elif not line.startswith('#') and line.strip():
                content_start = i
                break
        
        return '\n'.join(lines[content_start:]).strip()
    
    def create_langchain_documents(self, chunks_data: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert chunk data to LangChain Document objects
        
        Args:
            chunks_data: List of chunk dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        
        documents = []
        
        for chunk_data in chunks_data:
            # Create metadata copy (remove content for metadata)
            metadata = {k: v for k, v in chunk_data.items() if k != 'content'}
            
            # Create LangChain Document
            doc = Document(
                page_content=chunk_data['content'],
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} LangChain Document objects")
        return documents
    
    def test_ollama_connection(self) -> Tuple[bool, List[str]]:
        """
        Test connection to Ollama server and check available models
        
        Returns:
            Tuple of (connection_success, available_models)
        """
        try:
            import requests
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                print(f"🦙 Available Ollama models: {', '.join(model_names)}")
                
                if self.ollama_model in model_names:
                    print(f"✅ Embedding model '{self.ollama_model}' is available")
                    return True, model_names
                else:
                    print(f"⚠️  Embedding model '{self.ollama_model}' not found")
                    
                    # Check for alternative embedding models
                    embedding_models = [
                        "nomic-embed-text", "mxbai-embed-large", "all-minilm",
                        "snowflake-arctic-embed", "bge-large", "bge-base"
                    ]
                    
                    available_embedding_models = [m for m in model_names if any(em in m for em in embedding_models)]
                    
                    if available_embedding_models:
                        print(f"✅ Found alternative embedding models: {', '.join(available_embedding_models)}")
                        # Use the first available embedding model
                        self.ollama_model = available_embedding_models[0]
                        print(f"🔄 Switching to: {self.ollama_model}")
                        return True, model_names
                    else:
                        print(f"💡 Pull the recommended model with: ollama pull {self.ollama_model}")
                        print(f"🔄 Or try using any available model for basic embeddings")
                        return False, model_names
            else:
                print(f"❌ Failed to connect to Ollama server at {self.ollama_base_url}")
                return False, []
                
        except Exception as e:
            print(f"❌ Error testing Ollama connection: {e}")
            print(f"💡 Make sure Ollama is running: ollama serve")
            return False, []
    
    def create_faiss_vectorstore(self, 
                                documents: List[Document], 
                                save_path: str = "faiss_index_ollama") -> Optional[FAISS]:
        """
        Create FAISS vector store from documents using Ollama embeddings
        
        Args:
            documents: List of LangChain Document objects
            save_path: Path to save the vector store
            
        Returns:
            FAISS vector store or None if failed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        
        # Test Ollama connection first
        connection_success, available_models = self.test_ollama_connection()
        if not connection_success:
            print("❌ Cannot proceed without Ollama connection")
            return None
        
        try:
            print(f"🔍 Creating FAISS vector store with Ollama embeddings...")
            
            # Create Ollama embeddings
            embeddings = OllamaEmbeddings(
                model=self.ollama_model,
                base_url=self.ollama_base_url
            )
            
            # Test embeddings with a sample text
            print("🧪 Testing embeddings...")
            test_embedding = embeddings.embed_query("test query")
            print(f"✅ Embedding dimension: {len(test_embedding)}")
            
            # Create FAISS vector store from documents
            print("🏗️  Building vector store...")
            vector_store = FAISS.from_documents(documents, embeddings)
            
            # Save vector store locally
            vector_store.save_local(save_path)
            print(f"💾 Vector store saved to: {save_path}")
            
            # Test loading the vector store
            print("🧪 Testing vector store loading...")
            test_vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            print(f"✅ Vector store created and tested successfully!")
            print(f"📊 Total documents in vector store: {len(documents)}")
            
            return vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            if "connection" in str(e).lower():
                print("💡 Make sure Ollama is running: ollama serve")
            elif "model" in str(e).lower():
                print(f"💡 Make sure the model is available: ollama pull {self.ollama_model}")
            return None
    
    def query_vectorstore(self, 
                         vector_store: FAISS, 
                         query: str, 
                         k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents
        
        Args:
            vector_store: FAISS vector store
            query: Query string
            k: Number of results to return
            
        Returns:
            List of results with content, score, and metadata
        """
        try:
            # Perform similarity search with scores
            results = vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown")
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error querying vector store: {e}")
            return []
    
    def run_sample_queries(self, vector_store: FAISS) -> None:
        """Run sample queries on the vector store"""
        
        sample_queries = [
            "What is Vislona?",
            "How do I train AI models?",
            "What are the pricing plans?",
            "How does team collaboration work?",
            "What file formats are supported?",
            "How do I deploy my model?",
            "What security features are available?",
            "How do I get an internship?"
        ]
        
        print("\n🔍 Running Sample Queries:")
        print("=" * 60)
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n❓ Query {i}: {query}")
            print("-" * 40)
            
            results = self.query_vectorstore(vector_store, query, k=2)
            
            for j, result in enumerate(results, 1):
                print(f"📄 Result {j} (Score: {result['similarity_score']:.4f}, Chunk: {result['chunk_id']}):")
                content_preview = result['content'][:200].replace('\n', ' ')
                print(f"   {content_preview}{'...' if len(result['content']) > 200 else ''}")
            
            if not results:
                print("   No results found")

def main():
    """
    Main function to process Vislona chunks and create vector store using Ollama
    """
    print("🚀 Starting Vislona Vector Store Creation Process (Ollama)")
    print("=" * 60)
    
    # Initialize processor with custom Ollama settings if needed
    processor = VislonaVectorStoreProcessor(
        ollama_model="nomic-embed-text",  # Good embedding model for text
        ollama_base_url="http://localhost:11434"  # Default Ollama URL
    )
    
    # Load chunks from saved files
    chunks_data = processor.load_chunks_from_files()
    
    if not chunks_data:
        print("❌ No chunks found. Make sure chunks are saved first.")
        return
    
    # Create LangChain documents
    documents = processor.create_langchain_documents(chunks_data)
    
    # Create FAISS vector store
    vector_store = processor.create_faiss_vectorstore(documents)
    
    if vector_store:
        # Run sample queries to test the vector store
        processor.run_sample_queries(vector_store)
        
        print(f"\n🎉 Vector Store Creation Complete!")
        print(f"📊 {len(documents)} documents processed")
        print(f"💾 Vector store saved as 'faiss_index_ollama'")
        print(f"🔍 Ready for semantic search queries")
        
        # Example of how to use the vector store
        print(f"\n💡 Usage Example:")
        print("=" * 30)
        print("from langchain_community.embeddings import OllamaEmbeddings")
        print("from langchain.vectorstores import FAISS")
        print("")
        print("# Load the vector store")
        print("embeddings = OllamaEmbeddings(model='nomic-embed-text')")
        print("vector_store = FAISS.load_local('faiss_index_ollama', embeddings, allow_dangerous_deserialization=True)")
        print("")
        print("# Query the vector store")
        print("results = vector_store.similarity_search('What is Vislona?', k=3)")
        print("for result in results:")
        print("    print(result.page_content)")
        
    else:
        print("❌ Vector store creation failed")
    
    print("\n" + "=" * 60)

# Custom query function for interactive use
def query_vislona_vectorstore(query: str, 
                             k: int = 3, 
                             vector_store_path: str = "faiss_index_ollama",
                             ollama_model: str = "nomic-embed-text",
                             ollama_base_url: str = "http://localhost:11434"):
    """
    Quick function to query the Vislona vector store using Ollama
    
    Args:
        query: Question to ask
        k: Number of results
        vector_store_path: Path to saved vector store
        ollama_model: Ollama embedding model
        ollama_base_url: Ollama server URL
    """
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain not available")
        return
    
    try:
        # Load vector store with Ollama embeddings
        embeddings = OllamaEmbeddings(
            model=ollama_model,
            base_url=ollama_base_url
        )
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Query
        results = vector_store.similarity_search_with_score(query, k=k)
        
        print(f"🔍 Query: {query}")
        print("=" * 50)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n📄 Result {i} (Score: {score:.4f}):")
            print(f"Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}")
            print(f"Content: {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Error querying vector store: {e}")
        if "connection" in str(e).lower():
            print("💡 Make sure Ollama is running: ollama serve")
        elif "model" in str(e).lower():
            print(f"💡 Make sure the model is available: ollama pull {ollama_model}")

def check_ollama_models():
    """
    Utility function to check available Ollama models
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("🦙 Available Ollama models:")
            for model in models:
                print(f"   • {model['name']} ({model.get('size', 'unknown size')})")
        else:
            print("❌ Cannot connect to Ollama server")
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")

def setup_ollama_embeddings():
    """
    Helper function to set up recommended embedding models for Ollama
    """
    print("🦙 Setting up Ollama for embeddings...")
    print("\n📋 Recommended embedding models:")
    print("1. nomic-embed-text - General purpose text embeddings (recommended)")
    print("2. mxbai-embed-large - Large embedding model for better accuracy")
    print("3. all-minilm - Lightweight option")
    print("4. snowflake-arctic-embed - High-quality embeddings")
    print("\n💡 To install the recommended model, run:")
    print("   ollama pull nomic-embed-text")
    print("\n🚀 To start Ollama server:")
    print("   ollama serve")
    print("\n⚡ Quick setup commands:")
    print("   ollama pull nomic-embed-text")
    print("   # Then run your Python script")

def use_existing_model_for_embeddings(model_name: str = "gemma3:1b"):
    """
    Alternative approach: Use an existing chat model for embeddings
    Note: This is not ideal but can work as a fallback
    """
    print(f"\n🔄 Alternative: Using {model_name} for embeddings")
    print("⚠️  Note: Chat models aren't optimized for embeddings, but this can work as a temporary solution")
    
    processor = VislonaVectorStoreProcessor(
        ollama_model=model_name,
        ollama_base_url="http://localhost:11434"
    )
    return processor

if __name__ == "__main__":
    # Check if Ollama is set up
    print("🔧 Checking Ollama setup...")
    check_ollama_models()
    print("\n")
    
    # Run main process
    main()