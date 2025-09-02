import re
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# For PDF processing (install with: pip install PyPDF2 or pip install pdfplumber)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("Warning: PDF processing libraries not available. Install PyPDF2 or pdfplumber for PDF support.")

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    start_index: int
    end_index: int
    chunk_id: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RecursiveCharacterTextSplitter:
    """
    A text splitter that recursively splits text using different separators
    to maintain semantic coherence while respecting chunk size limits.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        length_function: callable = len
    ):
        """
        Initialize the RecursiveCharacterTextSplitter
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (in order of preference)
            keep_separator: Whether to keep the separator in the chunks
            is_separator_regex: Whether separators are regex patterns
            length_function: Function to calculate text length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self.length_function = length_function
        
        # Default separators in order of preference
        if separators is None:
            self.separators = [
                "\n\n",    # Double newlines (paragraphs)
                "\n",      # Single newlines
                " ",       # Spaces
                ""         # Characters (last resort)
            ]
        else:
            self.separators = separators
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Split text into chunks using recursive character splitting
        
        Args:
            text: The text to split
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}
            
        return self._split_text_recursive(text, 0, metadata)
    
    def _split_text_recursive(
        self, 
        text: str, 
        start_offset: int = 0, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """
        Recursively split text using different separators
        """
        chunks = []
        
        # If text is small enough, return as single chunk
        if self.length_function(text) <= self.chunk_size:
            if text.strip():  # Only create chunk if text is not empty
                chunk = TextChunk(
                    content=text,
                    start_index=start_offset,
                    end_index=start_offset + len(text),
                    chunk_id=0,
                    metadata=metadata.copy() if metadata else {}
                )
                chunks.append(chunk)
            return chunks
        
        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # Last resort: split by characters
                return self._split_by_characters(text, start_offset, metadata)
            
            # Split by current separator
            splits = self._split_by_separator(text, separator)
            
            if len(splits) == 1:
                # No splitting occurred, try next separator
                continue
            
            # Process splits
            current_chunks = []
            current_text = ""
            current_start = start_offset
            
            for i, split in enumerate(splits):
                # Calculate the length if we add this split
                test_text = current_text + split if current_text else split
                
                if self.length_function(test_text) <= self.chunk_size:
                    # Add to current chunk
                    current_text = test_text
                else:
                    # Current text exceeds limit
                    if current_text:
                        # Save current chunk and start new one
                        chunk = TextChunk(
                            content=current_text,
                            start_index=current_start,
                            end_index=current_start + len(current_text),
                            chunk_id=len(current_chunks),
                            metadata=metadata.copy() if metadata else {}
                        )
                        current_chunks.append(chunk)
                        
                        # Handle overlap
                        overlap_text = self._get_overlap_text(current_text)
                        current_start += len(current_text) - len(overlap_text)
                        current_text = overlap_text
                    
                    # If this split is still too large, recursively split it
                    if self.length_function(split) > self.chunk_size:
                        split_start = current_start + len(current_text)
                        recursive_chunks = self._split_text_recursive(
                            split, split_start, metadata
                        )
                        current_chunks.extend(recursive_chunks)
                        current_text = ""
                        current_start = split_start + len(split)
                    else:
                        current_text += split
            
            # Don't forget the last chunk
            if current_text.strip():
                chunk = TextChunk(
                    content=current_text,
                    start_index=current_start,
                    end_index=current_start + len(current_text),
                    chunk_id=len(current_chunks),
                    metadata=metadata.copy() if metadata else {}
                )
                current_chunks.append(chunk)
            
            # Update chunk IDs
            for i, chunk in enumerate(current_chunks):
                chunk.chunk_id = i
            
            return current_chunks
        
        # If we get here, return the text as is (shouldn't happen)
        return [TextChunk(
            content=text,
            start_index=start_offset,
            end_index=start_offset + len(text),
            chunk_id=0,
            metadata=metadata.copy() if metadata else {}
        )]
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a given separator"""
        if self.is_separator_regex:
            splits = re.split(separator, text)
        else:
            splits = text.split(separator)
        
        if not self.keep_separator or separator == "":
            return [s for s in splits if s]  # Remove empty strings
        
        # Reconstruct with separators
        result = []
        for i, split in enumerate(splits):
            if i == 0:
                result.append(split)
            elif split:  # Don't add empty splits
                result.append(separator + split)
            elif separator:  # Add separator even if split is empty
                result.append(separator)
        
        return [s for s in result if s]  # Remove any empty strings
    
    def _split_by_characters(
        self, 
        text: str, 
        start_offset: int = 0, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Split text by individual characters (last resort)"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunk = TextChunk(
                content=chunk_text,
                start_index=start_offset + start,
                end_index=start_offset + end,
                chunk_id=chunk_id,
                metadata=metadata.copy() if metadata else {}
            )
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
            chunk_id += 1
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current text"""
        if self.chunk_overlap <= 0 or len(text) <= self.chunk_overlap:
            return ""
        
        return text[-self.chunk_overlap:]
    
    def process_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Process a file and split it into chunks
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Add file information to metadata
        file_path_obj = Path(file_path)
        metadata.update({
            "file_path": str(file_path_obj.absolute()),
            "file_name": file_path_obj.name,
            "file_extension": file_path_obj.suffix.lower()
        })
        
        # Extract text based on file type
        if file_path_obj.suffix.lower() == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_path_obj.suffix.lower() in ['.txt', '.md']:
            text = self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path_obj.suffix}")
        
        return self.split_text(text, metadata)
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing library not available. Install PyPDF2 or pdfplumber.")
        
        text = ""
        
        # Try pdfplumber first (better text extraction)
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except ImportError:
            # Fallback to PyPDF2
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except ImportError:
                raise ImportError("No PDF processing library available. Install PyPDF2 or pdfplumber.")
        
        return text
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_lengths),
            "avg_chunk_size": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_size": min(chunk_lengths),
            "max_chunk_size": max(chunk_lengths),
            "original_text_length": chunks[-1].end_index if chunks else 0
        }
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin1') as file:
                return file.read()
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_lengths),
            "avg_chunk_size": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_size": min(chunk_lengths),
            "max_chunk_size": max(chunk_lengths),
            "original_text_length": chunks[-1].end_index if chunks else 0
        }

# Vislona-specific processing function
def process_vislona_chatbot_dataset(file_path: str = r"D:\vislona\chatbot\dataset\chatbot data.pdf"):
    """
    Process the Vislona chatbot dataset from the specified file path
    
    Args:
        file_path: Path to the Vislona chatbot dataset PDF file
        
    Returns:
        List of TextChunk objects containing the processed dataset
    """
    print(f"Processing Vislona chatbot dataset from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    
    # Initialize splitter optimized for chatbot Q&A format
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Good size for Q&A pairs
        chunk_overlap=100,  # Moderate overlap to preserve context
        separators=[
            "\n## ",      # Section headers
            "\n\n",       # Paragraph breaks
            "\nQ:",       # Questions
            "\nA:",       # Answers
            ". ",         # Sentences
            "\n",         # Line breaks
            " ",          # Word boundaries
            ""            # Character level (last resort)
        ],
        keep_separator=True
    )
    
    try:
        # Process the file
        chunks = splitter.process_file(
            file_path, 
            metadata={
                "source": "vislona_chatbot_dataset",
                "document_type": "chatbot_training_data",
                "company": "Vislona AI Platform",
                "processing_date": "2025-09-02"
            }
        )
        
        print(f"\n✅ Successfully processed {len(chunks)} chunks from Vislona dataset")
        
        # Display statistics
        stats = splitter.get_chunk_statistics(chunks)
        print("\n📊 Dataset Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.1f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Show sample chunks
        print(f"\n📝 Sample Chunks:")
        print("=" * 50)
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1} (ID: {chunk.chunk_id}):")
            print(f"Characters: {len(chunk.content)}")
            print(f"Position: {chunk.start_index}-{chunk.end_index}")
            print(f"Content Preview: {chunk.content[:150]}{'...' if len(chunk.content) > 150 else ''}")
            print("-" * 30)
        
        if len(chunks) > 3:
            print(f"... and {len(chunks) - 3} more chunks")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return []

def save_chunks_to_files(chunks: List[TextChunk], output_dir: str = r"D:\vislona\chatbot\dataset\chunks"):
    """
    Save chunks to individual text files for further processing
    
    Args:
        chunks: List of TextChunk objects to save
        output_dir: Directory to save chunk files
    """
    if not chunks:
        print("No chunks to save.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving {len(chunks)} chunks to: {output_dir}")
    
    for chunk in chunks:
        # Create filename
        filename = f"chunk_{chunk.chunk_id:03d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Save chunk content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Chunk {chunk.chunk_id}\n")
            f.write(f"# Source: {chunk.metadata.get('file_name', 'Unknown')}\n")
            f.write(f"# Position: {chunk.start_index}-{chunk.end_index}\n")
            f.write(f"# Length: {len(chunk.content)} characters\n\n")
            f.write(chunk.content)
    
    print(f"✅ Successfully saved {len(chunks)} chunk files")

def create_chunk_index(chunks: List[TextChunk], output_file: str = r"D:\vislona\chatbot\dataset\chunk_index.txt"):
    """
    Create an index file listing all chunks with metadata
    
    Args:
        chunks: List of TextChunk objects
        output_file: Path to save the index file
    """
    if not chunks:
        print("No chunks to index.")
        return
    
    print(f"\n📋 Creating chunk index: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Vislona Chatbot Dataset - Chunk Index\n")
        f.write(f"# Generated on: 2025-09-02\n")
        f.write(f"# Total chunks: {len(chunks)}\n\n")
        
        for chunk in chunks:
            f.write(f"Chunk {chunk.chunk_id:03d}:\n")
            f.write(f"  - File: {chunk.metadata.get('file_name', 'Unknown')}\n")
            f.write(f"  - Position: {chunk.start_index}-{chunk.end_index}\n")
            f.write(f"  - Length: {len(chunk.content)} chars\n")
            f.write(f"  - Preview: {chunk.content[:100].replace(chr(10), ' ')[:97]}{'...' if len(chunk.content) > 100 else ''}\n\n")
    
    print(f"✅ Successfully created chunk index")

# Example usage with the Vislona dataset
def process_vislona_dataset():
    """Example of how to use the splitter with the Vislona dataset"""
    
    # Sample text from the Vislona dataset (you would load the full document)
    vislona_text = """# Vislona AI Platform - Chatbot Training Dataset
## Company Information
Q: What is Vislona?
A: Vislona is a next-gen AI company that builds smart, subscription-based tools and agents for modern users without traditional code. We make AI creation simple, accessible, and beautiful.

Q: Is Vislona a single product or a platform?
A: Vislona is not a single app - it's a parent company that launches multiple AI-powered tools, each focused on a specific need or niche.

Q: What kind of tools does Vislona create?
A: We create AI-driven apps for productivity, personal care, lifestyle enhancement, content creation, and intelligent agents - all accessible without any coding."""
    
    # Initialize the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "Q:", "A:", " ", ""]
    )
    
    # Split the text
    chunks = splitter.split_text(
        vislona_text, 
        metadata={"source": "vislona_dataset", "document_type": "chatbot_training"}
    )
    
    # Print results
    print(f"Created {len(chunks)} chunks:")
    print("\n" + "="*50 + "\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (chars: {len(chunk.content)}):")
        print(f"Content: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")
        print(f"Range: {chunk.start_index}-{chunk.end_index}")
        print("-" * 30)
    
    # Show statistics
    stats = splitter.get_chunk_statistics(chunks)
    print("\nChunk Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return chunks

if __name__ == "__main__":
    # Process the Vislona chatbot dataset from the specified file path
    print("🚀 Starting Vislona Chatbot Dataset Processing...")
    print("=" * 60)
    
    # Main processing function
    chunks = process_vislona_chatbot_dataset()
    
    if chunks:
        # Save chunks to individual files
        save_chunks_to_files(chunks)
        
        # Create index file
        create_chunk_index(chunks)
        
        print("\n🎉 Processing Complete!")
        print(f"📁 Check the output directory for {len(chunks)} chunk files")
        print("📋 Chunk index created for easy reference")
    else:
        print("\n❌ Processing failed. Please check the file path and try again.")
    
    print("\n" + "=" * 60)

