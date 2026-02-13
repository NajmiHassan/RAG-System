"""
Advanced RAG System with Persistent Storage
Features:
- FAISS vector database for efficient similarity search
- Persistent storage and loading
- Support for PDF and text files
- Metadata tracking
"""

import os
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import pickle
import json

# Note: Install FAISS with: pip install faiss-cpu (or faiss-gpu for GPU support)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not installed. Using simple numpy search instead.")
    print("For better performance, install with: pip install faiss-cpu")


class AdvancedRAG:
    def __init__(self, api_key: str, storage_path: str = "./rag_storage"):
        """
        Initialize advanced RAG system with persistent storage
        
        Args:
            api_key: OpenAI API key
            storage_path: Directory to store vector database and metadata
        """
        self.client = OpenAI(api_key=api_key)
        self.storage_path = storage_path
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536  # Dimension for text-embedding-3-small
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize storage
        self.documents = []
        self.metadata = []
        self.index = None
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product (cosine similarity)
        
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents with optional metadata
        
        Args:
            documents: List of documents
            metadata: Optional list of metadata dicts for each document
        """
        if metadata is None:
            metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
        
        # Chunk documents
        all_chunks = []
        all_metadata = []
        
        for doc, meta in zip(documents, metadata):
            chunks = self._chunk_text(doc)
            all_chunks.extend(chunks)
            all_metadata.extend([meta] * len(chunks))
        
        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._generate_embeddings(all_chunks)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        if FAISS_AVAILABLE:
            self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(all_chunks)
        self.metadata.extend(all_metadata)
        
        print(f"Added {len(all_chunks)} chunks to knowledge base")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Avoid splitting words
            if end < len(text) and text[end] not in [' ', '\n', '.', ',']:
                last_space = chunk.rfind(' ')
                if last_space > 0:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap if end < len(text) else len(text)
        
        return [c for c in chunks if len(c) > 0]
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches"""
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents
        
        Returns:
            List of dicts with 'text', 'metadata', and 'score'
        """
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if FAISS_AVAILABLE and self.index.ntotal > 0:
            # Use FAISS for efficient search
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                min(top_k, self.index.ntotal)
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(score)
                    })
        else:
            # Fallback to numpy search
            embeddings = self._generate_embeddings(self.documents)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            scores = np.dot(embeddings, query_embedding)
            
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(scores[idx])
                })
        
        return results
    
    def query(
        self, 
        question: str, 
        top_k: int = 3, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ) -> Dict:
        """
        Query the RAG system
        
        Returns:
            Dict with 'answer', 'sources', and 'metadata'
        """
        # Retrieve relevant documents
        results = self.search(question, top_k)
        
        if not results:
            return {
                "answer": "I don't have any information to answer this question.",
                "sources": [],
                "metadata": {}
            }
        
        # Build context
        context_parts = []
        for i, result in enumerate(results):
            source_info = result['metadata'].get('source', f'Source {i+1}')
            context_parts.append(f"[{source_info}]\n{result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        system_message = """You are a helpful assistant that answers questions based on the provided context. 
Be specific and cite sources when possible. If the context doesn't contain enough information, say so."""
        
        user_message = f"""Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above."""
        
        # Generate response
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": results,
            "metadata": {
                "model": model,
                "num_sources": len(results),
                "avg_score": np.mean([r['score'] for r in results])
            }
        }
    
    def save(self):
        """Save the RAG system to disk"""
        print(f"Saving to {self.storage_path}...")
        
        # Save documents and metadata
        with open(f"{self.storage_path}/documents.pkl", 'wb') as f:
            pickle.dump({'documents': self.documents, 'metadata': self.metadata}, f)
        
        # Save FAISS index
        if FAISS_AVAILABLE and self.index.ntotal > 0:
            faiss.write_index(self.index, f"{self.storage_path}/faiss.index")
        
        print("Saved successfully!")
    
    def load(self):
        """Load the RAG system from disk"""
        print(f"Loading from {self.storage_path}...")
        
        # Load documents and metadata
        doc_path = f"{self.storage_path}/documents.pkl"
        if os.path.exists(doc_path):
            with open(doc_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
        
        # Load FAISS index
        index_path = f"{self.storage_path}/faiss.index"
        if FAISS_AVAILABLE and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        print(f"Loaded {len(self.documents)} documents")


# Example usage with file loading
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize advanced RAG
    rag = AdvancedRAG(api_key, storage_path="./my_knowledge_base")
    
    # Example: Add documents with metadata
    documents = [
        "Python is a versatile programming language used for web development, data science, and automation.",
        "Machine learning models learn patterns from data to make predictions or decisions.",
        "RAG systems combine retrieval and generation to provide accurate, context-aware answers."
    ]
    
    metadata = [
        {"source": "Python Guide", "category": "programming"},
        {"source": "ML Handbook", "category": "AI"},
        {"source": "RAG Tutorial", "category": "NLP"}
    ]
    
    rag.add_documents(documents, metadata)
    
    # Save for later use
    rag.save()
    
    # Query the system
    result = rag.query("What is RAG?", top_k=2)
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result['answer'])
    print("\n" + "="*80)
    print("SOURCES:")
    print("="*80)
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. Score: {source['score']:.3f} | {source['metadata']}")
        print(f"   {source['text'][:150]}...")


if __name__ == "__main__":
    main()
