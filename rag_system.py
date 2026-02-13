"""
RAG (Retrieval-Augmented Generation) System using OpenAI API
This system demonstrates how to build a complete RAG pipeline with:
- Document chunking
- Embedding generation
- Vector storage
- Semantic search
- Context-aware response generation
"""

import os
import numpy as np
from openai import OpenAI
from typing import List, Dict, Tuple
import json

class SimpleRAG:
    def __init__(self, api_key: str):
        """
        Initialize the RAG system with OpenAI API key
        
        Args:
            api_key: Your OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.documents = []
        self.embeddings = []
        self.embedding_model = "text-embedding-3-small"  # Cost-effective option
        # Alternative: "text-embedding-3-large" for higher quality
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
            
        return chunks
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base and generate embeddings
        
        Args:
            documents: List of text documents to add
        """
        print(f"Adding {len(documents)} documents to the knowledge base...")
        
        # Chunk documents if they're too long
        all_chunks = []
        for doc in documents:
            if len(doc) > 500:
                chunks = self.chunk_text(doc)
                all_chunks.extend(chunks)
            else:
                all_chunks.append(doc)
        
        self.documents = all_chunks
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        self.embeddings = self._generate_embeddings(all_chunks)
        print("Knowledge base ready!")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using OpenAI API
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        # Process in batches to handle rate limits
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
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most relevant documents using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        # Generate embedding for the query
        query_embedding = self._generate_embeddings([query])[0]
        
        # Calculate similarities with all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def query(self, question: str, top_k: int = 3, model: str = "gpt-4o-mini") -> Dict:
        """
        Query the RAG system with a question
        
        Args:
            question: User's question
            top_k: Number of relevant documents to retrieve
            model: OpenAI model to use for generation
            
        Returns:
            Dictionary with answer and source documents
        """
        # Retrieve relevant documents
        relevant_docs = self.search(question, top_k)
        
        # Build context from retrieved documents
        context = "\n\n".join([f"Source {i+1}:\n{doc}" 
                               for i, (doc, score) in enumerate(relevant_docs)])
        
        # Create prompt with context
        prompt = f"""Answer the question based on the context below. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate response using OpenAI
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": [{"text": doc, "similarity": float(score)} 
                       for doc, score in relevant_docs]
        }


# Example usage
def main():
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Initialize RAG system
    rag = SimpleRAG(api_key)
    
    # Example documents to add to knowledge base
    documents = [
        """
        Python is a high-level, interpreted programming language known for its 
        simplicity and readability. It was created by Guido van Rossum and first 
        released in 1991. Python supports multiple programming paradigms including 
        procedural, object-oriented, and functional programming.
        """,
        """
        Machine learning is a subset of artificial intelligence that enables systems 
        to learn and improve from experience without being explicitly programmed. 
        It focuses on developing computer programs that can access data and use it 
        to learn for themselves.
        """,
        """
        RAG (Retrieval-Augmented Generation) is a technique that combines information 
        retrieval with text generation. It retrieves relevant documents from a knowledge 
        base and uses them as context for generating more accurate and informative responses. 
        This approach helps reduce hallucinations in language models.
        """,
        """
        Vector embeddings are numerical representations of text that capture semantic 
        meaning. They allow us to measure similarity between pieces of text by calculating 
        distances in vector space. OpenAI's embedding models can convert text into 
        high-dimensional vectors suitable for semantic search.
        """
    ]
    
    # Add documents to the knowledge base
    rag.add_documents(documents)
    
    # Example queries
    questions = [
        "What is RAG and why is it useful?",
        "Who created Python?",
        "How do vector embeddings work?"
    ]
    
    print("\n" + "="*80)
    print("QUESTION & ANSWER EXAMPLES")
    print("="*80)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)
        
        result = rag.query(question)
        
        print(f"Answer: {result['answer']}\n")
        print("Sources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. (Similarity: {source['similarity']:.3f}) {source['text'][:100]}...")
        print()


if __name__ == "__main__":
    main()
