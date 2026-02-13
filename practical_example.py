"""
Practical RAG Example: Document Q&A System
This example shows a complete workflow for building a RAG system that can:
- Load documents from files
- Answer questions with citations
- Maintain conversation context
"""

import os
from openai import OpenAI
from typing import List, Dict
import numpy as np


class DocumentQA:
    """Simple document Q&A system using RAG"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.documents = []
        self.embeddings = None
        self.conversation_history = []
        
    def load_text_file(self, filepath: str) -> str:
        """Load a text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_directory(self, directory: str, extension: str = '.txt'):
        """Load all text files from a directory"""
        docs = []
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                filepath = os.path.join(directory, filename)
                content = self.load_text_file(filepath)
                docs.append({
                    'content': content,
                    'source': filename,
                    'filepath': filepath
                })
        return docs
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents for RAG
        Each document should be a dict with 'content' and optional metadata
        """
        print(f"Indexing {len(documents)} documents...")
        
        # Store documents
        self.documents = documents
        
        # Create chunks with metadata
        chunks = []
        chunk_metadata = []
        
        for doc in documents:
            content = doc['content']
            # Simple chunking by paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                if len(para) > 50:  # Skip very short paragraphs
                    chunks.append(para)
                    chunk_metadata.append({
                        'source': doc.get('source', 'unknown'),
                        'filepath': doc.get('filepath', '')
                    })
        
        print(f"Created {len(chunks)} chunks from documents")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(embeddings)
        self.chunks = chunks
        self.chunk_metadata = chunk_metadata
        
        print("Indexing complete!")
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant document chunks"""
        # Generate query embedding
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top_k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'metadata': self.chunk_metadata[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def ask(
        self, 
        question: str, 
        top_k: int = 3,
        use_conversation_history: bool = False
    ) -> Dict:
        """
        Ask a question and get an answer with sources
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve
            use_conversation_history: Whether to include previous Q&A in context
        """
        # Search for relevant documents
        relevant_chunks = self.search_documents(question, top_k)
        
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk['metadata']['source']
            context_parts.append(f"[Source: {source}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on provided documents. 
Always base your answers on the given context. If the context doesn't contain enough information, 
say so clearly. When possible, mention which source supports your answer."""
            }
        ]
        
        # Add conversation history if requested
        if use_conversation_history and self.conversation_history:
            for qa in self.conversation_history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": qa['question']})
                messages.append({"role": "assistant", "content": qa['answer']})
        
        # Add current question with context
        user_message = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}"""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate answer
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Store in conversation history
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'sources': relevant_chunks
        })
        
        return {
            'answer': answer,
            'sources': relevant_chunks,
            'confidence': np.mean([c['score'] for c in relevant_chunks])
        }
    
    def print_answer(self, result: Dict):
        """Pretty print the answer and sources"""
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        
        print("\n" + "="*80)
        print("SOURCES:")
        print("="*80)
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['metadata']['source']} (relevance: {source['score']:.2%})")
            print(f"   {source['text'][:200]}...")
        
        print(f"\nConfidence: {result['confidence']:.2%}")
        print("="*80 + "\n")


def demo_with_sample_documents():
    """Demo with sample documents"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create sample documents
    sample_docs = [
        {
            'content': """
# Python Programming Guide

Python is a high-level, interpreted programming language created by Guido van Rossum 
and first released in 1991. It emphasizes code readability with its use of significant 
indentation.

Python supports multiple programming paradigms, including structured, object-oriented, 
and functional programming. It is often described as a "batteries included" language 
due to its comprehensive standard library.

Common use cases for Python include:
- Web development (Django, Flask)
- Data science and machine learning (NumPy, Pandas, scikit-learn)
- Automation and scripting
- Scientific computing

Python 3 is the current version, with Python 2 reaching end of life in 2020.
            """,
            'source': 'python_guide.txt'
        },
        {
            'content': """
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on building 
systems that can learn from and make decisions based on data.

## Types of Machine Learning

1. Supervised Learning: The algorithm learns from labeled training data. Examples 
   include classification and regression tasks.

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Common 
   techniques include clustering and dimensionality reduction.

3. Reinforcement Learning: The algorithm learns through trial and error, receiving 
   rewards or penalties for actions.

## Common Algorithms

- Linear Regression: Predicts continuous values
- Decision Trees: Makes decisions based on feature values
- Neural Networks: Mimics human brain structure for complex pattern recognition
- K-Means Clustering: Groups similar data points

Machine learning has applications in computer vision, natural language processing, 
recommendation systems, and more.
            """,
            'source': 'ml_fundamentals.txt'
        },
        {
            'content': """
# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) is a technique in natural language processing 
that enhances the capabilities of large language models by incorporating external 
knowledge retrieval.

## How RAG Works

1. Query Processing: The user's question is converted into an embedding vector
2. Document Retrieval: Relevant documents are found using similarity search
3. Context Building: Retrieved documents are formatted as context
4. Answer Generation: An LLM generates an answer using the context

## Benefits of RAG

- Reduces hallucinations by grounding responses in real documents
- Allows models to access up-to-date information
- More cost-effective than fine-tuning for domain-specific knowledge
- Provides citations and sources for answers

## Key Components

- Embedding Model: Converts text to numerical vectors (e.g., OpenAI's text-embedding models)
- Vector Database: Stores and searches embeddings efficiently (e.g., FAISS, Pinecone)
- Language Model: Generates natural language answers (e.g., GPT-4, Claude)

RAG is particularly useful for question-answering systems, chatbots, and knowledge 
management applications.
            """,
            'source': 'rag_intro.txt'
        }
    ]
    
    # Initialize Q&A system
    qa = DocumentQA(api_key)
    qa.index_documents(sample_docs)
    
    # Demo questions
    questions = [
        "Who created Python and when?",
        "What are the three types of machine learning?",
        "What is RAG and what are its benefits?",
        "How is Python used in data science?"
    ]
    
    print("\n" + "🤖 Document Q&A System Demo".center(80, "="))
    print("\n")
    
    for question in questions:
        print(f"❓ Question: {question}")
        result = qa.ask(question, top_k=2)
        qa.print_answer(result)
        input("Press Enter for next question...")


if __name__ == "__main__":
    demo_with_sample_documents()
