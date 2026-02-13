# RAG System with OpenAI API and Embeddings

A complete implementation of a Retrieval-Augmented Generation (RAG) system using OpenAI's API and embeddings.

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### 2. Basic Usage

```python
from rag_system import SimpleRAG

# Initialize
rag = SimpleRAG(api_key="your-api-key")

# Add documents
documents = [
    "Your first document text here...",
    "Your second document text here..."
]
rag.add_documents(documents)

# Query
result = rag.query("What is...?")
print(result['answer'])
```

## 📚 How RAG Works

```
User Query → Embedding → Vector Search → Retrieve Top-K Docs → 
→ Build Context → Send to LLM → Generate Answer
```

### Key Components:

1. **Document Chunking**: Split large documents into manageable pieces
2. **Embeddings**: Convert text to numerical vectors (1536 dimensions)
3. **Vector Search**: Find semantically similar documents using cosine similarity
4. **Context Building**: Combine retrieved documents as context
5. **Generation**: Use GPT to generate answers based on context

## 🔧 Two Implementations

### Simple RAG (`rag_system.py`)
- Best for: Learning, small projects, prototyping
- Features: Basic chunking, numpy-based search
- No external dependencies except OpenAI

### Advanced RAG (`advanced_rag.py`)
- Best for: Production, large datasets
- Features: FAISS indexing, persistent storage, metadata tracking
- Requires: FAISS library

## 📖 Detailed Examples

### Example 1: Simple Q&A System

```python
from rag_system import SimpleRAG
import os

# Setup
rag = SimpleRAG(api_key=os.getenv("OPENAI_API_KEY"))

# Add knowledge base
docs = [
    """Python is a high-level programming language. 
    It's known for readability and versatility.""",
    
    """JavaScript is used for web development. 
    It runs in browsers and on servers via Node.js."""
]

rag.add_documents(docs)

# Ask questions
result = rag.query("What is Python known for?")
print(result['answer'])
# Output: "Python is known for its readability and versatility."

# Check sources
for source in result['sources']:
    print(f"Similarity: {source['similarity']:.3f}")
```

### Example 2: With Persistent Storage

```python
from advanced_rag import AdvancedRAG

rag = AdvancedRAG(api_key="your-key", storage_path="./knowledge_db")

# First run: Build knowledge base
documents = ["Document 1...", "Document 2..."]
metadata = [
    {"source": "Manual.pdf", "page": 1},
    {"source": "Manual.pdf", "page": 2}
]

rag.add_documents(documents, metadata)
rag.save()  # Save to disk

# Later runs: Load from disk
rag.load()
result = rag.query("Your question")
```

### Example 3: PDF Processing

```python
from advanced_rag import AdvancedRAG
import PyPDF2

def load_pdf(filepath):
    """Extract text from PDF"""
    with open(filepath, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Load PDF
pdf_text = load_pdf("document.pdf")

# Add to RAG
rag = AdvancedRAG(api_key="your-key")
rag.add_documents(
    [pdf_text],
    [{"source": "document.pdf", "type": "pdf"}]
)

# Query
result = rag.query("What does the document say about...?")
```

## ⚙️ Configuration Options

### Embedding Models

```python
# Cost-effective (default)
embedding_model = "text-embedding-3-small"  # 1536 dimensions, $0.02/1M tokens

# Higher quality
embedding_model = "text-embedding-3-large"  # 3072 dimensions, $0.13/1M tokens
```

### Generation Models

```python
# Fast and cheap (default)
model = "gpt-4o-mini"

# Higher quality
model = "gpt-4o"
model = "gpt-4-turbo"
```

### Chunking Parameters

```python
# Smaller chunks: Better precision, more API calls
chunk_size = 300
overlap = 30

# Larger chunks: More context, fewer API calls
chunk_size = 1000
overlap = 100
```

### Search Parameters

```python
# Retrieve more context (better recall)
result = rag.query("question", top_k=5)

# Retrieve less context (faster, cheaper)
result = rag.query("question", top_k=2)
```

## 💡 Best Practices

### 1. Document Preparation
- Clean text (remove headers, footers, page numbers)
- Split long documents into logical sections
- Include metadata (source, date, author)

### 2. Chunking Strategy
- Keep chunks between 300-800 characters
- Use overlap (50-100 chars) to preserve context
- Don't split mid-sentence

### 3. Query Optimization
- Use specific questions
- Include context in your query if needed
- Experiment with top_k values (2-5 usually optimal)

### 4. Cost Management
```python
# Estimate costs
num_documents = 100
avg_doc_length = 1000  # characters
tokens_per_doc = avg_doc_length / 4  # rough estimate

# Embedding cost (text-embedding-3-small)
embedding_cost = (num_documents * tokens_per_doc) / 1_000_000 * 0.02

# Query cost (gpt-4o-mini)
# ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
```

## 🐛 Troubleshooting

### "Rate limit exceeded"
```python
# Add delay between batches
import time
time.sleep(1)  # Wait 1 second between batches
```

### "Context too long"
```python
# Reduce chunk size or top_k
rag.query("question", top_k=2)  # Use fewer sources
```

### Poor search results
```python
# Try different embedding model
rag.embedding_model = "text-embedding-3-large"

# Increase top_k
result = rag.query("question", top_k=5)

# Improve document quality (remove noise, better chunking)
```

## 📊 Performance Tips

1. **Use FAISS for large datasets** (>10,000 chunks)
2. **Batch operations** when adding many documents
3. **Cache embeddings** to avoid regenerating
4. **Monitor token usage** to control costs
5. **Use async/await** for concurrent queries (advanced)

## 🔍 Advanced Features

### Hybrid Search (Keyword + Semantic)

```python
def hybrid_search(query, documents, top_k=5):
    # Get semantic results
    semantic_results = rag.search(query, top_k=top_k*2)
    
    # Get keyword results (simple implementation)
    keywords = query.lower().split()
    keyword_scores = []
    for doc in documents:
        score = sum(word in doc.lower() for word in keywords)
        keyword_scores.append(score)
    
    # Combine scores (normalize and weight)
    # ... implementation details
    
    return combined_results
```

### Re-ranking

```python
# Use a re-ranking model after initial retrieval
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get initial results
results = rag.search(query, top_k=10)

# Re-rank
pairs = [[query, r['text']] for r in results]
scores = reranker.predict(pairs)

# Sort by new scores
reranked = sorted(zip(results, scores), 
                  key=lambda x: x[1], 
                  reverse=True)[:3]
```

## 📝 API Key Setup

### Option 1: Environment Variable
```bash
export OPENAI_API_KEY='sk-...'
```

### Option 2: .env File
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-..." > .env

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

### Option 3: Direct in Code (not recommended)
```python
rag = SimpleRAG(api_key="sk-...")
```

## 🎯 Use Cases

- **Customer Support**: Answer questions from documentation
- **Research Assistant**: Query research papers and articles  
- **Internal Wiki**: Company knowledge base search
- **Educational Tools**: Study guides from textbooks
- **Code Documentation**: Search through code docs and examples

## 📚 Additional Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [RAG Best Practices](https://www.anthropic.com/index/retrieval-augmented-generation)

## 🤝 Contributing

Feel free to extend these implementations with:
- More sophisticated chunking strategies
- Support for more file formats
- Query expansion techniques
- Evaluation metrics
- Web interface
