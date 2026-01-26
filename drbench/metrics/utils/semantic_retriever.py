import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional

from drbench.config import get_run_config
from drbench.embeddings import get_embeddings


class SemanticRetriever:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        cfg = get_run_config()
        self.embedding_model = embedding_model or cfg.get_embedding_model() or "text-embedding-3-small"
        self.embedding_provider = embedding_provider or cfg.get_embedding_provider()
        self.chunks = []
        self.chunk_embeddings = None
        
    def chunk_text(self, text: str, source_title: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'source': source_title,
                'start_idx': i
            })
            
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts using configured provider."""
        embeddings = get_embeddings(
            texts,
            model=self.embedding_model,
            provider=self.embedding_provider,
        )
        return np.array(embeddings)
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the RAG system."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc['content'], doc['title'])
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        
        # Get embeddings for all chunks
        if self.chunks:
            chunk_texts = [chunk['text'] for chunk in self.chunks]
            # Process in batches to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_embeddings = self.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            self.chunk_embeddings = np.array(all_embeddings)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve most relevant chunks for a query."""
        if not self.chunks or self.chunk_embeddings is None:
            return []
        
        # Get embedding for the query
        query_embedding = self.get_embeddings([query])
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding[0].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings).flatten()
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum similarity threshold for embeddings
                chunk = self.chunks[idx].copy()
                chunk['similarity'] = similarities[idx]
                relevant_chunks.append(chunk)
        
        return relevant_chunks
