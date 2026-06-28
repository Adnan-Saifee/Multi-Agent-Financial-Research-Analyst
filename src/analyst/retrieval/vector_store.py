import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from analyst.config import settings
from analyst.ingestion.chunking import Chunk
from analyst.log import print_verbose

class VectorStore:
    def __init__(self, collection_name: str = "narrative_filings", upsert_batch_size: int = 100, chroma_dir: Optional[Path] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.chroma_dir = Path(chroma_dir or settings.CHROMA_DIR)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upsert_batch_size = upsert_batch_size
        self.collection_name = collection_name

        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=str(self.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

        print_verbose(self, f"VectorStore initialised — collection '{self.collection_name}' at {self.chroma_dir}")


    # Embed and store in vector store (in batches)
    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        
        if not chunks:
            print_verbose(self, error=True,message="upsert_chunks called with empty list — nothing to do.")
            return

        total = len(chunks)
        print_verbose(self, f"Upserting {total} chunks in batches of {self.upsert_batch_size}...")

        for batch_start in range(0, total, self.upsert_batch_size):
            batch = chunks[batch_start: batch_start + self.upsert_batch_size]
            batch_num = batch_start // self.upsert_batch_size + 1

            self.store.add_texts(
                texts     = [c.text for c in batch],
                metadatas = [c.to_chroma_metadata() for c in batch],
                ids       = [c.chunk_id for c in batch],
            )

            print_verbose(self, f"Batch {batch_num}: upserted chunks {batch_start + 1} - {batch_start + len(batch)} of {total}.")

        print_verbose(self, f"Upsert complete. Collection now contains {self.store._collection.count()} chunks.")


    def query(
        self,
        question: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Returns a plain list of dicts rather than LangChain Document objects so callers don't need to 
        import LangChain types just to read results.
        Will as_retriever() when wiring into a LangChain chain or LangGraph node.
        """
        search_kwargs: dict = {"k": k}
        if filters:
            search_kwargs["filter"] = filters

        docs_and_scores = self.store.similarity_search_with_relevance_scores(
            query=question,
            **search_kwargs,
        )

        return [
            {
                "text":     doc.page_content,
                "metadata": doc.metadata,
                "score":    score,
            }
            for doc, score in docs_and_scores
        ]

    def as_retriever(
        self,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> VectorStoreRetriever:
        
        search_kwargs: dict = {"k": k}
        if filters:
            search_kwargs["filter"] = filters

        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def count(self) -> int:
        return self.store._collection.count()

    def delete_collection(self) -> None:
        self.store._client.delete_collection(self.collection_name)
        # Reinitialise so the object stays usable after the wipe
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=str(self.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        print_verbose(self, f"Collection '{self.collection_name}' wiped and recreated.")