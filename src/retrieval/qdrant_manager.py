"""
Qdrant Vector Database Manager.
Handles collection creation, data ingestion, and configuration.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)
from qdrant_client.http import models
import numpy as np
from tqdm import tqdm
import uuid

from ..config import Settings
from ..models import QwenEmbedding


class QdrantManager:
    """
    Manager for Qdrant vector database operations.

    Handles:
    - Collection creation with dual collections (Q&A vs Plain Text)
    - Data ingestion with automatic embedding
    - Collection configuration
    """

    def __init__(
        self,
        settings: Settings,
        embedding_model: QwenEmbedding,
    ):
        """
        Initialize Qdrant manager.

        Args:
            settings: Application settings
            embedding_model: Embedding model instance
        """
        self.settings = settings
        self.embedding_model = embedding_model

        # Initialize Qdrant client
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )

        self.qa_collection = settings.qa_collection_name
        self.text_collection = settings.text_collection_name

    def create_collections(self, recreate: bool = False):
        """
        Create both Q&A and Plain Text collections.

        Args:
            recreate: If True, delete existing collections first
        """
        embedding_dim = self.embedding_model.embedding_dimension

        collections = [
            (self.qa_collection, "Q&A pairs collection"),
            (self.text_collection, "Plain text documents collection"),
        ]

        for collection_name, description in collections:
            # Check if collection exists
            exists = self.client.collection_exists(collection_name)

            if exists and recreate:
                self.client.delete_collection(collection_name)
                exists = False

            if not exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"✓ Created collection: {collection_name} ({description})")
            else:
                print(f"✓ Collection already exists: {collection_name}")

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Add documents to a collection with automatic embedding.

        Args:
            collection_name: Target collection name
            documents: List of document strings
            metadata: Optional metadata for each document
            batch_size: Batch size for processing

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in documents]

        if len(metadata) != len(documents):
            raise ValueError("Metadata list must match documents list length")

        total_added = 0

        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc=f"Adding to {collection_name}"):
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(batch_docs)

            # Create points
            points = []
            for doc, emb, meta in zip(batch_docs, embeddings, batch_metadata):
                point_id = str(uuid.uuid4())
                payload = {
                    "content": doc,
                    **meta,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb.tolist(),
                        payload=payload,
                    )
                )

            # Upload to Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            total_added += len(points)

        print(f"✓ Added {total_added} documents to {collection_name}")
        return total_added

    def add_qa_pairs(
        self,
        qa_pairs: List[Dict[str, str]],
        batch_size: int = 100,
    ) -> int:
        """
        Add Q&A pairs to the Q&A collection.

        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys
            batch_size: Batch size for processing

        Returns:
            Number of pairs added
        """
        # Create documents from Q&A pairs (embed the question)
        documents = []
        metadata = []

        for pair in qa_pairs:
            question = pair.get("question", "")
            answer = pair.get("answer", "")

            if not question or not answer:
                continue

            documents.append(question)
            metadata.append({
                "question": question,
                "answer": answer,
                "type": "qa_pair",
            })

        return self.add_documents(
            collection_name=self.qa_collection,
            documents=documents,
            metadata=metadata,
            batch_size=batch_size,
        )

    def add_plain_text(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Add plain text documents to the text collection.

        Args:
            texts: List of text strings
            metadata: Optional metadata for each text
            batch_size: Batch size for processing

        Returns:
            Number of texts added
        """
        # Add type to metadata
        if metadata is None:
            metadata = [{"type": "plain_text"} for _ in texts]
        else:
            for meta in metadata:
                meta["type"] = "plain_text"

        return self.add_documents(
            collection_name=self.text_collection,
            documents=texts,
            metadata=metadata,
            batch_size=batch_size,
        )

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
    ) -> List[ScoredPoint]:
        """
        Search in a collection using dense vector search.

        Args:
            collection_name: Collection to search
            query: Query string
            limit: Number of results

        Returns:
            List of scored points
        """
        # Embed query
        query_vector = self.embedding_model.embed_query(query)

        # Search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
        )

        return results

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Collection name

        Returns:
            Collection info dict
        """
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        self.client.delete_collection(collection_name)
        print(f"✓ Deleted collection: {collection_name}")

    def clear_all_collections(self):
        """Delete all managed collections."""
        for collection_name in [self.qa_collection, self.text_collection]:
            if self.client.collection_exists(collection_name):
                self.delete_collection(collection_name)
