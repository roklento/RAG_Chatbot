"""
Example 1: Setup Qdrant database and ingest sample data.

This script demonstrates:
1. Creating Qdrant collections
2. Adding Q&A pairs
3. Adding plain text documents
"""

import sys
sys.path.append('..')

from src.config import get_settings
from src.models import QwenEmbedding
from src.retrieval import QdrantManager


def main():
    # Load settings
    settings = get_settings()
    print("Settings loaded")
    print(f"  Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"  Q&A Collection: {settings.qa_collection_name}")
    print(f"  Text Collection: {settings.text_collection_name}\n")

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = QwenEmbedding(
        model_path=settings.embedding_model_path,
        device=settings.device,
        max_length=settings.embedding_max_length,
    )
    print(f"✓ Embedding dimension: {embedding_model.embedding_dimension}\n")

    # Initialize Qdrant manager
    print("Initializing Qdrant manager...")
    manager = QdrantManager(settings=settings, embedding_model=embedding_model)
    print("✓ Connected to Qdrant\n")

    # Create collections
    print("Creating collections...")
    manager.create_collections(recreate=True)
    print()

    # Sample Q&A pairs
    qa_pairs = [
        {
            "question": "Makine öğrenmesi nedir?",
            "answer": "Makine öğrenmesi, sistemlerin açıkça programlanmadan deneyimlerden öğrenmesini ve gelişmesini sağlayan yapay zekanın bir alt kümesidir."
        },
        {
            "question": "Denetimli ve denetimsiz öğrenme arasındaki fark nedir?",
            "answer": "Denetimli öğrenme, modelleri eğitmek için etiketlenmiş veri kullanırken, denetimsiz öğrenme etiketlenmemiş verilerde desenler bulur."
        },
        {
            "question": "Yapay sinir ağı nedir?",
            "answer": "Yapay sinir ağı, biyolojik sinir ağlarından esinlenen, katmanlar halinde düzenlenmiş birbirine bağlı düğümlerden (nöronlar) oluşan bir hesaplama sistemidir."
        },
        {
            "question": "Derin öğrenme nedir?",
            "answer": "Derin öğrenme, karmaşık desenleri öğrenmek için çok katmanlı sinir ağları (derin sinir ağları) kullanan makine öğrenmesinin bir alt kümesidir."
        },
        {
            "question": "Doğal dil işleme nedir?",
            "answer": "Doğal Dil İşleme (NLP), bilgisayarların insan dilini anlamasına, yorumlamasına ve üretmesine yardımcı olan yapay zekanın bir dalıdır."
        },
    ]

    # Add Q&A pairs
    print("Adding Q&A pairs...")
    count = manager.add_qa_pairs(qa_pairs)
    print(f"✓ Added {count} Q&A pairs\n")

    # Sample plain text documents
    plain_texts = [
        "Transformer'lar, 'Attention is All You Need' makalesinde tanıtılan bir sinir ağı mimarisi türüdür. Sıralı verileri işlemek için öz-dikkat mekanizmaları kullanırlar ve BERT ve GPT gibi modern NLP modellerinin temeli haline gelmişlerdir.",

        "Dikkat mekanizması, modellerin tahmin yaparken girdinin ilgili kısımlarına odaklanmasını sağlar. Girdi dizisinin farklı bölümlerine ne kadar odaklanılacağını belirleyen dikkat ağırlıklarını hesaplar.",

        "BERT (Bidirectional Encoder Representations from Transformers), maskelenmiş dil modelleme ve sonraki cümle tahmin görevleri üzerinde eğitilerek çift yönlü bağlamı öğrenen önceden eğitilmiş bir dil modelidir.",

        "GPT (Generative Pre-trained Transformer), önceki token'lar verildiğinde bir sonraki token'ı tahmin ederek metin üreten otoregresif bir dil modelidir. Büyük miktarda metin verisi üzerinde eğitilir.",

        "İnce ayar (fine-tuning), önceden eğitilmiş bir modeli alıp göreve özgü verilerle belirli bir görev üzerinde eğitme sürecidir. Bu, ön eğitim sırasında öğrenilen bilginin alt görevler için kullanılmasını sağlar.",

        "Vektör gömmeleri (embeddings), semantik anlamı yakalayan metinlerin sayısal temsilleridir. Benzer metinler, vektör uzayında benzer gömmelere sahiptir ve bu da semantik arama ve benzerlik karşılaştırmalarını mümkün kılar.",

        "RAG (Retrieval-Augmented Generation), geri getirme sistemlerini üretken modellerle birleştirir. Bir bilgi tabanından ilgili belgeleri getirir ve bunları daha doğru ve bilgilendirilmiş yanıtlar üretmek için kullanır.",
    ]

    # Add plain texts
    print("Adding plain text documents...")
    count = manager.add_plain_text(plain_texts)
    print(f"✓ Added {count} plain text documents\n")

    # Show collection info
    print("Collection Information:")
    for collection_name in [settings.qa_collection_name, settings.text_collection_name]:
        info = manager.get_collection_info(collection_name)
        print(f"\n{collection_name}:")
        print(f"  Points: {info['points_count']}")
        print(f"  Vectors: {info['vectors_count']}")
        print(f"  Status: {info['status']}")

    print("\n✓ Database setup complete!")


if __name__ == "__main__":
    main()
