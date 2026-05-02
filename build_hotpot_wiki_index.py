#!/usr/bin/env python3
"""Build a ChromaDB index from the official HotpotQA fullwiki abstracts corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from hotpot_retrieval import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingBackend,
    iter_wiki_paragraph_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma index for HotpotQA fullwiki abstracts.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Extracted directory containing many .bz2 files from the HotpotQA abstracts archive",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("data/wiki/chroma"),
        help="Directory where the persistent Chroma database will be stored",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path(".hf_cache"),
        help="Cache dir for embedding model downloads",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=512,
        help="How many paragraphs to buffer before writing to Chroma",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of paragraphs to index, useful for smoke tests",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the target collection before indexing",
    )
    return parser.parse_args()


def flush_batch(collection, embedding_backend: EmbeddingBackend, batch: list[dict[str, object]], batch_size: int) -> None:
    ids = [str(item["chunk_id"]) for item in batch]
    documents = [str(item["text"]) for item in batch]
    metadatas = [
        {
            "article_id": int(item["article_id"]) if item["article_id"] is not None else -1,
            "title": str(item["title"]),
            "paragraph_index": int(item["paragraph_index"]),
            "sentences_json": json.dumps(item["sentences"], ensure_ascii=False),
        }
        for item in batch
    ]
    embeddings = embedding_backend.encode(documents, batch_size=batch_size)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def main() -> None:
    args = parse_args()
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    import chromadb

    print(f"Loading source dir: {args.source_dir}", flush=True)
    args.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    if args.reset:
        try:
            client.delete_collection(args.collection_name)
        except Exception:  # noqa: BLE001
            pass
    collection = client.get_or_create_collection(name=args.collection_name, metadata={"hnsw:space": "cosine"})
    print(f"Loading embedding model: {args.embedding_model}", flush=True)
    embedding_backend = EmbeddingBackend(model_name=args.embedding_model, cache_dir=args.embedding_cache_dir)
    print("Embedding model ready. Start indexing.", flush=True)

    total_indexed = 0
    batch: list[dict[str, object]] = []
    progress = tqdm(desc="Indexing fullwiki paragraphs", unit="para", dynamic_ncols=True)
    for record in iter_wiki_paragraph_records(args.source_dir):
        batch.append(record)
        if args.limit is not None and total_indexed + len(batch) >= args.limit:
            keep = args.limit - total_indexed
            batch = batch[:keep]
        if len(batch) >= args.upsert_batch_size or (args.limit is not None and total_indexed + len(batch) >= args.limit):
            flush_batch(collection, embedding_backend, batch, batch_size=args.batch_size)
            total_indexed += len(batch)
            progress.update(len(batch))
            print(f"Indexed so far: {total_indexed}", flush=True)
            batch = []
        if args.limit is not None and total_indexed >= args.limit:
            break
    if batch:
        flush_batch(collection, embedding_backend, batch, batch_size=args.batch_size)
        total_indexed += len(batch)
        progress.update(len(batch))
        print(f"Indexed so far: {total_indexed}", flush=True)
    progress.close()

    metadata_path = args.chroma_dir / f"{args.collection_name}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_dir": str(args.source_dir),
                "collection_name": args.collection_name,
                "embedding_model": args.embedding_model,
                "total_indexed": total_indexed,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Indexed paragraphs: {total_indexed}")
    print(f"Chroma directory: {args.chroma_dir}")
    print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()
