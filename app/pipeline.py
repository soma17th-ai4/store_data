from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterator
from typing import Any

from app.mongodb_client import KiprisMongoRepository
from app.openai_embedding import OpenAIEmbeddingService
from app.openai_text import OpenAITextDescriptionService
from app.settings import AppSettings


class KiprisPatentPipeline:
    """KIPRIS 특허 문서의 `desc_v1` 생성과 `embedded_v2` 저장을 조율합니다."""

    def __init__(
        self,
        settings: AppSettings,
        mongo_repo: KiprisMongoRepository,
        embedding_service: OpenAIEmbeddingService,
        desc_service: OpenAITextDescriptionService,
    ) -> None:
        """파이프라인에 필요한 객체를 주입합니다.

        Args:
            settings: batch size, 필드명, 모델명 같은 실행 설정입니다.
            mongo_repo: MongoDB 읽기/쓰기 기능을 제공하는 repository입니다.
            embedding_service: `desc_v1`을 embedding vector로 변환하는 service입니다.
            desc_service: 특허 원본 필드로 `desc_v1` 설명을 생성하는 service입니다.
        """
        self.settings = settings
        self.mongo_repo = mongo_repo
        self.embedding_service = embedding_service
        self.desc_service = desc_service

    def run_all(self, limit: int = 0, dry_run: bool = False, workers: int | None = None) -> None:
        """누락 `desc_v1` 생성 후 `embedded_v2` 생성까지 순서대로 실행합니다.

        Args:
            limit: 각 단계에서 처리할 최대 문서 수입니다. 0이면 각 단계 전체를 처리합니다.
            dry_run: True면 LLM/API 호출 결과를 출력만 하고 MongoDB를 수정하지 않습니다.
            workers: `desc_v1` 생성에 사용할 병렬 worker 수입니다. None이면 `.env`의 `DESC_WORKERS`를 씁니다.
        """
        print("step 1/2: generate missing desc_v1")
        self.generate_missing_desc(limit=limit, dry_run=dry_run, workers=workers)
        print("step 2/2: embed desc_v1 into embedded_v2")
        self.embed_mongo(limit=limit, force=False, dry_run=dry_run)

    def generate_missing_desc(self, limit: int = 0, dry_run: bool = False, workers: int | None = None) -> None:
        """`desc_v1`이 비어 있는 문서에 일반 고객용 설명을 생성해 저장합니다.

        Args:
            limit: 0이면 전체 누락 문서를 처리합니다. 양수면 해당 개수만 처리합니다.
            dry_run: True면 생성 결과를 출력만 하고 MongoDB에는 저장하지 않습니다.
            workers: 병렬 API 호출 개수입니다. None이면 `.env`의 `DESC_WORKERS`를 씁니다.
        """
        self.mongo_repo.ping()
        total = self.mongo_repo.count_missing_desc_targets()
        if limit:
            total = min(total, limit)
        worker_count = max(1, workers or self.settings.desc_workers)
        print(f"missing desc_v1 documents: {total}")
        print(f"desc_v1 workers: {worker_count}")

        processed = 0
        updated = 0
        for docs in self._iter_missing_desc_batches(limit=limit, batch_size=worker_count):
            generated_docs = self._generate_desc_batch(docs=docs, workers=worker_count)
            if dry_run:
                for item in generated_docs:
                    print(f"_id: {item['_id']}")
                    print(f"generated desc_v1: {item['desc_v1']}")
                print("MongoDB was not updated.")
            else:
                updated += self.mongo_repo.save_desc_v1_many(
                    generated_docs=generated_docs,
                    model=self.settings.desc_model,
                )

            processed += len(generated_docs)
            print(f"processed={processed} updated={updated} dry_run={dry_run}")

        print(f"done processed={processed} updated={updated} dry_run={dry_run}")

    def embed_mongo(self, limit: int = 0, force: bool = False, dry_run: bool = False) -> None:
        """MongoDB `desc_v1`을 embedding해서 `embedded_v2`로 저장합니다.

        Args:
            limit: 0이면 전체 대상 문서를 처리합니다. 양수면 해당 개수만 처리합니다.
            force: True면 기존 `embedded_v2`가 있어도 다시 생성해서 덮어씁니다.
            dry_run: True면 OpenAI embedding 생성까지만 하고 MongoDB에는 저장하지 않습니다.
        """
        self.mongo_repo.ping()
        total = self.mongo_repo.count_embedding_targets(force=force)
        if limit:
            total = min(total, limit)
        print(f"target documents: {total}")

        processed = 0
        updated = 0
        for docs in self._iter_embedding_batches(limit=limit, force=force):
            embeddings = self.embedding_service.embed_texts([doc["text"] for doc in docs])
            if dry_run:
                dim = len(embeddings[0]) if embeddings else 0
                print(f"dry-run embedded batch size={len(docs)} dim={dim}")
            else:
                updated += self.mongo_repo.save_embeddings(
                    docs=docs,
                    embeddings=embeddings,
                    overwrite_existing=force,
                )

            processed += len(docs)
            print(f"processed={processed} updated={updated} dry_run={dry_run}")

        print(f"done processed={processed} updated={updated} dry_run={dry_run}")

    def test_first_embedding(self, preview_size: int = 10) -> None:
        """첫 번째 MongoDB 문서의 `desc_v1`을 embedding해서 결과만 출력합니다.

        Args:
            preview_size: 출력할 embedding 앞부분 원소 개수입니다.
        """
        self.mongo_repo.ping()
        doc = self.mongo_repo.get_first_embedding_target(force=True)
        if doc is None:
            print("No document with source text found.")
            return

        embeddings = self.embedding_service.embed_texts([doc["text"]])
        vector = embeddings[0]
        print(f"_id: {doc['_id']}")
        print(f"text preview: {doc['text'][:200]}")
        print(f"embedding dimension: {len(vector)}")
        print(f"embedding preview: {vector[:preview_size]}")
        print("MongoDB was not updated.")

    def _iter_embedding_batches(self, limit: int, force: bool) -> Iterator[list[dict[str, Any]]]:
        """embedding 대상 문서를 batch 단위로 묶어 반환합니다."""
        batch: list[dict[str, Any]] = []
        for doc in self.mongo_repo.iter_embedding_targets(force=force, limit=limit):
            batch.append(doc)
            if len(batch) >= self.settings.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _iter_missing_desc_batches(self, limit: int, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        """`desc_v1` 생성 대상 문서를 worker 수에 맞춰 batch 단위로 묶어 반환합니다."""
        batch: list[dict[str, Any]] = []
        for doc in self.mongo_repo.iter_missing_desc_targets(limit=limit):
            batch.append(doc)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _generate_desc_batch(self, docs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
        """문서 batch의 `desc_v1`을 병렬로 생성합니다."""
        if workers <= 1 or len(docs) == 1:
            return [
                {
                    "_id": doc["_id"],
                    "desc_v1": self.desc_service.generate_desc_v1(doc),
                }
                for doc in docs
            ]

        generated: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_doc = {
                executor.submit(self.desc_service.generate_desc_v1, doc): doc
                for doc in docs
            }
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                generated.append(
                    {
                        "_id": doc["_id"],
                        "desc_v1": future.result(),
                    }
                )
        return generated
