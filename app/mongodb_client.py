from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from pymongo import MongoClient, UpdateOne

from app.settings import AppSettings


class KiprisMongoRepository:
    """`crawler_db.kipris_patents` 컬렉션 접근을 담당하는 객체.

    외부에서는 `count_embedding_targets`, `iter_embedding_targets`,
    `get_first_embedding_target`, `save_embeddings`, `count_missing_desc_targets`,
    `iter_missing_desc_targets`, `save_desc_v1_many`만 사용하면 됩니다.
    """

    def __init__(self, settings: AppSettings) -> None:
        """MongoDB repository를 초기화합니다.

        Args:
            settings: `.env`에서 읽은 MongoDB 접속 정보와 필드명을 담은 설정 객체입니다.
        """
        self.settings = settings
        client_options: dict[str, Any] = {
            "host": settings.mongo_host,
            "port": settings.mongo_port,
            "serverSelectionTimeoutMS": 5000,
        }
        if settings.mongo_username:
            client_options["username"] = settings.mongo_username
            client_options["password"] = settings.mongo_password
            client_options["authSource"] = settings.mongo_auth_source

        self._client = MongoClient(
            **client_options,
        )
        self._collection = self._client[settings.mongo_db][settings.mongo_collection]

    # 공개 함수: MongoDB 연결 상태 확인용입니다.
    def ping(self) -> None:
        """MongoDB 연결 가능 여부를 확인합니다.

        Raises:
            pymongo.errors.PyMongoError: 인증 실패, 포트 오류, 서버 미실행 등 연결 문제가 있을 때 발생합니다.
        """
        self._client.admin.command("ping")

    # 공개 함수: embedding할 문서 수를 확인할 때 사용합니다.
    def count_embedding_targets(self, force: bool = False) -> int:
        """embedding 처리 대상 문서 수를 반환합니다.

        Args:
            force: True면 기존 `embedded_v2` 존재 여부와 관계없이 `desc_v1`이 있는 모든 문서를 셉니다.
                False면 `embedded_v2`가 없거나 null/빈 배열인 문서만 셉니다.

        Returns:
            int: 처리 대상 문서 개수입니다.
        """
        return self._collection.count_documents(self._build_embedding_target_filter(force))

    # 공개 함수: embedding할 MongoDB 문서를 스트리밍합니다.
    def iter_embedding_targets(self, force: bool = False, limit: int = 0) -> Iterator[dict[str, Any]]:
        """embedding 대상 문서를 하나씩 반환합니다.

        Args:
            force: True면 기존 embedding을 덮어쓸 대상으로 포함합니다.
            limit: 0이면 전체 대상 문서를 반환하고, 양수면 그 개수만큼만 반환합니다.

        Yields:
            dict[str, Any]: `_id`와 `text` 키를 가진 문서입니다.
                `_id`는 MongoDB ObjectId이고, `text`는 정규화된 `desc_v1` 문자열입니다.
        """
        projection = {"_id": 1, self.settings.source_text_field: 1}
        with self._client.start_session() as session:
            cursor = self._collection.find(
                self._build_embedding_target_filter(force),
                projection,
                no_cursor_timeout=True,
                session=session,
            ).batch_size(self.settings.mongo_page_size)
            if limit:
                cursor = cursor.limit(limit)

            try:
                for doc in cursor:
                    text = self._normalize_text(doc.get(self.settings.source_text_field))
                    if text:
                        yield {"_id": doc["_id"], "text": text}
            finally:
                cursor.close()

    # 공개 함수: 테스트용으로 첫 번째 embedding 대상 문서 하나만 가져옵니다.
    def get_first_embedding_target(self, force: bool = True) -> dict[str, Any] | None:
        """첫 번째 embedding 대상 문서 하나를 반환합니다.

        Args:
            force: True면 `embedded_v2` 존재 여부와 관계없이 `desc_v1`이 있는 첫 문서를 가져옵니다.
                False면 아직 `embedded_v2`가 없는 첫 문서를 가져옵니다.

        Returns:
            dict[str, Any] | None: `_id`와 `text`를 가진 문서입니다. 대상 문서가 없으면 None입니다.
        """
        projection = {"_id": 1, self.settings.source_text_field: 1}
        doc = self._collection.find_one(self._build_embedding_target_filter(force), projection)
        if doc is None:
            return None

        text = self._normalize_text(doc.get(self.settings.source_text_field))
        if not text:
            return None
        return {"_id": doc["_id"], "text": text}

    # 공개 함수: 생성된 embedding을 MongoDB에 저장합니다.
    def save_embeddings(
        self,
        docs: list[dict[str, Any]],
        embeddings: list[list[float]],
        overwrite_existing: bool = False,
    ) -> int:
        """embedding 결과를 MongoDB `embedded_v2` 필드에 저장합니다.

        Args:
            docs: `iter_embedding_targets()`에서 받은 문서 목록입니다. 각 항목에는 `_id`가 있어야 합니다.
            embeddings: OpenAI API가 반환한 embedding 목록입니다. `docs`와 같은 순서, 같은 길이어야 합니다.
            overwrite_existing: True면 기존 `embedded_v2`를 덮어씁니다.
                False면 저장 직전에도 `embedded_v2`가 비어 있는 문서만 업데이트합니다.

        Returns:
            int: MongoDB에서 수정되거나 upsert된 문서 수입니다.

        Raises:
            RuntimeError: `docs`와 `embeddings`의 개수가 다를 때 발생합니다.
        """
        if len(docs) != len(embeddings):
            raise RuntimeError(f"Embedding count mismatch: docs={len(docs)} embeddings={len(embeddings)}")

        operations = []
        for doc, embedding in zip(docs, embeddings, strict=True):
            query: dict[str, Any] = {"_id": doc["_id"]}
            if not overwrite_existing:
                query["$or"] = [
                    {self.settings.target_embedding_field: {"$exists": False}},
                    {self.settings.target_embedding_field: None},
                    {self.settings.target_embedding_field: []},
                ]

            operations.append(
                UpdateOne(
                    query,
                    {
                        "$set": {
                            self.settings.target_embedding_field: embedding,
                            "embedded_v2_model": self.settings.embedding_model,
                            "embedded_v2_dimensions": len(embedding),
                            "embedded_v2_source_field": self.settings.source_text_field,
                            "embedded_v2_updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                    },
                )
            )
        result = self._collection.bulk_write(operations, ordered=False)
        return result.modified_count

    # 공개 함수: desc_v1을 생성해야 하는 문서 수를 확인할 때 사용합니다.
    def count_missing_desc_targets(self) -> int:
        """`desc_v1`이 없거나 비어 있는 문서 수를 반환합니다.

        Returns:
            int: `desc_v1` 생성 대상 문서 개수입니다.
        """
        return self._collection.count_documents(self._build_missing_desc_filter())

    # 공개 함수: desc_v1 생성 대상 MongoDB 문서를 스트리밍합니다.
    def iter_missing_desc_targets(self, limit: int = 0) -> Iterator[dict[str, Any]]:
        """`desc_v1`이 없는 문서를 하나씩 반환합니다.

        Args:
            limit: 0이면 전체 대상 문서를 반환하고, 양수면 그 개수만큼만 반환합니다.

        Yields:
            dict[str, Any]: desc 생성에 필요한 주요 특허 필드를 포함한 MongoDB 문서입니다.
        """
        with self._client.start_session() as session:
            cursor = self._collection.find(
                self._build_missing_desc_filter(),
                self._desc_source_projection(),
                no_cursor_timeout=True,
                session=session,
            ).batch_size(self.settings.mongo_page_size)
            if limit:
                cursor = cursor.limit(limit)

            try:
                yield from cursor
            finally:
                cursor.close()

    # 공개 함수: 생성된 desc_v1 여러 개를 MongoDB에 bulk 저장합니다.
    def save_desc_v1_many(self, generated_docs: list[dict[str, Any]], model: str) -> int:
        """여러 문서의 `desc_v1` 생성 결과를 MongoDB에 한 번에 저장합니다.

        Args:
            generated_docs: `_id`와 `desc_v1` 키를 가진 생성 결과 목록입니다.
            model: `desc_v1` 생성에 사용한 모델명입니다. 예: `solar-mini-250422`.

        Returns:
            int: 수정된 문서 수입니다.
        """
        if not generated_docs:
            return 0

        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        operations = [
            UpdateOne(
                {
                    "_id": item["_id"],
                    "$or": [
                        {self.settings.source_text_field: {"$exists": False}},
                        {self.settings.source_text_field: None},
                        {self.settings.source_text_field: ""},
                    ],
                },
                {
                    "$set": {
                        self.settings.source_text_field: item["desc_v1"],
                        "desc_v1_model": model,
                        "desc_v1_generated_at": generated_at,
                    }
                },
            )
            for item in generated_docs
        ]
        result = self._collection.bulk_write(operations, ordered=False)
        return result.modified_count

    # 내부 함수: MongoDB query 생성에만 사용합니다.
    def _build_embedding_target_filter(self, force: bool) -> dict[str, Any]:
        """embedding 대상 MongoDB query를 만듭니다.

        Args:
            force: 기존 `embedded_v2`가 있는 문서까지 포함할지 여부입니다.

        Returns:
            dict[str, Any]: PyMongo `find`와 `count_documents`에 넣을 query입니다.
        """
        query: dict[str, Any] = {
            self.settings.source_text_field: {"$exists": True, "$nin": [None, ""]},
        }
        if not force:
            query["$or"] = [
                {self.settings.target_embedding_field: {"$exists": False}},
                {self.settings.target_embedding_field: None},
                {self.settings.target_embedding_field: []},
            ]
        return query

    # 내부 함수: desc_v1이 없는 문서를 찾는 MongoDB query를 만듭니다.
    def _build_missing_desc_filter(self) -> dict[str, Any]:
        """`desc_v1` 생성 대상 MongoDB query를 만듭니다.

        Returns:
            dict[str, Any]: `desc_v1`이 없거나 null/빈 문자열인 문서를 찾는 query입니다.
        """
        return {
            "$or": [
                {self.settings.source_text_field: {"$exists": False}},
                {self.settings.source_text_field: None},
                {self.settings.source_text_field: ""},
            ]
        }

    # 내부 함수: desc_v1 생성에 필요한 필드만 MongoDB에서 가져오도록 projection을 만듭니다.
    def _desc_source_projection(self) -> dict[str, int]:
        """desc 생성용 MongoDB projection을 반환합니다.

        Returns:
            dict[str, int]: `_id`와 특허 주요 텍스트/메타 필드를 포함하는 projection입니다.
        """
        return {
            "_id": 1,
            "applicantName": 1,
            "applicationDate": 1,
            "applicationNumber": 1,
            "astrtCont": 1,
            "indexNo": 1,
            "inventionTitle": 1,
            "ipcNumber": 1,
            "openDate": 1,
            "openNumber": 1,
            "publicationDate": 1,
            "publicationNumber": 1,
            "registerDate": 1,
            "registerNumber": 1,
            "registerStatus": 1,
            self.settings.source_text_field: 1,
        }

    # 내부 함수: MongoDB 원본 텍스트 정리에만 사용합니다.
    def _normalize_text(self, value: Any) -> str:
        """MongoDB 원본 값을 OpenAI embedding 입력 문자열로 정규화합니다.

        Args:
            value: `desc_v1` 값입니다. 보통 문자열이지만 None이나 다른 타입도 방어적으로 처리합니다.

        Returns:
            str: 연속 공백과 줄바꿈을 하나의 공백으로 줄인 문자열입니다.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return " ".join(value.split())
        return " ".join(str(value).split())
