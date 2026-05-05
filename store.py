#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from supabase import Client, create_client


PATENT_COLUMNS = [
    "_id",
    "applicantName",
    "applicationDate",
    "applicationNumber",
    "astrtCont",
    "bigDrawing",
    "drawing",
    "indexNo",
    "inventionTitle",
    "ipcNumber",
    "openDate",
    "openNumber",
    "publicationDate",
    "publicationNumber",
    "registerDate",
    "registerNumber",
    "registerStatus",
    "desc_v1",
    "embedded_v1",
    "is_migrated",
    "embedded_v2",
    "embedded_v2_dimensions",
    "embedded_v2_model",
    "embedded_v2_source_field",
    "embedded_v2_updated_at",
]

VECTOR_FIELDS = {
    "embedded_v1": 768,
    "embedded_v2": 4096,
}


class StoreSettings:
    """MongoDB에서 Supabase로 적재할 때 필요한 환경 설정."""

    def __init__(self, env_file: str = ".env") -> None:
        """`.env`를 읽어 설정 객체를 초기화합니다.

        Args:
            env_file: 읽을 dotenv 파일 경로입니다. 기본값은 `.env`입니다.
        """
        load_dotenv(env_file)
        self.mongo_host = os.getenv("MONGO_HOST", "localhost").strip()
        self.mongo_port = self._get_int("MONGO_PORT", 20060)
        self.mongo_username = os.getenv("MONGO_USERNAME", "").strip()
        self.mongo_password = os.getenv("MONGO_PASSWORD", "")
        self.mongo_auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin").strip()
        self.mongo_db = os.getenv("MONGO_DB", "crawler_db").strip()
        self.mongo_collection = os.getenv("MONGO_COLLECTION", "kipris_patents").strip()
        self.mongo_page_size = self._get_int("MONGO_PAGE_SIZE", 500)

        self.supabase_url = self._get_required("SUPABASE_URL")
        self.supabase_key = self._get_required("SUPABASE_KEY")
        self.supabase_table = os.getenv("SUPABASE_TABLE", "patents").strip()
        self.supabase_batch_size = self._get_int("SUPABASE_BATCH_SIZE", 100)

    def _get_required(self, name: str) -> str:
        """필수 환경변수를 읽습니다.

        Args:
            name: `.env`에 반드시 있어야 하는 환경변수 이름입니다.

        Returns:
            str: 공백을 제거한 환경변수 값입니다.
        """
        value = os.getenv(name, "").strip()
        if not value:
            raise RuntimeError(f"Missing required env var: {name}")
        return value

    def _get_int(self, name: str, default: int) -> int:
        """정수 환경변수를 읽습니다."""
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else default


class MongoPatentReader:
    """MongoDB `kipris_patents` 문서를 Supabase row 형태로 읽는 객체."""

    def __init__(self, settings: StoreSettings) -> None:
        """MongoDB reader를 초기화합니다.

        Args:
            settings: MongoDB 접속 정보와 batch size를 담은 설정 객체입니다.
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

        self._client = MongoClient(**client_options)
        self._collection = self._client[settings.mongo_db][settings.mongo_collection]

    def ping(self) -> None:
        """MongoDB 연결 가능 여부를 확인합니다."""
        self._client.admin.command("ping")

    def count_documents(self) -> int:
        """Supabase 적재 대상 문서 수를 반환합니다.

        `embedded_v2`가 없으면 검색용 Supabase row로 쓰지 않으므로 적재 대상에서 제외합니다.
        """
        return self._collection.count_documents(self._load_filter())

    def iter_rows(self, limit: int = 0) -> Iterator[dict[str, Any]]:
        """MongoDB 문서를 Supabase `patents` row 형태로 변환해 반환합니다.

        Args:
            limit: 0이면 전체 문서를 읽고, 양수면 해당 개수만 읽습니다.

        Yields:
            dict[str, Any]: Supabase upsert에 바로 넘길 row입니다.
        """
        with self._client.start_session() as session:
            cursor = self._collection.find(
                self._load_filter(),
                self._projection(),
                no_cursor_timeout=True,
                session=session,
            ).batch_size(self.settings.mongo_page_size)
            if limit:
                cursor = cursor.limit(limit)

            try:
                for doc in cursor:
                    yield self._to_supabase_row(doc)
            finally:
                cursor.close()

    def _projection(self) -> dict[str, int]:
        """Supabase 테이블 컬럼에 대응되는 필드만 MongoDB에서 읽습니다."""
        return {column: 1 for column in PATENT_COLUMNS}

    def _load_filter(self) -> dict[str, Any]:
        """Supabase 적재 대상 MongoDB query를 반환합니다.

        Returns:
            dict[str, Any]: `embedded_v2`가 존재하고 null/빈 배열이 아닌 문서만 찾는 query입니다.
        """
        return {
            "embedded_v2": {
                "$exists": True,
                "$nin": [None, []],
            }
        }

    def _to_supabase_row(self, doc: dict[str, Any]) -> dict[str, Any]:
        """MongoDB 문서를 Supabase row로 변환합니다."""
        row: dict[str, Any] = {}
        for column in PATENT_COLUMNS:
            if column not in doc:
                continue
            value = doc[column]
            if column == "_id":
                row[column] = str(value)
            elif column in VECTOR_FIELDS:
                row[column] = self._to_pgvector(value, expected_dimensions=VECTOR_FIELDS[column])
            else:
                row[column] = self._to_json_value(value)
        return row

    def _to_pgvector(self, value: Any, expected_dimensions: int) -> str | None:
        """Python list 벡터를 pgvector text literal로 변환합니다.

        Args:
            value: MongoDB에 저장된 embedding 배열입니다.
            expected_dimensions: Supabase vector 컬럼의 차원입니다.

        Returns:
            str | None: pgvector가 받을 수 있는 `[1.0,2.0]` 문자열입니다. 값이 없으면 None입니다.

        Raises:
            ValueError: 벡터 길이가 Supabase 컬럼 차원과 다르면 발생합니다.
        """
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(f"Vector field must be list, got {type(value).__name__}")
        if len(value) != expected_dimensions:
            raise ValueError(f"Vector dimension mismatch: expected={expected_dimensions}, actual={len(value)}")
        return "[" + ",".join(str(float(item)) for item in value) + "]"

    def _to_json_value(self, value: Any) -> Any:
        """MongoDB/BSON 값을 Supabase JSON payload에 들어갈 값으로 변환합니다."""
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return value


class SupabasePatentWriter:
    """Supabase `patents` 테이블에 row를 batch upsert하는 객체."""

    def __init__(self, settings: StoreSettings) -> None:
        """Supabase writer를 초기화합니다.

        Args:
            settings: Supabase URL, API key, table 이름을 담은 설정 객체입니다.
        """
        self.settings = settings
        self._client: Client = create_client(settings.supabase_url, settings.supabase_key)

    def upsert_rows(self, rows: list[dict[str, Any]]) -> int:
        """Supabase에 row batch를 upsert합니다.

        Args:
            rows: `MongoPatentReader.iter_rows()`가 만든 row 목록입니다.

        Returns:
            int: upsert 요청한 row 수입니다.
        """
        if not rows:
            return 0
        self._client.table(self.settings.supabase_table).upsert(rows, on_conflict="_id").execute()
        return len(rows)


class MongoToSupabaseStore:
    """MongoDB 전체 특허 문서를 Supabase로 적재하는 실행 객체."""

    def __init__(self, reader: MongoPatentReader, writer: SupabasePatentWriter, settings: StoreSettings) -> None:
        """적재 파이프라인을 초기화합니다."""
        self.reader = reader
        self.writer = writer
        self.settings = settings

    def run(self, limit: int = 0, dry_run: bool = False) -> None:
        """MongoDB 문서를 읽어 Supabase로 batch upsert합니다.

        Args:
            limit: 0이면 전체 문서를 적재합니다. 양수면 해당 개수만 적재합니다.
            dry_run: True면 MongoDB row 변환까지만 하고 Supabase에는 쓰지 않습니다.
        """
        self.reader.ping()
        total = self.reader.count_documents()
        if limit:
            total = min(total, limit)
        print(f"source MongoDB documents: {total}")
        print(f"supabase table: {self.settings.supabase_table}")
        print(f"batch size: {self.settings.supabase_batch_size}")

        processed = 0
        upserted = 0
        batch: list[dict[str, Any]] = []
        for row in self.reader.iter_rows(limit=limit):
            batch.append(row)
            if len(batch) >= self.settings.supabase_batch_size:
                upserted += self._flush(batch=batch, dry_run=dry_run)
                processed += len(batch)
                print(f"processed={processed} upserted={upserted} dry_run={dry_run}")
                batch = []

        if batch:
            upserted += self._flush(batch=batch, dry_run=dry_run)
            processed += len(batch)
            print(f"processed={processed} upserted={upserted} dry_run={dry_run}")

        print(f"done processed={processed} upserted={upserted} dry_run={dry_run}")

    def _flush(self, batch: list[dict[str, Any]], dry_run: bool) -> int:
        """batch 하나를 Supabase에 쓰거나 dry-run 출력으로 대체합니다."""
        if dry_run:
            first_id = batch[0].get("_id") if batch else None
            print(f"dry-run batch rows={len(batch)} first_id={first_id}")
            return 0
        return self.writer.upsert_rows(batch)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Load MongoDB KIPRIS patent documents into Supabase patents table.")
    parser.add_argument("--limit", type=int, default=0, help="Load only N MongoDB documents.")
    parser.add_argument("--dry-run", action="store_true", help="Convert rows but do not write Supabase.")
    return parser.parse_args()


def main() -> None:
    """store.py entrypoint."""
    args = parse_args()
    settings = StoreSettings()
    reader = MongoPatentReader(settings)
    writer = SupabasePatentWriter(settings)
    MongoToSupabaseStore(reader=reader, writer=writer, settings=settings).run(
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
