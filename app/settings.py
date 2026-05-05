from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppSettings:
    """프로젝트 전체 설정값을 담는 읽기 전용 객체.

    각 값은 `.env`에서 읽어오며, 실행 중에는 변경하지 않는다는 전제로 사용합니다.
    """

    openai_api_key: str
    openai_base_url: str
    embedding_model: str
    embedding_dimensions: int | None
    embedding_encoding_format: str
    desc_model: str
    mongo_host: str
    mongo_port: int
    mongo_username: str
    mongo_password: str
    mongo_auth_source: str
    mongo_db: str
    mongo_collection: str
    source_text_field: str
    target_embedding_field: str
    batch_size: int
    desc_workers: int
    mongo_page_size: int
    openai_max_retries: int


class EnvSettingsLoader:
    """`.env` 파일과 환경변수를 읽어서 `AppSettings` 객체를 만드는 클래스."""

    def __init__(self, env_file: str = ".env") -> None:
        """환경 설정 로더를 초기화합니다.

        Args:
            env_file: 읽어올 dotenv 파일 경로입니다. 기본값은 프로젝트 루트의 `.env`입니다.
        """
        self.env_file = env_file

    # 공개 함수: `.env`를 읽어서 AppSettings 객체를 제공합니다.
    def load(self) -> AppSettings:
        """환경변수를 읽어 `AppSettings`로 반환합니다.

        Returns:
            AppSettings: MongoDB, OpenAI, 배치 실행에 필요한 모든 설정값입니다.

        Raises:
            RuntimeError: 필수값인 `OPENAI_API_KEY`가 비어 있을 때 발생합니다.
            ValueError: 정수로 읽어야 하는 환경변수가 숫자가 아닐 때 발생합니다.
        """
        load_dotenv(self.env_file)
        dimensions_raw = os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "").strip()
        return AppSettings(
            openai_api_key=self._get_required("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "solar-embedding-1-large-passage").strip(),
            embedding_dimensions=int(dimensions_raw) if dimensions_raw else None,
            embedding_encoding_format=os.getenv("OPENAI_EMBEDDING_ENCODING_FORMAT", "").strip(),
            desc_model=os.getenv("OPENAI_DESC_MODEL", "solar-mini-250422").strip(),
            mongo_host=os.getenv("MONGO_HOST", "localhost").strip(),
            mongo_port=self._get_int("MONGO_PORT", 20060),
            mongo_username=os.getenv("MONGO_USERNAME", "").strip(),
            mongo_password=os.getenv("MONGO_PASSWORD", ""),
            mongo_auth_source=os.getenv("MONGO_AUTH_SOURCE", "admin").strip(),
            mongo_db=os.getenv("MONGO_DB", "crawler_db").strip(),
            mongo_collection=os.getenv("MONGO_COLLECTION", "kipris_patents").strip(),
            source_text_field=os.getenv("SOURCE_TEXT_FIELD", "desc_v1").strip(),
            target_embedding_field=os.getenv("TARGET_EMBEDDING_FIELD", "embedded_v2").strip(),
            batch_size=self._get_int("BATCH_SIZE", 64),
            desc_workers=self._get_int("DESC_WORKERS", 5),
            mongo_page_size=self._get_int("MONGO_PAGE_SIZE", 500),
            openai_max_retries=self._get_int("OPENAI_MAX_RETRIES", 5),
        )

    # 내부 함수: 필수 환경변수 검증에만 사용합니다.
    def _get_required(self, name: str) -> str:
        """필수 환경변수를 읽습니다.

        Args:
            name: `.env` 또는 OS 환경변수에 존재해야 하는 변수명입니다.

        Returns:
            str: 앞뒤 공백을 제거한 환경변수 값입니다.
        """
        value = os.getenv(name, "").strip()
        if not value:
            raise RuntimeError(f"Missing required env var: {name}")
        return value

    # 내부 함수: 정수 환경변수 변환에만 사용합니다.
    def _get_int(self, name: str, default: int) -> int:
        """정수 환경변수를 읽습니다.

        Args:
            name: 정수로 변환할 환경변수 이름입니다.
            default: 환경변수가 비어 있을 때 사용할 기본값입니다.

        Returns:
            int: 환경변수 값 또는 기본값입니다.
        """
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else default
