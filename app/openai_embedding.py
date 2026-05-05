from __future__ import annotations

import sys
import time
from typing import Any

from openai import OpenAI

from app.settings import AppSettings


class OpenAIEmbeddingService:
    """OpenAI Embeddings API 호출을 담당하는 객체."""

    def __init__(self, settings: AppSettings) -> None:
        """OpenAI client를 초기화합니다.

        Args:
            settings: `OPENAI_API_KEY`, 선택적 `OPENAI_BASE_URL`, embedding 모델명, 차원 수,
                선택적 encoding format, 재시도 횟수를 담은 설정 객체입니다.
                `OPENAI_BASE_URL`이 비어 있으면 SDK 기본값을 씁니다.
        """
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )

    # 공개 함수: 외부에서는 이 함수만 호출해서 embedding을 생성합니다.
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """여러 문자열을 embedding 벡터로 변환합니다.

        Args:
            texts: embedding할 문자열 목록입니다. 비어 있지 않은 한국어/영어 텍스트를 넣습니다.
                이 프로젝트에서는 MongoDB `desc_v1` 값을 정규화한 문자열이 들어갑니다.

        Returns:
            list[list[float]]: `texts`와 같은 순서의 embedding 벡터 목록입니다.

        Raises:
            RuntimeError: OpenAI API 호출이 설정된 재시도 횟수만큼 실패할 때 발생합니다.
        """
        if not texts:
            return []

        request_body = self._build_embedding_request(texts)
        last_error: Exception | None = None
        for attempt in range(self.settings.openai_max_retries):
            try:
                response = self._client.embeddings.create(**request_body)
                return [item.embedding for item in response.data]
            except Exception as exc:
                last_error = exc
                sleep_seconds = min(2**attempt, 30)
                print(f"OpenAI request failed, retrying in {sleep_seconds}s: {exc}", file=sys.stderr)
                time.sleep(sleep_seconds)

        raise RuntimeError(f"OpenAI embedding failed after retries: {last_error}") from last_error

    # 내부 함수: OpenAI API 요청 payload 생성에만 사용합니다.
    def _build_embedding_request(self, texts: list[str]) -> dict[str, Any]:
        """OpenAI Embeddings API 요청 payload를 만듭니다.

        Args:
            texts: API의 `input`에 넣을 문자열 목록입니다.

        Returns:
            dict[str, Any]: `client.embeddings.create(**payload)`에 넣을 키워드 인자입니다.
        """
        request_body: dict[str, Any] = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        if self.settings.embedding_dimensions is not None:
            request_body["dimensions"] = self.settings.embedding_dimensions
        if self.settings.embedding_encoding_format:
            request_body["encoding_format"] = self.settings.embedding_encoding_format
        return request_body
