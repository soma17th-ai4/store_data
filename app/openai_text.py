from __future__ import annotations

import sys
import time
from typing import Any

from openai import OpenAI

from app.settings import AppSettings


class OpenAITextDescriptionService:
    """특허 원본 필드로 일반 고객용 `desc_v1` 설명을 생성하는 객체."""

    def __init__(self, settings: AppSettings) -> None:
        """텍스트 생성용 OpenAI 호환 client를 초기화합니다.

        Args:
            settings: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_DESC_MODEL`,
                재시도 횟수를 담은 설정 객체입니다.
        """
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )

    # 공개 함수: MongoDB 문서 하나를 받아 desc_v1 문장을 생성합니다.
    def generate_desc_v1(self, patent_doc: dict[str, Any]) -> str:
        """특허 문서 필드로 `desc_v1` 값을 생성합니다.

        Args:
            patent_doc: MongoDB 원본 특허 문서입니다. `inventionTitle`, `astrtCont`,
                `applicantName`, `ipcNumber`, `registerStatus` 같은 필드를 포함하면 품질이 좋아집니다.

        Returns:
            str: 일반 고객이 이해하기 쉬운 한국어 한 문장 설명입니다.

        Raises:
            RuntimeError: 모델 호출이 설정된 재시도 횟수만큼 실패하거나 빈 응답을 반환할 때 발생합니다.
        """
        messages = self._build_messages(patent_doc)
        last_error: Exception | None = None
        for attempt in range(self.settings.openai_max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.settings.desc_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=160,
                )
                content = response.choices[0].message.content
                desc = self._clean_desc(content)
                if desc:
                    return desc
                raise RuntimeError("empty desc_v1 response")
            except Exception as exc:
                last_error = exc
                sleep_seconds = min(2**attempt, 30)
                print(f"desc_v1 generation failed, retrying in {sleep_seconds}s: {exc}", file=sys.stderr)
                time.sleep(sleep_seconds)

        raise RuntimeError(f"desc_v1 generation failed after retries: {last_error}") from last_error

    # 내부 함수: 모델에 전달할 system/user 메시지를 만듭니다.
    def _build_messages(self, patent_doc: dict[str, Any]) -> list[dict[str, str]]:
        """특허 문서를 LLM 프롬프트 메시지로 변환합니다.

        Args:
            patent_doc: MongoDB 원본 특허 문서입니다.

        Returns:
            list[dict[str, str]]: Chat Completions API의 `messages` 값입니다.
        """
        field_lines = self._format_patent_fields(patent_doc)
        return [
            {
                "role": "system",
                "content": (
                    "너는 특허 내용을 일반 고객이 검색하고 이해하기 쉬운 한국어 설명으로 바꾸는 전문가다. "
                    "전문가용 청구항 문체를 피하고, 실제 고객이 얻는 효과와 활용 가치를 중심으로 쓴다."
                ),
            },
            {
                "role": "user",
                "content": (
                    "아래 특허 정보를 바탕으로 desc_v1 값을 만들어라.\n"
                    "규칙:\n"
                    "- 한국어 한 문장만 출력한다.\n"
                    "- 80자에서 160자 사이로 쓴다.\n"
                    "- 일반 고객/client가 검색할 법한 말로 쓴다.\n"
                    "- 특허번호, 출원번호, IPC 코드, 날짜, 회사명은 꼭 필요할 때만 쓴다.\n"
                    "- '본 발명', '개시된', '상기', '청구항' 같은 특허 문체는 쓰지 않는다.\n"
                    "- 따옴표, 번호, 마크다운, 설명문 없이 desc_v1 문장만 출력한다.\n\n"
                    f"특허 정보:\n{field_lines}"
                ),
            },
        ]

    # 내부 함수: desc 생성에 필요한 주요 필드만 사람이 읽기 쉬운 형태로 정리합니다.
    def _format_patent_fields(self, patent_doc: dict[str, Any]) -> str:
        """MongoDB 문서에서 desc 생성에 쓸 필드를 문자열로 만듭니다.

        Args:
            patent_doc: MongoDB 원본 특허 문서입니다.

        Returns:
            str: 프롬프트에 넣을 필드 목록 문자열입니다.
        """
        field_names = [
            "inventionTitle",
            "astrtCont",
            "applicantName",
            "ipcNumber",
            "registerStatus",
            "applicationDate",
            "openDate",
            "registerDate",
            "applicationNumber",
            "openNumber",
            "registerNumber",
        ]
        lines: list[str] = []
        for name in field_names:
            value = patent_doc.get(name)
            if value is None or value == "":
                continue
            lines.append(f"- {name}: {self._compact(value)}")
        return "\n".join(lines)

    # 내부 함수: 모델 출력값에서 불필요한 감싸기 문자를 제거합니다.
    def _clean_desc(self, content: str | None) -> str:
        """모델 응답을 MongoDB에 저장할 desc_v1 문자열로 정리합니다.

        Args:
            content: 모델이 반환한 원본 문자열입니다.

        Returns:
            str: 앞뒤 공백, 따옴표, bullet 문자를 제거한 문자열입니다.
        """
        if content is None:
            return ""
        desc = " ".join(content.split()).strip()
        return desc.strip("\"'`-• ")

    # 내부 함수: 긴 필드값을 프롬프트에 넣기 좋게 줄입니다.
    def _compact(self, value: Any, max_length: int = 900) -> str:
        """필드값을 한 줄 문자열로 압축합니다.

        Args:
            value: MongoDB 필드값입니다.
            max_length: 너무 긴 값을 자를 최대 문자 수입니다.

        Returns:
            str: 연속 공백을 줄이고 길이를 제한한 문자열입니다.
        """
        text = " ".join(str(value).split())
        if len(text) <= max_length:
            return text
        return text[:max_length].rstrip() + "..."
