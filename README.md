# KIPRIS Patent MongoDB Enrichment

이 프로젝트는 `crawler_db.kipris_patents` MongoDB 컬렉션의 KIPRIS 특허 문서를 검색용으로 보강하는 배치 스크립트입니다.

현재 하는 일은 MongoDB 문서 안에서만 끝납니다. Supabase 적재는 아직 포함하지 않았습니다.

## What This Does

문서마다 아래 두 필드를 채우는 것이 목적입니다.

- `desc_v1`: 특허 원문 필드를 일반 고객이 이해하고 검색하기 쉬운 한 문장 설명으로 요약한 텍스트
- `embedded_v2`: `desc_v1`을 embedding한 4096차원 벡터

처리 흐름은 두 단계입니다.

1. `desc_v1`이 없거나 비어 있는 문서를 찾고, `inventionTitle`, `astrtCont`, `applicantName`, `ipcNumber`, 등록 상태/날짜/번호 같은 기존 필드로 `desc_v1`을 생성합니다.
2. `desc_v1`은 있지만 `embedded_v2`가 없는 문서를 찾아 embedding을 생성해 `embedded_v2`에 저장합니다.

## Models

현재 Upstage API를 OpenAI 호환 SDK로 호출합니다.

- 설명 생성: `solar-mini-250422`
- 문서 embedding: `solar-embedding-1-large-passage`
- 검색 query embedding용 참고 모델: `solar-embedding-1-large-query`

`desc_v1`은 검색될 문서 본문이므로 passage embedding 모델을 사용합니다. 사용자가 입력하는 검색어를 나중에 embedding할 때는 query 모델을 쓰면 됩니다.

## Safety Rules

기본 실행은 기존 데이터를 최대한 덮어쓰지 않게 되어 있습니다.

- `generate-missing-desc`는 `desc_v1`이 없거나 `null`이거나 빈 문자열인 문서만 수정합니다.
- `embed-mongo`는 `desc_v1`이 있고, `embedded_v2`가 없거나 `null`이거나 빈 배열인 문서만 수정합니다.
- `embed-mongo --force`를 쓰면 기존 `embedded_v2`도 다시 생성해 덮어씁니다.
- `--dry-run`을 쓰면 API 호출 결과를 출력만 하고 MongoDB는 수정하지 않습니다.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

환경 파일을 만듭니다.

```bash
cp .env.example .env
```

`.env`에 실제 값을 넣습니다.

```dotenv
OPENAI_API_KEY=your-upstage-api-key
OPENAI_BASE_URL=https://api.upstage.ai/v1
OPENAI_EMBEDDING_MODEL=solar-embedding-1-large-passage
OPENAI_QUERY_EMBEDDING_MODEL=solar-embedding-1-large-query
OPENAI_DESC_MODEL=solar-mini-250422

MONGO_HOST=localhost
MONGO_PORT=20060
MONGO_USERNAME=root
MONGO_PASSWORD=your-password
MONGO_AUTH_SOURCE=admin
MONGO_DB=crawler_db
MONGO_COLLECTION=kipris_patents
```

`.env`는 `.gitignore`에 들어가 있으므로 커밋하지 않습니다.

## Commands

전체 보강 작업을 순서대로 실행합니다.

```bash
python main.py run-all
```

먼저 1개만 API 호출까지 해보고 MongoDB는 수정하지 않습니다.

```bash
python main.py run-all --limit 1 --dry-run
```

`desc_v1` 생성은 기본적으로 병렬 worker를 사용합니다. 기본값은 `.env`의 `DESC_WORKERS=5`이고, 실행 시 바꿀 수 있습니다.

```bash
python main.py run-all --workers 8
```

`desc_v1` 누락 문서만 생성합니다.

```bash
python main.py generate-missing-desc
```

`desc_v1` 생성만 worker 수를 바꿔 실행할 수도 있습니다.

```bash
python main.py generate-missing-desc --workers 8
```

`desc_v1` 누락 문서 1개만 테스트합니다.

```bash
python main.py generate-missing-desc --limit 1 --dry-run
```

`embedded_v2` 누락 문서만 embedding합니다.

```bash
python main.py embed-mongo
```

embedding API와 벡터 차원만 확인합니다. MongoDB는 수정하지 않습니다.

```bash
python main.py test-first-embedding
```

기존 `embedded_v2`를 다시 만들 때만 아래 옵션을 씁니다.

```bash
python main.py embed-mongo --force
```

## MongoDB Fields Written

`generate-missing-desc`가 쓰는 필드:

- `desc_v1`
- `desc_v1_model`
- `desc_v1_generated_at`

`embed-mongo`가 쓰는 필드:

- `embedded_v2`
- `embedded_v2_model`
- `embedded_v2_dimensions`
- `embedded_v2_source_field`
- `embedded_v2_updated_at`

## Project Structure

- `main.py`: CLI entrypoint
- `app/settings.py`: `.env` 로딩과 설정 객체
- `app/pipeline.py`: desc 생성과 embedding 저장 흐름 조율
- `app/mongodb_client.py`: MongoDB query/update 전담 repository
- `app/openai_text.py`: `desc_v1` 생성 API 호출
- `app/openai_embedding.py`: embedding API 호출

## Recommended Team Workflow

처음 실행할 때는 항상 dry-run으로 API 응답과 대상 문서 수를 확인합니다.

```bash
python main.py run-all --limit 1 --dry-run
```

출력이 정상이고 모델/DB 설정이 맞으면 전체 실행합니다.

```bash
python main.py run-all
```

이미 `desc_v1` 생성이 끝난 뒤 embedding만 다시 이어가려면 아래만 실행합니다.

```bash
python main.py embed-mongo
```
