#!/usr/bin/env python3
from __future__ import annotations

import argparse

from app.mongodb_client import KiprisMongoRepository
from app.openai_embedding import OpenAIEmbeddingService
from app.openai_text import OpenAITextDescriptionService
from app.pipeline import KiprisPatentPipeline
from app.settings import EnvSettingsLoader


class CliApp:
    """터미널 인자를 해석하고 KIPRIS 특허 migration 작업을 실행합니다."""

    def run(self) -> None:
        """CLI 인자를 읽고 요청된 command를 실행합니다."""
        args = self._parse_args()
        settings = EnvSettingsLoader().load()
        pipeline = KiprisPatentPipeline(
            settings=settings,
            mongo_repo=KiprisMongoRepository(settings),
            embedding_service=OpenAIEmbeddingService(settings),
            desc_service=OpenAITextDescriptionService(settings),
        )

        if args.command == "run-all":
            pipeline.run_all(limit=args.limit, dry_run=args.dry_run)
        elif args.command == "generate-missing-desc":
            pipeline.generate_missing_desc(limit=args.limit, dry_run=args.dry_run)
        elif args.command == "embed-mongo":
            pipeline.embed_mongo(limit=args.limit, force=args.force, dry_run=args.dry_run)
        elif args.command == "test-first-embedding":
            pipeline.test_first_embedding(preview_size=args.preview_size)
        else:
            raise RuntimeError(f"Unknown command: {args.command}")

    def _parse_args(self) -> argparse.Namespace:
        """CLI command와 옵션을 정의합니다."""
        parser = argparse.ArgumentParser(
            description="Generate KIPRIS patent desc_v1 and embedded_v2 fields in MongoDB."
        )
        sub = parser.add_subparsers(dest="command", required=True)

        run_all = sub.add_parser(
            "run-all",
            help="Generate missing desc_v1 values first, then embed available desc_v1 values.",
        )
        run_all.add_argument("--limit", type=int, default=0, help="Process only N documents per step.")
        run_all.add_argument("--dry-run", action="store_true", help="Call APIs but do not update MongoDB.")

        desc = sub.add_parser(
            "generate-missing-desc",
            help="Generate desc_v1 for MongoDB documents where desc_v1 is missing.",
        )
        desc.add_argument("--limit", type=int, default=0, help="Process only N missing desc_v1 documents.")
        desc.add_argument("--dry-run", action="store_true", help="Generate desc_v1 but do not update MongoDB.")

        embed = sub.add_parser("embed-mongo", help="Create embeddings from MongoDB desc_v1 and save embedded_v2.")
        embed.add_argument("--limit", type=int, default=0, help="Process only N documents.")
        embed.add_argument("--force", action="store_true", help="Overwrite existing embedded_v2 values.")
        embed.add_argument("--dry-run", action="store_true", help="Call embedding API but do not update MongoDB.")

        test = sub.add_parser(
            "test-first-embedding",
            help="Embed only the first MongoDB document and print the vector without saving.",
        )
        test.add_argument("--preview-size", type=int, default=10, help="Print only N vector values.")

        return parser.parse_args()


if __name__ == "__main__":
    CliApp().run()
