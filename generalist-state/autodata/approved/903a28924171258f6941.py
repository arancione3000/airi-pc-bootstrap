import argparse
import os

from .config import EXTRACTION_BUFFER_PATH, logger
from .github_client import github_client, verify_github_session
from .pipeline import extract_and_push


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scarica collezioni Normattiva, converte in markdown e pusha su GitHub.",
    )
    parser.add_argument(
        "root_path",
        nargs="?",
        default=os.getenv("ROOT_PATH"),
        help="Cartella di lavoro temporanea. Se omessa usa ROOT_PATH o chiede in console.",
    )
    args = parser.parse_args()
    root_path = (args.root_path or "").strip()
    if not root_path:
        root_path = (EXTRACTION_BUFFER_PATH or "").strip()
    if not root_path:
        parser.error("Specifica root_path o imposta ROOT_PATH / EXTRACTION_BUFFER_PATH.")

    logger.info("Root path: %s", root_path)
    gh = github_client()
    verify_github_session(gh)
    extract_and_push(root_path, gh)
    logger.info("Done")


if __name__ == "__main__":
    main()
