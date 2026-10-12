"""Run the mortgage underwriting workflow against OCR-extracted images."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from .mortgage_models import UnderwritingInput
from .mortgage_workflow import MortgageUnderwritingWorkflow

TASK_QUEUE = os.environ.get("MORTGAGE_TASK_QUEUE", "mortgage-underwriting")
TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")


def _load_dotenv() -> None:
    """Load .env values into the process environment (non-destructive)."""

    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'").strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _find_case_ids(image_dir: Path) -> list[str]:
    if not image_dir.exists():
        return []

    case_ids: set[str] = set()
    pattern = re.compile(r"^(?P<case_id>.+)_p\d+\.(png|jpg|jpeg)$", re.IGNORECASE)
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            case_ids.add(match.group("case_id"))
    return sorted(case_ids)


def _has_any_images(image_dir: Path) -> bool:
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        if any(image_dir.glob(pattern)):
            return True
    return False


async def _run_case(client: Client, case_id: str, image_dir: str) -> None:
    input_data = UnderwritingInput(case_id=case_id, image_dir=image_dir)

    result = await client.execute_workflow(
        MortgageUnderwritingWorkflow.run,
        input_data,
        id=f"mortgage-{case_id}",
        task_queue=TASK_QUEUE,
    )

    print("=" * 80)
    print(f"Case {case_id}")
    print(f"Workflow ID: mortgage-{case_id}")
    print(f"Final decision: {result.final_decision}")
    print(f"Risk score: {result.risk_score}")
    print(f"Human review required: {result.human_review_required}")
    if result.human_review_required:
        print("Review needed. Start the review UI and submit a decision.")
    print("Risk flags:")
    for flag in result.risk_flags:
        print(f"- {flag}")
    print("Bias flags:")
    for flag in result.bias_flags:
        print(f"- {flag}")
    print("Decision memo preview:")
    print(result.decision_memo[:600])
    print()


async def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Run OCR-first mortgage demo.")
    parser.add_argument(
        "--image-dir",
        default="datasets/images",
        help="Directory containing OCR input images (default: datasets/images).",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    case_ids = _find_case_ids(image_dir)
    if not case_ids:
        if not _has_any_images(image_dir):
            print(f"No case images found in {image_dir}.")
            return
        # Fall back to a single synthetic case ID to process all images in the directory.
        case_ids = ["DEMO-UNSCOPED"]
        print("No case-scoped filenames found. Processing all images as DEMO-UNSCOPED.")

    client = await Client.connect(
        TEMPORAL_ADDRESS,
        data_converter=pydantic_data_converter,
    )
    # Run all cases concurrently so a case waiting on human review doesn't block the others.
    results = await asyncio.gather(
        *(_run_case(client, case_id, str(image_dir)) for case_id in case_ids),
        return_exceptions=True,
    )
    for case_id, outcome in zip(case_ids, results, strict=True):
        if isinstance(outcome, Exception):
            print(f"Case {case_id} failed: {outcome}")


if __name__ == "__main__":
    asyncio.run(main())
