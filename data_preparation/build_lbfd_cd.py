#!/usr/bin/env python3
"""Generate category-specific split files from the current LBFD-CD dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SPLITS = ("train", "val", "test")
COMPONENTS = ("t1", "t2", "label")
LANDSLIDE_SOURCES = ("jiuzaigou", "shimen", "taitung")
EXPECTED_COUNTS = {
    "landslide": {"train": 575, "val": 161, "test": 83},
    "building": {"train": 1815, "val": 522, "test": 259},
}


def collect_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise ValueError(f"Missing dataset directory: {directory}")
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def classify(name: str) -> str:
    if name.startswith("LBFD_whu_"):
        return "building"
    if any(name.startswith(f"LBFD_{source}_") for source in LANDSLIDE_SOURCES):
        return "landslide"
    raise ValueError(f"Unrecognized LBFD-CD sample name: {name}")


def scan_dataset(dataset_root: Path) -> dict[str, dict[str, list[str]]]:
    manifests = {
        category: {split: [] for split in SPLITS}
        for category in EXPECTED_COUNTS
    }
    seen: dict[str, str] = {}

    for split in SPLITS:
        names_by_component = {
            component: collect_names(dataset_root / split / component)
            for component in COMPONENTS
        }
        if not (
            names_by_component["t1"]
            == names_by_component["t2"]
            == names_by_component["label"]
        ):
            raise ValueError(f"t1/t2/label filenames differ in the {split} split")

        for name in sorted(names_by_component["t1"]):
            previous_split = seen.setdefault(name, split)
            if previous_split != split:
                raise ValueError(
                    f"Sample {name} appears in both {previous_split} and {split}"
                )
            manifests[classify(name)][split].append(name)

    for category, split_counts in EXPECTED_COUNTS.items():
        for split, expected_count in split_counts.items():
            actual_count = len(manifests[category][split])
            if actual_count != expected_count:
                raise ValueError(
                    f"{category}_{split}: found {actual_count} samples; "
                    f"expected {expected_count}"
                )
    return manifests


def manifest_text(names: list[str]) -> str:
    return "\n".join(names) + "\n"


def write_manifests(
    manifests: dict[str, dict[str, list[str]]],
    output_dir: Path,
    check: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mismatches = []

    for category in ("landslide", "building"):
        for split in SPLITS:
            path = output_dir / f"{category}_{split}.txt"
            expected = manifest_text(manifests[category][split])
            if check:
                actual = path.read_text(encoding="utf-8-sig") if path.is_file() else None
                if actual != expected:
                    mismatches.append(path.name)
            else:
                path.write_text(expected, encoding="utf-8", newline="\n")

    if mismatches:
        raise ValueError(
            "Split files are missing or out of date: " + ", ".join(mismatches)
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=script_dir.parent,
        help="LBFD-CD root containing train/val/test (default: repository root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "splits",
        help="directory for the six split files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing split files instead of rewriting them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifests = scan_dataset(args.dataset_root)
        write_manifests(manifests, args.output_dir, args.check)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    action = "verified" if args.check else "generated"
    print(f"Successfully {action} six split files in {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
