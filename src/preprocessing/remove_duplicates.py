"""Remove duplicate WikiArt images from the tracked raw dataset."""

from __future__ import annotations

import json
from pathlib import Path


def _resolve_runtime_roots(script_file: Path) -> tuple[Path, Path, Path]:
    """Return the script directory, repository root, and raw WikiArt directory."""
    script_dir = script_file.resolve().parent
    repo_root = script_dir.parents[1]
    dataset_root = (repo_root / "data" / "wikiart").resolve()
    return script_dir, repo_root, dataset_root


def _resolve_json_path(json_file: str | Path | None, script_dir: Path) -> Path:
    """Resolve the duplicate-list JSON path from the caller or script directory."""
    if json_file is None:
        return script_dir / "images_to_remove.json"

    json_path = Path(json_file).expanduser()
    if json_path.is_absolute():
        return json_path

    return (Path.cwd() / json_path).resolve(strict=False)


def _resolve_image_path(image_path: str, repo_root: Path, dataset_root: Path) -> Path:
    """Resolve one JSON image path to a safe target inside the raw WikiArt tree."""
    normalized_repo_root = repo_root.resolve(strict=False)
    normalized_dataset_root = dataset_root.resolve(strict=False)
    raw_path = Path(image_path).expanduser()

    if raw_path.is_absolute():
        candidate = raw_path.resolve(strict=False)
    else:
        normalized_path = raw_path
        if normalized_path.parts and normalized_path.parts[0] == normalized_repo_root.name:
            normalized_path = Path(*normalized_path.parts[1:])

        if normalized_path.parts[:2] == ("data", "wikiart"):
            candidate = (normalized_repo_root / normalized_path).resolve(strict=False)
        elif normalized_path.parts[:1] == ("wikiart",):
            candidate = (normalized_repo_root / "data" / normalized_path).resolve(strict=False)
        else:
            candidate = (normalized_dataset_root / normalized_path).resolve(strict=False)

    if not candidate.is_relative_to(normalized_dataset_root):
        raise ValueError(f"path resolves outside {normalized_dataset_root}: {image_path}")

    return candidate


def _load_images_to_remove(json_path: Path) -> list[str]:
    """Load and validate the duplicate image path list from JSON."""
    with json_path.open("r", encoding="utf-8") as file_handle:
        images_to_remove = json.load(file_handle)

    if not isinstance(images_to_remove, list):
        raise ValueError(f"{json_path} must contain a JSON array of image paths")

    return images_to_remove


def _remove_duplicate_images(
    json_path: Path,
    repo_root: Path,
    dataset_root: Path,
) -> dict[str, int]:
    """Delete duplicate images listed in JSON and return removal statistics."""
    images_to_remove = _load_images_to_remove(json_path)
    stats = {"removed": 0, "missing": 0, "skipped_invalid": 0}

    for image_path in images_to_remove:
        if not isinstance(image_path, str):
            print(f"Skipped invalid path entry: {image_path!r}")
            stats["skipped_invalid"] += 1
            continue

        try:
            target_path = _resolve_image_path(image_path, repo_root, dataset_root)
        except ValueError:
            print(f"Skipped invalid path: {image_path}")
            stats["skipped_invalid"] += 1
            continue

        if target_path.exists():
            target_path.unlink()
            stats["removed"] += 1
        else:
            print(f"File not found: {target_path}")
            stats["missing"] += 1

    return stats


def remove_duplicate_images(json_file: str | Path | None = None) -> None:
    """Remove duplicate WikiArt images using the configured JSON path list."""
    script_dir, repo_root, dataset_root = _resolve_runtime_roots(Path(__file__))
    json_path = _resolve_json_path(json_file, script_dir)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"WikiArt dataset directory not found: {dataset_root}")

    stats = _remove_duplicate_images(json_path, repo_root, dataset_root)
    print(f"Removed {stats['removed']} duplicate images")
    print(f"Missing files: {stats['missing']}")
    print(f"Skipped invalid paths: {stats['skipped_invalid']}")


if __name__ == "__main__":
    remove_duplicate_images()
