"""Run duplicate cleanup and dataset splitting in sequence."""

from src.data import split_dataset
from src.preprocessing import remove_duplicates


def main() -> None:
    """Run raw WikiArt cleanup before generating dataset splits."""
    print("Step 1/2: Removing duplicate raw WikiArt images...")
    remove_duplicates.remove_duplicate_images()
    print("=" * 100)
    print("Step 2/2: Generating train/validation/test splits...")
    split_dataset.main()


if __name__ == "__main__":
    main()
