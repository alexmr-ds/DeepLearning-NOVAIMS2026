# DeepLearning-NOVAIMS2026

WikiArt artist classification project for the NOVA IMS 2026 deep learning coursework. The current repository focuses on three things: exploring the WikiArt dataset, generating deterministic train/validation/test splits, and training TensorFlow transfer-learning baselines with a ResNet50 backbone.

## Current Status

- The tracked raw dataset lives in `data/wikiart/` and currently contains 23 artist classes with 13,340 `.jpg` images.
- The working split pipeline is `src/data/split_dataset.py`, which creates `data/train/`, `data/validation/`, and `data/test/` from the tracked raw dataset.
- The raw-dataset cleanup utility is `src/preprocessing/remove_duplicates.py`, which removes duplicate WikiArt images listed in `src/preprocessing/images_to_remove.json` directly from `data/wikiart/`.
- The top-level preprocessing entrypoint is `main.py`, which runs duplicate cleanup first and dataset splitting second.
- The current reusable training components are:
  - `src/models/resnet50.py` for the ResNet50 transfer-learning model with built-in augmentation
  - `src/metrics/classification.py` for sparse-label macro F1
  - `src/utils/utils.py` for dataset loading, MixUp, and image hashing helpers
- The active experiment entrypoints are `EDA/EDA.ipynb`, `notebooks/explore_wikiart.ipynb`, `notebooks/alexandre_NN.ipynb`, and `notebooks/alexandre_NN_regularized.ipynb`.
- `notebooks/alexandre_NN_regularized.ipynb` is the isolated regularized training notebook for the next staged ResNet50 run.
- The latest tracked experiment artifact is `notebooks/results/history3.csv`, which ends at `val_macro_f1 = 0.7325`.
- `tests/test_main.py` validates that the preprocessing entrypoint runs cleanup before splitting and that split-module imports stay quiet.

## Quick Start

Set up the environment with `uv`:

```bash
uv venv
source .venv/bin/activate
uv sync --all-groups
```

If you need the older pip-style dependency flow, `requirements.txt` is still available, but `uv` is the primary setup path for this project.

Run duplicate cleanup followed by split generation:

```bash
uv run python main.py
```

Run only the train/validation/test split:

```bash
uv run python src/data/split_dataset.py
```

Remove duplicate raw WikiArt images before regenerating splits when needed:

```bash
uv run python src/preprocessing/remove_duplicates.py
```

Important split-script behavior:

- It reads the tracked raw dataset from `data/wikiart/`.
- It writes generated folders to `data/train/`, `data/validation/`, and `data/test/`.
- It uses ratios `0.70 / 0.15 / 0.15` with deterministic seed `73`.
- It only copies `.jpg` files.
- It refuses to run if split folders already exist, so remove or rename them before regenerating.

With the current tracked raw dataset, the generated split sizes are:

- `train`: 9,326 images
- `validation`: 1,992 images
- `test`: 2,022 images

Launch the notebooks from the `notebooks/` directory so the current relative-path assumptions stay valid:

```bash
cd notebooks
uv run jupyter lab
```

Open:

- `alexandre_NN_regularized.ipynb` for the current regularized staged fine-tuning run
- `alexandre_NN.ipynb` for the current training and fine-tuning workflow
- `explore_wikiart.ipynb` for raw dataset inspection

For dataset-wide exploratory analysis and saved figures, use `EDA/EDA.ipynb`.

## Workflow

1. Inspect the raw dataset in `data/wikiart/` with `EDA/EDA.ipynb` or `notebooks/explore_wikiart.ipynb`.
2. Run `main.py` to remove known duplicate raw images and then generate local split folders in one sequence.
3. Use `src/preprocessing/remove_duplicates.py` or `src/data/split_dataset.py` directly only when you need to run one preprocessing step in isolation.
4. Train from `notebooks/alexandre_NN_regularized.ipynb`, which:
   - appends `src/` to `sys.path`
   - loads split folders with `load_image_datasets(...)`
   - builds a ResNet50 classifier with `build_model(...)`
   - tracks macro F1 with `SparseMacroF1`
   - trains in two stages: classifier head first, then fine-tunes the top 30 backbone layers
   - adds Adam weight decay, early stopping, ReduceLROnPlateau, and stronger head dropout
   - writes isolated artifacts to `training_log_regularized.csv`, `best_model_regularized.keras`, and `results/history_regularized.csv`

## Repository Tree

```text
DeepLearning-NOVAIMS2026/
├── .gitignore
├── .python-version
├── EDA/
│   ├── EDA.ipynb
│   ├── images_per_artist.png
│   ├── pixel_intensity_boxplot.png
│   ├── pixel_intensity_boxplot_rgb.png
│   ├── pixel_intensity_by_artist.png
│   ├── pixel_intensity_rgb_by_artist.png
│   └── shape_combinations.png
├── HPC_SETUP.md
├── README.md
├── data/
│   └── wikiart/
│       ├── Albrecht_Durer/
│       ├── ...
│       └── Vincent_van_Gogh/
├── documents/
│   └── Deep_Learning_Project.pdf
├── main.py
├── notebooks/
│   ├── Data Understanding - Group 8.ipynb
│   ├── alexandre_NN.ipynb
│   ├── alexandre_NN_regularized.ipynb
│   ├── explore_wikiart.ipynb
│   ├── results/
│   │   ├── history1.csv
│   │   ├── history2.csv
│   │   └── history3.csv
│   └── training_log.csv
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── data/
│   │   └── split_dataset.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── classification.py
│   ├── models/
│   │   └── resnet50.py
│   ├── preprocessing/
│   │   ├── images_to_remove.json
│   │   └── remove_duplicates.py
│   └── utils/
│       └── utils.py
├── tests/
│   ├── test_main.py
│   └── test_remove_duplicates.py
└── uv.lock
```

Tracked files and directories:

- `.gitignore`: ignores generated split folders, virtual environments, caches, and saved model artifacts.
- `.python-version`: local Python version pin for tools that respect it.
- `EDA/EDA.ipynb`: main exploratory data analysis notebook for the tracked raw WikiArt dataset.
- `EDA/images_per_artist.png`: saved chart of image counts per artist.
- `EDA/pixel_intensity_boxplot.png`: saved grayscale intensity boxplot.
- `EDA/pixel_intensity_boxplot_rgb.png`: saved RGB intensity boxplot.
- `EDA/pixel_intensity_by_artist.png`: saved grayscale intensity summary by artist.
- `EDA/pixel_intensity_rgb_by_artist.png`: saved RGB intensity summary by artist.
- `EDA/shape_combinations.png`: saved chart of image shape frequencies.
- `HPC_SETUP.md`: Deucalion HPC access, environment setup, and execution notes.
- `README.md`: project overview, workflow, and repository map.
- `data/wikiart/<artist>/*.jpg`: tracked raw WikiArt images organized by artist.
- `documents/Deep_Learning_Project.pdf`: project brief and supporting reference material.
- `main.py`: sequential preprocessing entrypoint that removes duplicate raw WikiArt images and then builds train/validation/test splits.
- `notebooks/Data Understanding - Group 8.ipynb`: early dataset understanding notebook.
- `notebooks/alexandre_NN.ipynb`: current training notebook for the ResNet50 pipeline.
- `notebooks/alexandre_NN_regularized.ipynb`: isolated regularized training notebook with staged fine-tuning and separate artifact paths.
- `notebooks/explore_wikiart.ipynb`: notebook for raw dataset inspection and visualization.
- `notebooks/results/history1.csv`: archived training history from an earlier experiment.
- `notebooks/results/history2.csv`: archived training history from an intermediate experiment.
- `notebooks/results/history3.csv`: latest tracked consolidated training history.
- `notebooks/training_log.csv`: CSV log generated by notebook training runs.
- `pyproject.toml`: project metadata and the primary dependency definition for `uv`.
- `requirements.txt`: legacy pip-style dependency list kept for alternate environments.
- `src/data/split_dataset.py`: deterministic dataset splitter for train/validation/test generation.
- `src/metrics/__init__.py`: metrics package marker.
- `src/metrics/classification.py`: custom macro-F1 metric for sparse integer labels.
- `src/models/resnet50.py`: ResNet50 transfer-learning model builder with augmentation layers.
- `src/preprocessing/images_to_remove.json`: curated list of raw WikiArt image paths flagged for duplicate removal.
- `src/preprocessing/remove_duplicates.py`: raw-dataset cleanup script that normalizes duplicate-path entries and deletes matching files only from `data/wikiart/`.
- `tests/test_main.py`: tests for the top-level preprocessing pipeline order and quiet split-module imports.
- `src/utils/utils.py`: dataset loading, MixUp, exact hash, and perceptual hash utilities.
- `tests/test_remove_duplicates.py`: regression tests for duplicate-removal path normalization, safety checks, and cwd-independent execution.
- `uv.lock`: locked dependency resolution for reproducible `uv` installs.

## Data Layout

Tracked raw dataset:

```text
data/
└── wikiart/
    ├── artist_a/
    │   ├── image_001.jpg
    │   └── ...
    ├── artist_b/
    └── ...
```

Generated local split folders after running the splitter:

```text
data/
├── wikiart/      # tracked raw dataset
├── train/        # generated locally, ignored by Git
├── validation/   # generated locally, ignored by Git
└── test/         # generated locally, ignored by Git
```

The current notebooks and utility functions expect the generated split folders to exist under `data/`.

## Notes

- `main.py` is a preprocessing entrypoint, not a training CLI; notebook workflows remain the primary training path.
- `src/preprocessing/remove_duplicates.py` resolves duplicate file paths relative to the repository root, so it can be launched from outside `src/preprocessing/`.
- `notebooks/alexandre_NN.ipynb` assumes a 23-class problem in the current training helpers and notebook logic.
- `notebooks/alexandre_NN_regularized.ipynb` keeps sparse-label training, leaves MixUp disabled on purpose, and writes separate training artifacts so the baseline notebook outputs are preserved.
- `EDA/EDA.ipynb` is more robust about locating the project root than the training notebook, so it can be run from more locations.
- Cluster-specific execution instructions live in `HPC_SETUP.md`.
