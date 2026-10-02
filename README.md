# PharMDTA

PharMDTA is a multimodal framework for drug–target binding affinity prediction. This repository contains the research implementation, benchmark configurations, and training and evaluation utilities for BindingDB and KIBA.

The model integrates pharmacophore-aware molecular representations, frozen ESM-C-6B protein residue embeddings, and sequence-aligned binding-pocket graphs. Bidirectional atom–residue co-attention models drug–target interactions and provides attention-based evidence alongside affinity predictions.

## Repository structure

```text
PharMDTA/
├── configs/
│   ├── bindingdb.json           # BindingDB model and training settings
│   └── kiba.json                # KIBA model and training settings
├── data/
│   ├── component_sequences.csv  # Protein component sequence lookup
│   ├── bindingdb/               # Curated splits and dataset manifest
│   └── kiba/                    # Curated splits and dataset manifest
├── scripts/                     # Training entry points and reporting utilities
├── src/model/                   # Model, input validation, training, and evaluation
├── environment.yml             # Conda environment specification
└── README.md
```

## Installation

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate PharMDTA
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The minimal environment pins Python 3.10.18 and the six direct runtime dependencies: PyTorch 2.6.0 (CUDA 12.4), PyTorch Geometric 2.7.0, RDKit 2023.9.6, NumPy 1.26.4, pandas 2.2.2, and tqdm 4.66.5. These support training, evaluation, dataset splitting, and reporting with precomputed features.

The specification targets Linux with CUDA 12.4 PyTorch wheels. GPU training requires a compatible NVIDIA driver; PyTorch resolves its CUDA runtime dependencies without a separate CUDA toolkit installation. The `--device cpu` option is available for small smoke runs; memory requirements depend on the input sizes and micro-batch size.

The graph operations used here do not require the optional `torch-scatter`, `torch-sparse`, `torch-cluster`, or `torch-spline-conv` extensions; see the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html). ESM-C and pocket preprocessing tools are outside this runtime environment because the repository consumes their precomputed outputs.

After installation, check dependency consistency and run the attention regression tests:

```bash
python -m pip check
python -m unittest discover -s tests -v
```

Direct dependencies are pinned; transitive dependencies are resolved by the package managers. Archive `conda list --explicit` and `python -m pip freeze` with published experiment results to record the exact installed environment.

## Data and feature requirements

Local split CSVs and generated feature artifacts are excluded from Git and are not bundled with this repository. Supply the benchmark split files described by the dataset manifests before generating features or starting training.

Training requires four inputs:

| Argument | Required input |
| --- | --- |
| `--data-dir` | Curated dataset directory containing `manifest.json` and `splits/{train,val,test}.csv`. |
| `--pocket-contract` | A sequence-aligned version-4 target-pocket contract JSON and the graph files it references. |
| `--component-sequences` | Protein component sequence lookup CSV; the repository provides `data/component_sequences.csv`. |
| `--esmc6b-embeddings` | Precomputed ESM-C-6B residue embeddings keyed by target identity. |

The dataset manifests define fixed drug-wise splits with seed 42. Unique canonical SMILES are assigned to training, validation, and test sets at approximately 64%, 16%, and 20%, respectively. Ligands are disjoint across the three sets, while targets may overlap. These are single held-out splits without cross-validation. The manifest in each dataset directory records the exact protocol, row counts, label statistics.

### Affinity labels

| Dataset | Endpoint | Label used by the implementation |
| --- | --- | --- |
| BindingDB | IC50 | `pAffinity`: pIC50 = −log10(IC50 [M]) = 9 − log10(IC50 [nM]). |
| KIBA | Integrated KIBA score | `kiba_score_raw`: the released score without transformation. |

Inputs must satisfy the repository's dataset contracts, including source-data revision 6 and the recorded Lipinski-filter audit. Training reads the curated splits without changing labels or reassigning samples.

### Precomputed features

Pocket graphs are target-level, ligand-independent inputs. Their contract must specify sequence alignment and the required feature schema. The default configurations use 2,568-dimensional pocket node inputs and 28-dimensional residue physicochemical features. Pairs with pockets declared unavailable are excluded, and the training and validation exclusion counts are recorded in the run configuration.

The ESM-C cache may be a directory of per-target `.pt`, `.pth`, `.pkl`, or `.pickle` files, or a single mapping file from target identity to residue embeddings. Each embedding must have shape `[sequence_length, 2560]`; conventional leading and trailing special-token embeddings are also accepted. Target identifiers and residue order must match the sequences used by the dataset and pocket contract.

Pocket graphs and ESM-C feature caches can be generated locally using the scripts below. Training consumes the completed artifacts at `data/features/pockets/contract.json` and `data/features/esmc6b/`.

### Generating pocket graphs and ESM-C caches

Preprocessing requires protein structures for every target, local ESM-C-6B model weights, fpocket, DSSP (`mkdssp`), and Biopython. The ESM-C inference interpreter must use a Transformers implementation supporting the local checkpoint's `esmc` model type; these offline preprocessing dependencies are separate from the minimal training environment.

Prepare the target inventory from the dataset sequences and a source contract whose `entries` map target identities to `structure_path` and `source_group`:

```bash
python scripts/prepare_feature_targets.py \
  --source-contract /path/to/source_pocket_contract.json
```

Run the complete pipeline on two available GPUs, specifying the Python interpreters for ESM-C inference and pocket preprocessing:

```bash
python scripts/run_feature_generation.py \
  --esmc-python /path/to/esmc_python \
  --pocket-python /path/to/pocket_python \
  --model-dir /path/to/ESMC-6B
```

The pipeline reruns fpocket and DSSP on sequence-aligned protein-only structures and generates ESM-C residue states from the supplied sequences. Full sequences are encoded in non-overlapping chunks of at most 2,046 residues, then stored in FP16 without BOS/EOS rows. Pocket node features combine the corresponding new ESM-C states with eight DSSP features. Structures may originate from holo complexes; retaining only protein records does not establish an apo conformation.

Progress and logs are saved under `data/features/logs/`. The fpocket timeout defaults to 1,800 seconds to accommodate large protein structures. The final pocket contract is written after all graph features are attached and is followed by runtime graph validation. Unavailable pockets are explicitly marked for exclusion by the training loader. Rerunning the scripts resumes completed outputs; use a fresh output directory for a new generation run.

After the pipeline finishes, validate all six dataset loaders and run an untrained-model forward smoke check:

```bash
python scripts/validate_generated_features.py --device cuda:0
```

The validation report is saved to `data/features/validation.json`. This checks input compatibility and finite outputs; it does not evaluate prediction accuracy.

## Training

### BindingDB

```bash
python scripts/train.py \
  --config configs/bindingdb.json \
  --data-dir data/bindingdb \
  --pocket-contract data/features/pockets/contract.json \
  --component-sequences data/component_sequences.csv \
  --esmc6b-embeddings data/features/esmc6b \
  --output-dir runs \
  --device cuda \
  --micro-batch-size 16 \
  --num-workers 4
```

### KIBA

```bash
python scripts/train.py \
  --config configs/kiba.json \
  --data-dir data/kiba \
  --pocket-contract data/features/pockets/contract.json \
  --component-sequences data/component_sequences.csv \
  --esmc6b-embeddings data/features/esmc6b \
  --output-dir runs \
  --device cuda \
  --micro-batch-size 16 \
  --num-workers 4
```

Replace the feature paths with the locations of the corresponding benchmark artifacts. Default run directories are `runs/bindingdb_seed42/` and `runs/kiba_seed42/`. Use `--run-name` to select a different name. A new run requires an empty or nonexistent run directory.

The shell wrapper provides an alternative entry point:

```bash
DATASET=kiba \
DATA_DIR=data/kiba \
POCKET_CONTRACT=data/features/pockets/contract.json \
COMPONENT_SEQUENCES=data/component_sequences.csv \
ESMC6B_EMBEDDINGS=data/features/esmc6b \
bash scripts/train.sh
```

### Experimental settings

| Setting | BindingDB | KIBA |
| --- | --- | --- |
| Hidden dimension | 384 | 384 |
| Molecular Transformer layers | 8 | 8 |
| Protein Transformer layers | 4 | 4 |
| Pocket graph layers | 3 | 3 |
| Atom–residue co-attention layers | 3 | 3 |
| Maximum SMILES token length | 128 | 128 |
| Maximum protein sequence length | 1,000 | 1,000 |
| Optimizer | AdamW | AdamW |
| Learning rate | 5 × 10⁻⁵ | 2.5 × 10⁻⁵ |
| Weight decay | 0.075 | 0.075 |
| Batch size | 128 | 128 |
| Warm-up steps | 3,500 | 500 |
| Maximum epochs | 100 | 100 |
| Early-stopping patience | 10 | 10 |
| Random seed | 42 | 42 |

The JSON configurations are the authoritative source for all settings. Training minimizes MSE after standardizing labels using the retained training samples. Validation standardized MSE determines checkpoint selection and early stopping. The test split is evaluated after selecting the best checkpoint. Mixed precision is enabled on CUDA, and micro-batching controls memory usage within each configured batch.

### Resuming a run

Repeat the original training command with the same configuration, inputs, and run directory, adding:

```bash
--resume runs/bindingdb_seed42/last.pt
```

The resume checkpoint must belong to the selected run directory. Resumption restores model, optimizer, scheduler, mixed-precision scaler, and random-number-generator states and checks the saved configuration.

## Evaluation and interpretation

Evaluate the best checkpoint on the test split:

```bash
python -m model.evaluate \
  --checkpoint runs/bindingdb_seed42/best.pt \
  --split test \
  --output-dir results/bindingdb_test \
  --device cuda \
  --micro-batch-size 16
```

Use `--split val` for validation or the KIBA checkpoint path for KIBA evaluation. Keep the checkpoint together with its run-level `config.json`, which supplies the configuration and input paths. If inputs have moved, override their locations with `--data-dir`, `--pocket-contract`, `--component-sequences`, and `--esmc6b-embeddings`. The output directory must be empty or nonexistent.

Evaluation writes `predictions.csv` and `metrics.json`. Reported metrics include MSE, RMSE, MAE, Pearson correlation, Spearman correlation, concordance index (CI), R², and residual standard deviation (SD), all computed on the original dataset label scale. CI excludes tied labels and assigns half credit to tied predictions.

Add `--attention-limit 10` to export atom–residue attention matrices and top-15 atom and residue importance summaries for the first ten retained split samples. These outputs describe the model's attention along the prediction path and support inspection of individual predictions.

## Run artifacts

| Artifact | Description |
| --- | --- |
| `best.pt` | Checkpoint with the lowest validation standardized MSE. |
| `last.pt` | Latest checkpoint, including states needed to resume training. |
| `config.json` | Run configuration, resolved input paths, and training label statistics. |
| `data_contract.json` | Input validation receipt and input paths. |
| `history.csv` | Per-epoch training loss, validation metrics, and learning rate. |
| `final_metrics.json` | Best epoch and validation/test metrics for the selected model. |
| `predictions/val_best.csv` | Validation predictions from the best checkpoint. |
| `predictions/test_best.csv` | Test predictions from the best checkpoint. |
| `training_stop.json` | Completed epoch count and early-stopping status. |

For reproducible comparisons, retain the configurations, manifests, feature artifacts, and run outputs together, and record the hardware and software environment used for each experiment. Actual checkpoint selection is determined by validation performance during the run.
