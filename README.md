# MSFold — Minimal Public Release

This release accompanies the manuscript *Sampling in structure-token space enables accurate prediction of multiple conformations*.

MSFold samples protein conformational ensembles using parallel tempering
with block-wise replica exchange, built on ESM3 structure tokens.

## What this repository provides

- **MSFold sampling**: Run parallel-tempered block Gibbs sampling on an
  input amino acid sequence.
- **SLL ranking**: Rank sampled conformations by sequence log-likelihood
  (inverse folding score).
- **Configuration**: Default hyperparameters matching the released MSFold setup.
- **Example input**: A placeholder sequence for smoke testing.

## What this repository does NOT provide

- TM-score evaluation scripts
- Benchmark reproduction scripts
- PDB download or chain extraction scripts
- Reference PDB/mmCIF structures
- Full raw benchmark outputs
- Table or figure reproduction scripts
- Any experimental script variants
- Source-data files under `data/`

## Installation

MSFold is intended to be used together with the official ESM codebase.

1. Clone this MSFold repository.

   Example:

   ```bash
   git clone <your-private-msfold-repo-url>
   ```

2. Clone the official ESM repository and follow its installation instructions:

   - `https://github.com/evolutionaryscale/esm`

3. Create and activate a fresh conda environment for MSFold:

```bash
conda env create -f environment.yml
conda activate msfold
```

4. From the standalone `release_msfold/` directory, install MSFold itself:

```bash
cd release_msfold
pip install -e .
```

MSFold must be installed and run from the standalone `release_msfold/` directory after cloning it from its own repository. It must not depend on any surrounding development workspace. It does require the official external ESM3 code and package installation.

The full manuscript source data are distributed separately from this code repository. Download the complete `data/` release from Zenodo.

This release is tested against the following dependency baseline:

- `esm==3.1.1`
- `torch==2.5.1`
- `transformers==4.46.3`
- `huggingface_hub==0.26.5`

Do not replace these with arbitrary newer package versions.

If `esm` is not resolved from the official ESM installation, fix that first before running MSFold.

## ESM3 weights

ESM3 open weights are hosted on Hugging Face and are gated.

Please follow the official ESM instructions and model page for access and download:

- Official ESM repository: `https://github.com/evolutionaryscale/esm`
- Model page: `https://huggingface.co/biohub/esm3-sm-open-v1`

In practice this usually means logging in with Hugging Face before the first run:

```bash
huggingface-cli login
```

## Run MSFold sampling

A minimal example is provided in `examples/example.fasta`.

```bash
python scripts/run_msfold_sample.py \
  --fasta examples/example.fasta \
  --config configs/default.yaml \
  --out outputs/example \
  --device cuda
```

Expected outputs include sampled PDB files, `samples.csv`, and auxiliary metadata under `outputs/example/`.

## Rank samples by sequence log-likelihood

```bash
python scripts/rank_sll.py \
  --sample_dir outputs/example \
  --out outputs/example/sll_ranking.csv
```

## Included files

The standalone MSFold code repository should include:

- `README.md`
- `environment.yml`
- `pyproject.toml`
- `.gitignore`
- `test_release.sh`
- `configs/`
- `examples/`
- `msfold/`
- `scripts/`

The standalone MSFold code repository should not include:

- `data/`
- internal compare scripts
- experimental workspace files
- generated outputs such as `outputs/`, `*.pdb`, or `*.pkl`
