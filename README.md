# MSFold

This release accompanies the manuscript *Sampling in structure-token space enables accurate prediction of multiple conformations*.

MSFold samples protein conformational ensembles using parallel tempering
with block-wise replica exchange, built on ESM3 structure tokens.

This repository provides the released MSFold sampling and sample-ranking
workflow together with the default configuration and a small example input.

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

The full `data/` release is distributed separately on Zenodo.

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

## Repository contents

The MSFold code repository contains the code, configuration, and examples
needed to install and run MSFold. The full `data/` release is distributed
separately on Zenodo.
