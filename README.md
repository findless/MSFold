# MSFold

This release accompanies the manuscript *Sampling in structure-token space enables accurate prediction of multiple conformations*.

MSFold samples protein conformational ensembles using parallel tempering
with block-wise replica exchange, built on ESM3 structure tokens.

## Installation

Clone the official ESM repository into `./esm` and follow its installation instructions:

- `https://github.com/evolutionaryscale/esm`

Then create and activate the MSFold environment from the repository root:

```bash
conda env create -f environment.yml
conda activate msfold
pip install -e .
```

## ESM3 weights

Please follow the official ESM instructions for ESM3 weights access and download:

- `https://github.com/evolutionaryscale/esm`

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

