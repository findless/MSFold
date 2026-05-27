# MSFold

This release accompanies the manuscript *Sampling in structure-token space enables accurate prediction of multiple conformations*.

MSFold samples protein conformational ensembles using parallel tempering
with block-wise replica exchange, built on ESM3 structure tokens.

## Setup

### Step 1. Clone MSFold

Use the following commands to clone this repository and enter the project directory.

```bash
git clone git@github.com:findless/MSFold.git
cd MSFold
```

### Step 2. Clone ESM, switch to the required version, and complete ESM3 setup

MSFold depends on the official ESM3 codebase. Clone the ESM repository into `./esm`, switch it to `v3.1.1`, and then follow the official ESM instructions for any additional ESM3 setup and weight access.

```bash
git clone https://github.com/evolutionaryscale/esm.git esm
cd esm
git checkout v3.1.1
cd ..
```

Official ESM instructions:

- `https://github.com/evolutionaryscale/esm`

### Step 3. Create the environment and install ESM + MSFold
```bash
conda env create -f environment.yml
conda activate msfold
pip install -e ./esm
pip install -e .
```

## Run MSFold sampling

### Step 4. Run the example sampling job

Use the example FASTA and default configuration to verify that the installation works.

```bash
python scripts/run_msfold_sample.py \
  --fasta examples/example.fasta \
  --config configs/default.yaml \
  --out outputs/example \
  --device cuda
```

### Step 5. Check the output files

After the run finishes, the following files should appear under `outputs/example/`:

- `samples.csv`
- `*.pdb`
- `config_used.json`
- `run_metadata.json`

## Rank samples by sequence log-likelihood

### Step 6. Rank the sampled structures

Run the following command to write the SLL ranking table.

```bash
python scripts/rank_sll.py \
  --sample_dir outputs/example \
  --out outputs/example/sll_ranking.csv
```
