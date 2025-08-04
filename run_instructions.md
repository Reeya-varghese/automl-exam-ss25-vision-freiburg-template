# AutoML Exam - SS25 (Vision Data)

This repository contains the codebase for our submission to the AutoML SS25 exam at the University of Freiburg.
It includes our AutoML system, training code, dataset configuration, and prediction generation pipeline.

---

## Project Setup & Installation

Before running the code, set up your environment. You can use either `venv` or `conda`.

### Using `venv`

```bash
python3 -m venv automl-vision-env
source automl-vision-env/bin/activate
```

### Using `conda`

```bash
conda create -n automl-vision-env python=3.11
conda activate automl-vision-env
```

### Install Dependencies

Install the repo in editable mode:

```bash
pip install -e .
```

If the installation fails or you encounter import errors for `codecarbon` or `optuna`, install them manually:

```bash
pip install codecarbon
pip install optuna
```

Test the install:

```bash
python -c "import automl"
```

---

## Code Structure

Our codebase is organized as follows:

```
src/
└── automl/
    ├── dac.py
    ├── model.py
    ├── Plots.py
    ├── run.py
    ├── training.py
    ├── utils.py
    ├── vision_datasets.py
    └── Zero_cost.py
```

* `run.py` — Main entry point for running training and prediction.
* `training.py`, `model.py`, `utils.py` — Core modules implementing the AutoML logic.
* `dac.py`, `Zero_cost.py` — Additional search space and zero-cost proxy components.
* `Plots.py` — For generating performance plots.

We used both **Kaggle (P100 GPU)** and **Google Colab (T4 GPU)** for training and evaluation.

---

## Running Training & Predictions

You can run the pipeline like this :

```bash
!python src/automl/run.py --n-trials 10 --dataset fashion --seed 42 --output-path
```

The script supports the following arguments:

```python
parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials.")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["fashion", "flowers", "emotions", "skin_cancer"])
parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

parser.add_argument("--carbon-budget", type=float, default=0.15, help="Carbon budget in kg CO2eq")
parser.add_argument("--enable-progressive", action="store_true", help="Enable progressive training strategy")
parser.add_argument("--enable-carbon-manager", action="store_true", help="Enable carbon budget manager")
parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
```

---

##
