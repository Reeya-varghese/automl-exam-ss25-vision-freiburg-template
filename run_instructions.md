# AutoML Exam - SS25 (Vision Data) Team Name - GenZs

This repository contains the codebase for our submission to the AutoML SS25 exam at the University of Freiburg.
It includes our AutoML system, training code, and prediction generation pipeline.

---

## Project Setup & Installation

Before running the code, set up your environment.

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
We have used all these modules in our project.
* `torchvision`
* `pandas`
* `scikit-learn`
* `numpy<2.0`
* `IPython`
* `optuna`
* `codecarbon`
As you can see we have added codecarbon and optuna to the requirements. You can install it by the following commands.

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

* `run.py` — Main entry point for running Hyperparameter tunning, training and prediction.
* `training.py`, `model.py`, `utils.py` — Core modules implementing the AutoML logic.
* `dac.py`, `Zero_cost.py` — Additional search space and zero-cost proxy components.

We used both **Kaggle (P100 GPU)** and **Google Colab (T4 GPU)** for training and evaluation. It approximately took us around 3 hours to get the prediction on skin cancer dataset. The file is saved as
final_test_preds.npy

---

## Running Training & Predictions

You can run the pipeline like this to get the final_test_preds.npy :

```bash
!python src/automl/run.py --dataset skin_cancer --n-trials 10 --seed 42  --carbon-budget 0.15 --enable-carbon-manager
```
### Model Performance Across Seeds (10 Trials Each)

| Dataset     | Metric                      | Seed 6        | Seed 42       | Seed 92       |
|-------------|-----------------------------|---------------|---------------|---------------|
| **flowers** | acc                         | 0.94          | 0.93          | 0.97          |
|             | F1                          | 0.93          | 0.98          | 0.96          |
| **emotions**| acc                         | 0.6847        | 0.6463        | 0.6779        |
|             | F1                          | 0.6686        | 0.6364        | 0.6623        |
| **fashion** | acc                         | 0.9454        | 0.9333        | 0.9378        |
|             | F1                          | 0.9453        | 0.9331        | 0.9375        |
| **skincancer** | acc - Val                | 0.78          | 0.84          | 0.73          |
|             |                             |               |   **0.86**    |               |

### Final Submission Metrics on the predictions.npy of Skin Cancer
- Accuracy: **0.8626**
- Precision (Micro): **0.8004**
- F1 (Micro): **0.7961**

## Submission checklist:

- [ ] Poster  
- [x] Test predictions  
- [x] Reproducibility instructions  
- [x] Team info  


