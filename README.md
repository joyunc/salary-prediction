# Salary Prediction for College Graduates
### Machine Learning + TabTransformer · Python · PyTorch

> Python reproduction and deep learning extension of the master's thesis:
> *"Applying Machine Learning and SEM to Explore Factors Affecting the Salary of College Graduates"*
> Fu Jen Catholic University, July 2024

---

## Project Overview

Predicts the salary of Taiwanese university graduates at two career stages using classical ML and deep learning models for tabular data, based on the Taiwan Education Panel Survey (TEPS / TEPS-B) conducted by Academia Sinica.

| Target | Survey Year | Description | Sample Size |
|--------|-------------|-------------|-------------|
| **Starting Salary (M1)** | 2014 | Salary upon entering the workforce | n = 1,511 |
| **Current Salary (M2)** | 2019 | Salary ~5 years into career | n = 769 |

---

## Models

Five models are trained and compared end-to-end:

| Model | Type | Notes |
|-------|------|-------|
| CART | Decision Tree | Complexity tuned via 10-fold CV |
| Random Forest | Ensemble | 400 trees, mtry = p/3 |
| MLP | Deep Learning | 3-layer MLP baseline |
| **TabTransformer** | Deep Learning | Self-attention over categorical embeddings |
| **FT-Transformer** | Deep Learning | Feature tokenizer + Transformer |

---

## Results

Test set performance (30% held-out split, random seed = 2056):

### Starting Salary (M1, n = 1,511)

| Model | Test R² | Test RMSE (NTD) | Test MAPE |
|-------|---------|-----------------|-----------|
| CART | -0.603 | 10,041 | 24.75% |
| **Random Forest** | **0.284** | **6,710** | **16.48%** |
| MLP | 0.209 | 7,053 | 17.13% |
| TabTransformer | 0.137 | 7,369 | 18.22% |
| FT-Transformer | 0.168 | 7,233 | 17.49% |

### Current Salary (M2, n = 769)

| Model | Test R² | Test RMSE (NTD) | Test MAPE |
|-------|---------|-----------------|-----------|
| CART | -0.918 | 15,681 | 26.45% |
| **Random Forest** | **0.250** | **9,805** | **18.32%** |
| MLP | 0.136 | 10,526 | 18.45% |
| TabTransformer | 0.009 | 11,274 | 20.49% |
| FT-Transformer | 0.226 | 9,961 | 17.74% |

**Random Forest achieves the best generalization on both targets.** Deep learning models (TabTransformer, MLP) exhibit significant overfitting, which is expected given the relatively small sample size for neural networks.

---

## Key Findings

- **Occupation type** is the strongest salary predictor in both models (RF importance ~11%)
- **Cognitive ability** (comprehensive test, math test, GSAT score) ranks consistently in the top 5
- **School characteristics** (type, major) have stronger influence on starting salary than current salary
- **Survey-based features have low intrinsic predictive power** (max Pearson r ≈ 0.18), capping model R² regardless of architecture — this reflects the fundamental limitation of education survey data, not model failure
- **Random Forest > deep learning** on this dataset, consistent with literature showing tree ensembles outperform Transformers when n < 5,000 for tabular data

---

## Project Structure

```
salary_project/
├── src/
│   ├── preprocessing.py      # Data loading, imputation, merging pipeline
│   ├── modeling.py           # Training & evaluation for all 5 models
│   ├── tab_transformer.py    # TabTransformer (PyTorch, from scratch)
│   ├── ft_transformer.py     # FT-Transformer (PyTorch, from scratch)
│   └── mlp.py                # MLP baseline (PyTorch)
├── notebooks/
│   └── salary_analysis.ipynb # Full reproducible analysis
├── data/                     # Raw CSVs — access required (see below)
├── outputs/                  # Generated plots
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/salary-prediction.git
cd salary-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Obtain data and place CSVs in ./data/
#    Apply at: https://srda.sinica.edu.tw/
#    Required files: 2005_parents.csv, 2007_student.csv,
#                    cp2013.csv, cpn2014.csv, cp2019.csv

# 4. Launch notebook
jupyter lab notebooks/salary_analysis.ipynb
```

---

## Data

| File | Survey | Content |
|------|--------|---------|
| `2005_parents.csv` | TEPS 2005 | Family monthly income |
| `2007_student.csv` | TEPS 2007 | Cognitive ability tests, educational expectations |
| `cp2013.csv` | TEPS-B 2013 | Gender, marital status |
| `cpn2014.csv` | TEPS-B 2014 | Starting salary, occupation, education, job characteristics |
| `cp2019.csv` | TEPS-B 2019 | Current salary, updated job characteristics |

**Access**: [Survey Research Data Archive, Academia Sinica](https://srda.sinica.edu.tw/)
Raw data is not included in this repository due to data use agreements.

---

## Tech Stack

- **Python 3.9**
- **PyTorch 2.2** — TabTransformer, FT-Transformer, MLP
- **scikit-learn 1.6** — CART, Random Forest, preprocessing
- **pandas / numpy / scipy** — data wrangling
- **matplotlib / seaborn** — visualization

---

## Reference

```bibtex
@mastersthesis{chang2024salary,
  author = {Jo-Yun Chang},
  title  = {Applying Machine Learning and Structural Equation Model
             to Explore Factors Affecting the Salary of College Graduates},
  school = {Fu Jen Catholic University},
  year   = {2024},
}
```
