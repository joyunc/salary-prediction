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
| **Starting Salary (M1)** | 2014 | Salary upon entering the workforce | n = 1,591 |
| **Current Salary (M2)** | 2019 | Salary ~5 years into career | n = 849 |

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

### Starting Salary (M1, n = 1,591)

| Model | Test R² | Test RMSE (NTD) | Test MAPE |
|-------|---------|-----------------|-----------|
| CART | 0.194 | 7,402 | 18.70% |
| **Random Forest** | **0.277** | **7,011** | **17.90%** |
| MLP | 0.173 | 7,495 | 18.67% |
| TabTransformer | 0.150 | 7,603 | 18.62% |
| FT-Transformer | 0.259 | 7,098 | 17.21% |

### Current Salary (M2, n = 849)

| Model | Test R² | Test RMSE (NTD) | Test MAPE |
|-------|---------|-----------------|-----------|
| CART | 0.013 | 12,805 | 23.16% |
| **Random Forest** | **0.156** | **11,842** | **21.50%** |
| MLP | 0.080 | 12,365 | 21.23% |
| TabTransformer | -0.010 | 12,949 | 22.37% |
| FT-Transformer | 0.069 | 12,438 | 20.48% |

**Random Forest achieves the best generalization on both targets.** All models show significant overfitting on M2, likely due to the smaller sample size (n=849) and higher variance in current salary compared to starting salary.

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
