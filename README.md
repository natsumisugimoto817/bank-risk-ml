# Interpretable Credit Scoring with EBM

Can a glass-box model be as accurate as a black-box model for credit risk?

On Lending Club loans, an **Explainable Boosting Machine (EBM)** came within **0.004 AUC of XGBoost** on an out-of-time test year, while every prediction stays fully explainable.

This is Project 1 of a series on explainable ML for banking:

| # | Project | Status |
|---|---|---|
| 1 | Credit scoring with EBM (this notebook: `src/01_credit_scoring_ebm.ipynb`) | Done |
| 2 | Model monitoring: drift and recalibration | Planned |
| 3 | Financial crime with PU learning (some fraud is never labelled) | Planned |

## Why EBM

A credit model has to be accurate, but it also has to be explained: to regulators (e.g. MAS FEAT principles), to model risk teams, and to customers who ask why their loan was rejected.

- **Logistic regression** is easy to explain, but it can only learn straight-line effects.
- **XGBoost** is usually the most accurate, but it is a black box. Explanations need a separate tool such as SHAP, which approximates the model after the fact.
- **EBM** tries to give both: accuracy close to boosted trees, and a model you can read directly.

## How EBM works

EBM is a Generalized Additive Model (GAM). The prediction is a sum of one function per feature, plus a few pairwise terms:

```
logit(p) = intercept + f1(fico) + f2(dti) + ... + f20(purpose) + f(x_i, x_j) + ...
```

Each `f` is a **shape function** learned from the data. It can take any shape, not only a straight line.

**Training (cyclic gradient boosting)**

1. Take one feature. Fit a very small tree on that feature only, to the current errors, with a very low learning rate.
2. Move to the next feature and repeat. Cycle through all features for thousands of rounds. Because each step is small and the features take turns, the order of the features doesn't matter much.
3. Add up all the small trees of each feature. The result is one shape function per feature, stored as a lookup table of bins.
4. Find the most useful feature pairs and learn small pairwise terms the same way (up to 10 pairs in this project).
5. Repeat on several random samples of the data (bagging). The average gives the final shape; the spread gives an uncertainty band.

**Prediction**: look up each feature value in its shape function, add the values, and apply the sigmoid.

**Why this matters**: the contribution of each feature is part of the model itself. It is exact, and there is no extra step like SHAP.

**Trade-offs**: training can be slower than XGBoost (here 42 s vs 35 s on 167k loans), interactions are limited to pairs, and correlated features share their effect, so each can look less important than the group really is.

## Data and setup

- **Data**: [Lending Club accepted loans, 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
- **Scope**: 36-month loans issued 2010–2015. All of them matured by the end of 2018, so the outcome is known. 612,745 loans, default rate 13.9%.
- **Target**: 1 = Charged Off / Default, 0 = Fully Paid. Loans still current or late are dropped.
- **Features**: 20 features known at application time (16 numeric, 4 categorical), including two I built: `loan_to_income` and `credit_hist_months`.
  - Lending Club's own `grade` and `int_rate` are **not used**: they are outputs of Lending Club's risk model, so using them would copy their model.
  - Information known only after the loan starts (payments, recoveries) is not used, to avoid target leakage.
- **Out-of-time split**: train on older loans, test on newer loans, the way a bank actually uses a model.

| Split | Years | Loans | Default rate |
|---|---|---|---|
| Train | 2010–2013 | 167,149 | 12.4% |
| Validation | 2014 | 162,570 | 13.7% |
| Test | 2015 | 283,026 | 14.9% |

## Results: EBM vs other models (test year 2015)

| Model | AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|
| **EBM** | **0.661** | **0.323** | **0.235** | **0.240** | **0.1220** |
| XGBoost | 0.666 | 0.332 | 0.241 | 0.245 | 0.1216 |
| Logistic regression | 0.653 | 0.306 | 0.222 | 0.234 | 0.1226 |
| Random model / predict the average | 0.500 | 0.000 | 0.000 | 0.149 | 0.1267 |

- EBM is **0.004 AUC below XGBoost** and 0.008 above logistic regression. We give up almost nothing for a fully explainable model.
- Ranking power holds from 2014 to 2015 (EBM AUC 0.660 → 0.661).
- An AUC around 0.66 is modest. That is expected when only application data is used, without Lending Club's own grade.

![Model comparison: ROC, precision-recall, KS and calibration on the 2015 test year](images/model_comparison.png)

## What drives default risk

### Permutation importance

How much AUC drops when one feature is shuffled. `fico` is by far the most important. The top four features are the same for EBM and XGBoost, so the two models rely on the same information.

![Permutation importance for EBM and XGBoost](images/permutation_importance.png)

### Shape functions

Each plot is one feature's contribution to the log-odds of default. Above 0 = riskier than average.

![EBM shape functions for the top 6 features](images/shape_functions.png)

- `fico`: risk falls steadily as FICO goes up.
- `inq_last_6mths`: each extra credit inquiry in the last 6 months adds risk.
- `loan_to_income`: borrowing more compared with income increases risk.
- `annual_inc`: lower income means higher risk; the effect flattens above about $100k.
- `revol_bal`: a very low card balance is riskier, other things being equal (possibly a thin credit file).
- `loan_amnt`: flat up to about $25k, then risk goes up.

### EBM vs a one-variable view

Blue = EBM shape (other features taken into account). Red = default rate by bin from the data alone, on the same log-odds scale.

![EBM shape vs one-variable view](images/shape_vs_one_variable.png)

- `fico`, `inq_last_6mths`, `loan_to_income`: almost the same in both views. They work on their own.
- `annual_inc`: the red line drops more than the blue one. Part of the income effect is already carried by `loan_to_income` and `dti`, which are built from income.
- `loan_amnt`: on its own, bigger loans look **safer**, because bigger loans go to higher-income borrowers. Once income is taken into account, loans above about $25k are **riskier**. A one-variable analysis would give the wrong answer here.

## Reason codes for each applicant

For any applicant, EBM gives the exact contribution of each feature. The highest-risk applicant in the test year (predicted default probability 64%) has these top reasons: 5 credit inquiries in the last 6 months, a loan of half their annual income, and a FICO score of 662.

![Reason codes for the highest-risk applicant](images/reason_codes.png)

## Business impact

If the bank rejects the riskiest 10% of applicants by EBM score (test year 2015):

- 19.2% of all defaulters are caught.
- 28.6% of the rejected applicants would have defaulted, about 2x the average rate of 14.9%.
- The bad rate of approved loans falls from 14.9% to 13.4%. At an 80% approval rate, it falls to 12.2%.

![Approval rate vs bad rate of approved loans](images/approval_curve.png)

## Limitations and next steps

- **Probabilities shift over time.** The ranking holds in 2015, but the actual default rate is higher than predicted in every score band (top band: 28.6% actual vs 27.5% predicted). The Brier score is only a little better than predicting the average for everyone (0.1220 vs 0.1267). If the probabilities are used directly (pricing, provisions), the model should be recalibrated on recent data and monitored. This is Project 2.
- **Correlated features share importance.** For example, `pub_rec` and `pub_rec_bankruptcies` have a Spearman correlation of 0.93. Next step: measure importance for groups of related features, an idea from a patent I co-invented at KPMG Japan.
- **US data.** Features like FICO don't exist in the same form in Singapore, but the approach carries over.

## Repository structure

```
bank-risk-ml/
├── src/
│   ├── 01_credit_scoring_ebm.ipynb   # this project
│   ├── eda_utils.py                  # data quality, distributions, correlation, PSI
│   ├── eval_utils.py                 # metrics table, ROC / PR / KS / calibration, permutation importance
│   └── ebm_utils.py                  # shape functions, interactions, EBM vs one-variable view
├── images/                           # figures used in this README
├── data/                             # raw data (not in git)
└── model/                            # saved models (not in git)
```

## How to run

```bash
pip install interpret xgboost scikit-learn pandas matplotlib joblib ipywidgets
kaggle datasets download -d wordsforthewise/lending-club -p data/raw/
```

Unzip the data into `data/raw/`, then run `src/01_credit_scoring_ebm.ipynb` from top to bottom.

## Background

At KPMG Japan, I built a fraud risk model for financial statements using EBM. It is used by over 1,000 auditors, who needed to understand why a company got a high risk score. This project applies the same idea to retail credit with public data.
