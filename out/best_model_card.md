## Label definition
Step 2 binary label is reused exactly: for bar t, TP hit if future high >= tp_level and SL hit if future low <= sl_level over horizon bars excluding t; label=1 only when TP hit and SL not hit.

## Feature set
Step 3 feature set is reused exactly (16 trailing-only features from price/volume/indicators).

## Models compared
- LogisticRegression
- KNeighborsClassifier
- RandomForestClassifier
- GaussianNB

## Cross-validation results
TimeSeriesSplit with n_splits=5 on train-only chronology.

| Model | Mean ROC AUC | Std ROC AUC | Mean Bal Acc | Std Bal Acc |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.503792 | 0.025440 | 0.493205 | 0.033835 |
| KNeighborsClassifier | 0.484879 | 0.089956 | 0.482485 | 0.035029 |
| RandomForestClassifier | 0.472812 | 0.069934 | 0.483312 | 0.028579 |
| GaussianNB | 0.523965 | 0.062084 | 0.486127 | 0.026650 |

## Winner and rationale
GaussianNB selected by highest mean CV ROC AUC; balanced accuracy used as tie-breaker. Threshold fixed at 0.50.

## Test set results
- test_roc_auc: 0.7170865664016349
- test_balanced_acc: 0.499856

## Failure modes
- label strictness sensitivity (tp/sl/horizon)
- class imbalance changes
- non-stationarity / regime shifts
- leakage risk if features ever use future info
- threshold sensitivity (0.5 may be wrong)

## Operational notes
- Demo synthetic dataset is for pipeline validation and stable end-to-end runs.
- Realistic synthetic dataset is market-like and can produce lower positive rates.
