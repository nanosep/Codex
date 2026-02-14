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
| LogisticRegression | 0.452186 | 0.144428 | 0.566928 | 0.144852 |
| KNeighborsClassifier | 0.589739 | 0.053890 | 0.619153 | 0.193271 |
| RandomForestClassifier | 0.612214 | 0.162478 | 0.608576 | 0.196045 |
| GaussianNB | 0.416896 | 0.197420 | 0.577016 | 0.226647 |

## Winner and rationale
RandomForestClassifier selected by highest mean CV ROC AUC; balanced accuracy used as tie-breaker. Threshold fixed at 0.50.

## Test set results
- test_roc_auc: 0.46553446553446554
- test_balanced_acc: 0.421079

## Failure modes
- label strictness sensitivity (tp/sl/horizon)
- class imbalance changes
- non-stationarity / regime shifts
- leakage risk if features ever use future info
- threshold sensitivity (0.5 may be wrong)

## Operational notes
- Demo synthetic dataset is for pipeline validation and stable end-to-end runs.
- Realistic synthetic dataset is market-like and can produce lower positive rates.
