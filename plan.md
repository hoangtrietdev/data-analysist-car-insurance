# Car Insurance Claim Prediction — Project Plan

**Course:** Python for Data Science | ESIGELEC  
**Team:** Work must be completed **in pairs**  
**Dataset:** `car_insurance.csv` — 10,000 rows × 18 columns  
**Goal:** Supervised binary classification — predict whether a customer will file a car insurance claim (`outcome = 1`) or not (`outcome = 0`).  
**Deliverables:** (1) Commented Jupyter Notebook `.ipynb` with all code and outputs, (2) Concise written report answering all theoretical questions posed in each step.

---

## Dataset Overview

| Property | Details |
|---|---|
| Rows | 10,000 |
| Target column | `outcome` (0 = no claim, 1 = claim) |
| Class distribution | 6,867 no claim vs 3,133 claim — **imbalanced** |
| Missing values | `credit_score`: 982 missing (9.8%) · `annual_mileage`: 957 missing (9.6%) |

### Column Reference

| Column | Type | Values / Notes |
|---|---|---|
| `id` | int | Unique ID — drop before training |
| `age` | int (0–3) | Encoded age group |
| `gender` | int | 0 / 1 |
| `driving_experience` | str | `0-9y`, `10-19y`, `20-29y`, `30y+` |
| `education` | str | `none`, `high school`, `university` |
| `income` | str | `poverty`, `working class`, `middle class`, `upper class` |
| `credit_score` | float | 0–1, **982 missing** |
| `vehicle_ownership` | float | 0.0 / 1.0 |
| `vehicle_year` | str | `before 2015`, `after 2015` |
| `married` | float | 0.0 / 1.0 |
| `children` | float | 0.0 / 1.0 |
| `postal_code` | int | Geographic code — drop (no ordinal meaning) |
| `annual_mileage` | float | **957 missing** |
| `vehicle_type` | str | `sedan`, `sports car` |
| `speeding_violations` | int | Count |
| `duis` | int | Count |
| `past_accidents` | int | Count |
| `outcome` | float | **Target** — 0.0 or 1.0 |

---

## Project Steps

### Step 1 — Data Import

**Objective:** Load the dataset into a pandas DataFrame and verify the import.

- [ ] Import `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn` at the top of the notebook
- [ ] Load `car_insurance.csv` using `pd.read_csv()`
- [ ] Verify the import by displaying:
  - Variable information and data types: `df.info()`
  - First few rows: `df.head()`
  - Shape: `df.shape`
  - Basic statistics: `df.describe()`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                              recall_score, f1_score, classification_report)
from sklearn.pipeline import Pipeline

df = pd.read_csv('car_insurance.csv')

print("Shape:", df.shape)
df.info()
df.describe()
```

---

### Step 2 — Data Review

**Objective:** Identify qualitative data, missing values, and outliers.

- [ ] Identify qualitative (categorical) columns and their unique values: `df.select_dtypes(include='object').nunique()`
- [ ] Count missing values per column: `df.isna().sum()`
- [ ] Plot histograms for all numerical variables to observe distributions: `df.hist(...)`
- [ ] Plot box plots for numerical variables to detect outliers: `df.boxplot(...)`
- [ ] Report which columns contain missing values and which contain visible outliers

> **Report questions to answer:**
> - Which variables are qualitative? Which are quantitative?
> - Which columns have missing values, and how many?
> - Which numerical variables show outliers in the box plots?

```python
# Identify qualitative columns
print("Categorical columns:\n", df.select_dtypes(include='object').columns.tolist())

# Missing values
print("\nMissing values:\n", df.isna().sum())

# Histograms — distributions of all numerical features
df.hist(figsize=(16, 12), bins=20)
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Box plots — outlier detection
numerical_cols = df.select_dtypes(include='number').columns.tolist()
df[numerical_cols].plot(kind='box', subplots=True, layout=(3, 5), figsize=(18, 10))
plt.suptitle('Box Plots for Outlier Detection')
plt.tight_layout()
plt.show()
```

---

### Step 3 — Data Preparation

**Objective:** Clean the dataset so it is ready for machine learning.

#### 3.1 — Remove Irrelevant Columns

- [ ] Drop `id` — unique identifier with no predictive value
- [ ] Drop `postal_code` — high cardinality with no ordinal meaning; justify in report

```python
df = df.drop(columns=['id', 'postal_code'])
```

#### 3.2 — Handle Missing Values

**Rule:** If a variable has **more than 1/3 (~33%) of its values missing**, drop it with `drop()`. Otherwise, impute with `fillna()`.

- [ ] Apply the 1/3 rule to each column: `df.isna().sum() / len(df)`
- [ ] `credit_score` has 9.8% missing → **impute with median** (numerical, skewed)
- [ ] `annual_mileage` has 9.6% missing → **impute with median** (numerical, skewed)
- [ ] Verify zero missing values remain after imputation

```python
missing_ratio = df.isna().sum() / len(df)
print("Missing ratios:\n", missing_ratio[missing_ratio > 0])

# Both are well below 1/3 → impute with median
df['credit_score']   = df['credit_score'].fillna(df['credit_score'].median())
df['annual_mileage'] = df['annual_mileage'].fillna(df['annual_mileage'].median())

print("\nRemaining missing values:\n", df.isna().sum())
```

#### 3.3 — Handle Outliers

- [x] For each numerical column with outliers identified in Step 2, cap values using the IQR method (replace values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR with the boundary value)
- [ ] Re-plot box plots after capping to confirm outlier removal
- [ ] Justify your chosen replacement strategy in the report

```python
def cap_outliers(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return series.clip(lower=lower, upper=upper)

# Apply to numerical columns with visible outliers (e.g., speeding_violations, duis, past_accidents)
for col in ['speeding_violations', 'duis', 'past_accidents', 'annual_mileage']:
    df[col] = cap_outliers(df[col])

# Verify with box plots
df[['speeding_violations', 'duis', 'past_accidents', 'annual_mileage']].plot(
    kind='box', subplots=True, layout=(1, 4), figsize=(14, 4))
plt.suptitle('Box Plots After Outlier Capping')
plt.tight_layout()
plt.show()
```

#### 3.4 — Encode Categorical Variables

**Rule:** Use `replace()` for boolean/binary columns and `LabelEncoder` for other categorical columns.

- [ ] Boolean/binary columns — use `replace()` with explicit mappings:
  - `vehicle_year`: `{'before 2015': 0, 'after 2015': 1}`
  - `vehicle_type`: `{'sedan': 0, 'sports car': 1}`
- [ ] Multi-class ordinal columns — use `LabelEncoder` (or `replace()` with ordered mapping):
  - `driving_experience`: `{'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3}`
  - `education`: `{'none': 0, 'high school': 1, 'university': 2}`
  - `income`: `{'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3}`
- [ ] Verify all columns are now numeric: `df.dtypes`

> **Report question:** Why do we use `replace()` for boolean columns vs. `LabelEncoder` for others? What is the difference between nominal and ordinal encoding?

```python
# Boolean/binary → replace()
df['vehicle_year'] = df['vehicle_year'].replace({'before 2015': 0, 'after 2015': 1})
df['vehicle_type'] = df['vehicle_type'].replace({'sedan': 0, 'sports car': 1})

# Ordinal multi-class → LabelEncoder (learns mapping from data)
le = LabelEncoder()
for col in ['driving_experience', 'education', 'income']:
    df[col] = le.fit_transform(df[col])

print(df.dtypes)
print(df.head())
```

#### 3.5 — Feature Scaling

- [ ] Apply `StandardScaler` to normalize all input variables (zero mean, unit variance)
- [ ] Note: Scaling is applied **after** the train/test split (in Step 5) to prevent data leakage — `fit_transform` on train, `transform` only on test

> **Report question:** Why is normalization necessary? What would happen if features were on very different scales?

---

### Step 4 — Search for Correlations

**Objective:** Identify which inputs correlate most with each other and with the output class.

- [ ] Compute correlation coefficients using `df.corr()`
- [ ] Visualize the full correlation matrix as a heatmap (seaborn)
- [ ] Identify inputs most correlated with `outcome` (target variable)
- [ ] Identify pairs of inputs highly correlated with each other (multicollinearity risk)
- [ ] Select the most promising/correlated features and visualize them with `scatter_matrix()`

> **Report questions to answer:**
> - Which input variable is most correlated with `outcome`?
> - Are there pairs of inputs strongly correlated with each other? What is the risk?
> - What do the scatter plots reveal about the relationships between variables?

```python
# Correlation matrix
corr = df.corr()
print("Correlation with outcome:\n", corr['outcome'].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Scatter matrix — top correlated features with outcome
top_features = corr['outcome'].abs().nlargest(5).index.tolist()
scatter_matrix(df[top_features], figsize=(12, 12), alpha=0.3)
plt.suptitle('Scatter Matrix — Most Promising Variables')
plt.tight_layout()
plt.show()
```

---

### Step 5 — Train / Test Split

**Objective:** Divide the dataset into training and testing subsets before fitting any model.

- [ ] Separate features (`X`) from target (`y`)
- [ ] Cast `y` to integer
- [ ] Use `train_test_split()` from Scikit-Learn
- [ ] Apply `StandardScaler` — fit **only** on training data, transform both sets

```python
X = df.drop(columns=['outcome'])
y = df['outcome'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "  Test size:", X_test.shape)

# Scale AFTER split to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

---

### Step 6 — Train a Logistic Regression Model

**Objective:** Train a binary classifier using `LogisticRegression` and understand its parameters.

- [ ] Instantiate and fit `LogisticRegression` on the training set
- [ ] Print the learned coefficients (`coef_`) and intercept (`intercept_`)
- [ ] Interpret which features have the largest positive/negative influence

> **Report questions to answer (mandatory):**
> 1. **Logit hypothesis:** What is the hypothesis function $h_\theta(x)$ for Logistic Regression? Write the sigmoid function formula.
> 2. **Cost function:** What cost function is minimized during training (log-loss / binary cross-entropy)? Write its formula.
> 3. **Learned parameters:** After fitting, what do `coef_` and `intercept_` represent? Which feature has the largest absolute coefficient, and what does that mean?

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

```python
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Learned parameters
coef_df = pd.DataFrame({
    'feature':     X.columns,
    'coefficient': log_reg.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("Intercept:", log_reg.intercept_[0])
print("\nCoefficients (sorted by |magnitude|):")
print(coef_df.to_string(index=False))

# Visualize
sns.barplot(data=coef_df, x='coefficient', y='feature', palette='coolwarm')
plt.title('Logistic Regression Coefficients')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()
```

---

### Step 7 — Model Evaluation + Cross-Validation

**Objective:** Measure performance on unseen data using all required metrics, then validate with k-fold cross-validation.

#### 7.1 — Standard Evaluation (Hold-out Test Set)

- [ ] Predict classes on the test set: `log_reg.predict(X_test_scaled)`
- [ ] Compute **all required metrics** individually:
  - `accuracy_score()`
  - `confusion_matrix()` — visualize as annotated heatmap
  - `precision_score()`
  - `recall_score()`
  - `f1_score()`
- [ ] Print a full `classification_report()`
- [ ] Discuss what each quadrant of the confusion matrix means (TP, FP, FN, TN)

> **Report question:** Given the class imbalance (69% / 31%), why is accuracy alone a misleading metric? Which metric should be prioritized and why?

```python
y_pred = log_reg.predict(X_test_scaled)

print("Accuracy  :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall    :", recall_score(y_test, y_pred))
print("F1 Score  :", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred,
      target_names=['No Claim', 'Claim']))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Claim', 'Claim'],
            yticklabels=['No Claim', 'Claim'])
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('Confusion Matrix — Logistic Regression')
plt.tight_layout()
plt.show()
```

#### 7.2 — K-Fold Cross-Validation

- [ ] Use `KFold` with `n_splits=5`, `shuffle=True`
- [ ] Wrap scaler + model in a `Pipeline` to prevent data leakage across folds
- [ ] Apply `cross_val_score()` and report mean ± standard deviation
- [ ] Compare cross-validated accuracy vs. the hold-out test accuracy from 7.1

> **Report question:** Why does cross-validation give a more reliable estimate than a single train/test split? What does a high standard deviation across folds indicate?

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

pipeline_lr = Pipeline([
    ('scaler',     StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

cv_acc = cross_val_score(pipeline_lr, X, y, cv=kf, scoring='accuracy')
cv_f1  = cross_val_score(pipeline_lr, X, y, cv=kf, scoring='f1')

print(f"Hold-out Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"CV Accuracy       : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV F1 Score       : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
```

---

### Step 8 — Comparison with Other Algorithms + Hyperparameter Tuning

**Objective:** Compare 4 classifiers and tune hyperparameters for the best-performing model.

#### 8.1 — Algorithm Comparison

**Required classifiers (4 total — all mandatory):**

| # | Algorithm | scikit-learn Class |
|---|---|---|
| 1 | Logistic Regression | `LogisticRegression` (baseline) |
| 2 | Perceptron | `Perceptron` |
| 3 | K-Nearest Neighbors | `KNeighborsClassifier` |
| 4 | Your choice (e.g. Decision Tree) | `DecisionTreeClassifier` |

- [ ] Train and evaluate all 4 models on the **same** scaled train/test sets
- [ ] Record Accuracy, F1 Score, and 5-fold CV Accuracy for each
- [ ] Produce a summary comparison table (as a DataFrame)
- [ ] Plot a grouped bar chart comparing model accuracies and F1 scores

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Perceptron':           Perceptron(max_iter=1000, random_state=42),
    'KNN (k=5)':            KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':        DecisionTreeClassifier(random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    yp  = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, yp)
    f1  = f1_score(y_test, yp)
    cv  = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1, 'CV Accuracy': cv})
    print(f"{name:22s}  Acc={acc:.4f}  F1={f1:.4f}  CV={cv:.4f}")

results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
print("\n", results_df.to_string(index=False))

# Bar chart
results_df.set_index('Model')[['Accuracy', 'F1 Score', 'CV Accuracy']].plot(
    kind='bar', figsize=(10, 5), rot=30)
plt.title('Model Comparison'); plt.ylabel('Score'); plt.ylim(0.5, 1.0)
plt.tight_layout(); plt.show()
```

#### 8.2 — Hyperparameter Tuning

- [ ] Select the best-performing model from 8.1 for tuning
- [ ] Use `GridSearchCV` with cross-validation (`cv=5`) to search over hyperparameter combinations
- [ ] Report the best parameters and the corresponding CV score
- [ ] Refit the best model and compare its test performance to the un-tuned baseline

> **Report question:** What hyperparameter had the most impact on performance? Did tuning significantly improve results?

```python
from sklearn.model_selection import GridSearchCV

# --- Example: tune KNN ---
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn,
                        cv=5, scoring='f1', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)
print("Best KNN params :", grid_knn.best_params_)
print("Best KNN CV F1  :", grid_knn.best_score_)

# --- Example: tune Logistic Regression ---
param_grid_lr = {
    'C':       [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                       param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
print("\nBest LR params  :", grid_lr.best_params_)
print("Best LR CV F1   :", grid_lr.best_score_)

# Evaluate the best model on hold-out test set
best_model = grid_lr.best_estimator_   # or grid_knn.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print("\nTuned model — Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("Tuned model — Test F1 Score:", f1_score(y_test, y_pred_best))
```

---

### Step 9 — Save and Load the Trained Model

**Objective:** Serialize the best-performing model using `pickle` for production deployment.

- [ ] Use **`pickle.dump()`** to save the best model to a `.pkl` file
- [ ] Use **`pickle.load()`** to reload the model from disk
- [ ] Verify the reloaded model produces **identical** predictions to the original
- [ ] Save the `StandardScaler` separately (required for inference on new data)

> **Report question:** What is serialization? Why must the same scaler used during training be saved and reused at inference time?

```python
import pickle

# Save best model and scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved with pickle.")

# Load and verify
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

X_test_reloaded  = loaded_scaler.transform(X_test)
y_pred_reloaded  = loaded_model.predict(X_test_reloaded)

print("Predictions identical:", (y_pred_reloaded == y_pred_best).all())
print("Accuracy (reloaded)  :", accuracy_score(y_test, y_pred_reloaded))
```

---

## Deliverables Checklist

| # | Deliverable | Status |
|---|---|---|
| 1 | Jupyter Notebook `.ipynb` with all code, outputs, and markdown | [ ] |
| 2 | Written report answering all theoretical questions | [ ] |
| 3 | Step 1 — Data import verified (`info()`, `head()`, `describe()`) | [ ] |
| 4 | Step 2 — EDA: histograms + box plots, qualitative/missing/outlier identification | [ ] |
| 5 | Step 3 — Missing values handled (1/3 rule applied, imputed with median) | [ ] |
| 6 | Step 3 — Outliers replaced with appropriate values | [ ] |
| 7 | Step 3 — Categorical encoding: `replace()` for booleans, `LabelEncoder` for others | [ ] |
| 8 | Step 4 — Correlation matrix (`corr()`) + scatter_matrix on top features | [ ] |
| 9 | Step 5 — Train/test split + `StandardScaler` fitted on train only | [ ] |
| 10 | Step 6 — Logistic Regression trained + coefficients printed + theory answered | [ ] |
| 11 | Step 7 — All 5 metrics: `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score` | [ ] |
| 12 | Step 7 — Cross-validation: `KFold` + `cross_val_score`, mean ± std, comparison with hold-out | [ ] |
| 13 | Step 8 — 4 classifiers compared (LR, Perceptron, KNN, + 1 of choice) with bar chart | [ ] |
| 14 | Step 8 — Hyperparameter tuning with `GridSearchCV` on best model | [ ] |
| 15 | Step 9 — Model saved and loaded with **`pickle`** (`dump`/`load`), predictions verified | [ ] |

---

## Technical Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, `corr()`, `scatter_matrix()` |
| `numpy` | Numerical operations |
| `matplotlib` | Plots and visualizations |
| `seaborn` | Statistical visualizations (heatmaps, etc.) |
| `pickle` | **Model serialization** (`dump` / `load`) — required by project |
| `sklearn.model_selection` | `train_test_split`, `KFold`, `cross_val_score`, `GridSearchCV` |
| `sklearn.preprocessing` | `StandardScaler`, `LabelEncoder` |
| `sklearn.linear_model` | `LogisticRegression`, `Perceptron` |
| `sklearn.neighbors` | `KNeighborsClassifier` |
| `sklearn.tree` | `DecisionTreeClassifier` (4th required algorithm) |
| `sklearn.metrics` | `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `classification_report` |
| `sklearn.pipeline` | `Pipeline` (scaler + model for cross-validation, prevents leakage) |

---

## Report — Theoretical Questions Summary

The written report must answer the following questions (one section per step):

| Step | Question |
|---|---|
| Step 2 | Which variables are qualitative? Which have missing values? Which show outliers? |
| Step 3 | Justify the imputation strategy for each missing column. Explain `replace()` vs. `LabelEncoder`. |
| Step 4 | Which input correlates most with `outcome`? Are there multicollinear input pairs? |
| Step 6 | Write the logit/sigmoid hypothesis. Write the log-loss cost function. Interpret the learned coefficients. |
| Step 7 | Why is accuracy alone misleading for imbalanced data? Why is cross-validation preferred over a single split? |
| Step 8 | Which hyperparameter had the most impact? Did tuning improve generalization? |
| Step 9 | What is serialization? Why must the scaler be saved alongside the model? |

---

## Implementation Notes

### Step Ordering (Must Follow Official Sequence)
The official project defines this exact order: Data Import → Data Review → Data Preparation → **Correlations** → **Train/Test Split** → Train Model → Evaluate + CV → Compare + Tune → Save. Do **not** mix correlation analysis into Step 2.

### Data Leakage Prevention
- `StandardScaler` must be `fit_transform` on train data only, then `transform` on test — **never fit on test data**.
- During cross-validation, always wrap scaler + model in a `Pipeline` so each fold scales independently.

### Model Saving Library
The project **requires `pickle`** — do not use `joblib`. Always open in binary mode (`'wb'` to write, `'rb'` to read).

### Handling Class Imbalance
- Dataset is 69% / 31% — moderately imbalanced.
- Use `stratify=y` in `train_test_split`.
- Report **F1 Score** and **Precision/Recall** alongside accuracy; accuracy alone is misleading.

### Feature Importance
Expected most-predictive features based on correlation: `past_accidents`, `speeding_violations`, `duis`, `driving_experience`. Confirm via `coef_` after fitting Logistic Regression.

---

## Grading Risk Areas

| Risk | Mitigation |
|---|---|
| Using `joblib` instead of `pickle` | The project specification requires `pickle` — use `pickle.dump` / `pickle.load` |
| Fitting scaler on test data | `fit_transform` on train, `transform` only on test |
| Putting correlation analysis in Step 2 | Correlation (`corr()`, `scatter_matrix`) belongs in Step 4, not Step 2 |
| Missing `precision_score` or `recall_score` | Call them individually, not just via `classification_report` |
| Only comparing 3 algorithms | 4 are **mandatory**: LR + Perceptron + KNN + 1 of choice |
| No theoretical answers in report | Each step has required written questions — answer all of them |
| Not applying 1/3 missing-value rule | Show `isna().sum() / len(df)` and justify keep vs. drop decision |
| No outlier replacement step | Acknowledge and handle outliers in Step 3 (IQR capping or other) |
| Cross-validation outside a Pipeline | Use `Pipeline([scaler, model])` inside `cross_val_score` |
