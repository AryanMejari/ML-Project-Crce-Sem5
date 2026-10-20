# IoT Machine Failure Prediction

Machine-learning experiments for predicting machine failures from the AI4I 2020 predictive-maintenance dataset.

The project compares several classification algorithms, evaluates their predictions, and combines selected models into a soft-voting ensemble.

## Project status

This repository currently contains:

- A Jupyter notebook with the complete exploratory and modeling workflow.
- The dataset in CSV format.
- The same dataset in Excel format.
- A comparison of individual models and an ensemble model.

The main workflow is in [`first.ipynb`](./first.ipynb).

## Dataset

The dataset contains 10,000 machine-operation records and the following columns:

| Column | Description |
|---|---|
| `UDI` | Unique row identifier |
| `Product ID` | Product identifier |
| `Type` | Product quality/type category: `L`, `M`, or `H` |
| `Air temperature [K]` | Ambient air temperature in Kelvin |
| `Process temperature [K]` | Machine process temperature in Kelvin |
| `Rotational speed [rpm]` | Rotational speed in revolutions per minute |
| `Torque [Nm]` | Applied torque in Newton-metres |
| `Tool wear [min]` | Tool usage/wear time in minutes |
| `Machine failure` | Target variable: `0` for no failure and `1` for failure |
| `TWF` / `Tool_Wear_Failure` | Tool wear failure indicator |
| `HDF` / `Heat_Dissipation_Failure` | Heat dissipation failure indicator |
| `PWF` / `Power_Failure` | Power failure indicator |
| `OSF` / `Overstrain_Failure` | Overstrain failure indicator |
| `RNF` / `Random_Failure` | Random failure indicator |

The CSV contains 6,500 non-failure rows and 3,500 failure rows. The target is therefore not perfectly balanced.

## Important leakage warning

The five failure-mode columns are strongly related to the target:

```text
TWF
HDF
PWF
OSF
RNF
```

They are useful for understanding the type or cause of a failure. However, they may be generated from the same operating measurements or become available only after a failure has been identified. If so, using them as model inputs creates **target leakage** and can produce unrealistically high scores.

This project intentionally experiments with these columns as predictors, but results should be interpreted carefully.

For a realistic early-warning system, use only measurements available before the failure:

```python
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
```

For a failure-explanation or post-event classification task, the failure-mode columns may be included, provided that their availability at prediction time is documented.

## Repository structure

```text
.
├── first.ipynb
├── AI4I_2020_10000_Rows_35pct_Failure.csv
├── AI4I_2020_10000_Rows_35pct_Failure.xlsx
└── README.md
```

## Requirements

Python 3.9 or newer is recommended.

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

The notebook uses:

- `pandas` for loading and manipulating the data.
- `numpy` for numerical operations.
- `matplotlib` and `seaborn` for plots.
- `scikit-learn` for preprocessing, models, metrics, and ensembles.
- `openpyxl` for reading the Excel version of the dataset.
- Jupyter for running the notebook.

## Running the project

Clone the repository and enter its directory:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Start Jupyter:

```bash
jupyter notebook
```

Open [`first.ipynb`](./first.ipynb) and run the cells from top to bottom.

The notebook expects the CSV file to be available in the same directory. If the data is stored elsewhere, update the path used by `pd.read_csv()`.

## Workflow

The notebook follows this general process:

1. Import Python and machine-learning libraries.
2. Load the CSV dataset.
3. Inspect data types, shape, columns, and summary statistics.
4. Rename the five failure-mode columns for readability.
5. Select model features and the target.
6. Split the data into training and test sets.
7. Preserve the target class proportions with stratification.
8. One-hot encode the categorical `Type` column.
9. Keep preprocessing and modeling together in scikit-learn pipelines.
10. Train multiple classification models.
11. Generate predictions and failure probabilities.
12. Compare accuracy, precision, recall, F1-score, and ROC-AUC.
13. Plot confusion matrices and model performance.
14. Train a soft-voting ensemble.

## Feature selection

The current notebook uses the following feature set:

```python
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Tool_Wear_Failure",
    "Heat_Dissipation_Failure",
    "Power_Failure",
    "Overstrain_Failure",
    "Random_Failure"
]

target = "Machine failure"

X = df[features]
y = df[target]
```

`UDI` and `Product ID` are not used as predictors because they are identifiers rather than physical measurements.

## Train-test split

The notebook uses an 80/20 stratified split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`random_state=42` makes the split reproducible. `stratify=y` keeps a similar failure/non-failure ratio in both sets.

The test set should be treated as unseen data and used for final evaluation, not repeated model tuning.

## Preprocessing

`Type` is categorical, so it is converted to numerical indicator columns using one-hot encoding:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = ["Type"]

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Tool_Wear_Failure",
    "Heat_Dissipation_Failure",
    "Power_Failure",
    "Overstrain_Failure",
    "Random_Failure"
]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "numeric",
        "passthrough",
        numeric_features
    )
])
```

The notebook includes compatibility handling for older scikit-learn versions:

```python
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )
```

## Models trained

### 1. Decision Tree

The first model is a constrained Decision Tree:

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion="gini",
    random_state=42
)
```

Decision Trees are interpretable and can be visualized as decision rules. `max_depth` and `min_samples_leaf` limit complexity and help reduce overfitting.

### 2. Random Forest

The Random Forest configuration used in the notebook is:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
```

It combines many decision trees and usually provides a strong non-linear baseline for tabular sensor data.

### 3. K-Nearest Neighbours

The KNN configuration is:

```python
KNeighborsClassifier(
    n_neighbors=21,
    weights="uniform",
    metric="manhattan"
)
```

KNN predicts based on nearby training observations. Its results depend strongly on feature scaling, the number of neighbours, the distance metric, and the weighting scheme.

### 4. Gaussian Naive Bayes

```python
GaussianNB()
```

Naive Bayes is a simple probabilistic baseline. It assumes conditional independence between features within each class.

### 5. AdaBoost

```python
AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
```

AdaBoost trains weak learners sequentially and focuses subsequent learners on examples that were previously misclassified.

### 6. Gradient Boosting

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

Gradient Boosting builds an additive model in stages, with each stage improving the errors of the previous stages.

### 7. Feed-forward neural network

The FFNN uses scikit-learn's `MLPClassifier`:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=42
)
```

The network has two hidden layers with 64 and 32 neurons. `early_stopping=True` reserves part of the training data for validation and stops training when validation performance stops improving.

### 8. Perceptron

```python
Perceptron(
    max_iter=1000,
    eta0=0.01,
    penalty="l2",
    random_state=42
)
```

The Perceptron is a simple linear classifier. It is useful as a low-complexity baseline.

### 9. Support Vector Classifier

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)
```

The RBF kernel allows the SVC to model non-linear decision boundaries. `probability=True` enables probability estimates for ROC-AUC calculations and soft voting.

## Ensemble model

The notebook combines four models using soft voting:

```python
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(
    estimators=[
        ("random_forest", rf_model),
        ("gradient_boosting", gb_model),
        ("ffnn", ffnn_model),
        ("svc", svc_model)
    ],
    voting="soft",
    weights=[3, 3, 2, 2]
)
```

Soft voting averages the class probabilities from the component models. The weights give more influence to the Random Forest and Gradient Boosting models in this experiment.

The ensemble is not automatically guaranteed to outperform every individual model. It should be compared against the individual models using the same test set or, preferably, cross-validation.

## Evaluation metrics

The notebook calculates the following metrics:

### Accuracy

The percentage of all predictions that are correct:

```text
(true positives + true negatives) / all predictions
```

Accuracy can be misleading when classes are imbalanced or when the cost of missing a failure is high.

### Precision

Of the rows predicted as failures, precision measures how many were actually failures.

High precision means fewer false alarms.

### Recall

Of all actual failures, recall measures how many were detected.

High recall means fewer missed failures. This is often an important metric in predictive maintenance.

### F1-score

The F1-score is the harmonic mean of precision and recall. It is useful when both false alarms and missed failures matter.

### ROC-AUC

ROC-AUC measures how well the model ranks failure cases above non-failure cases across classification thresholds.

Rough interpretation:

```text
0.50  random ranking
0.70  acceptable
0.80  good
0.90+ excellent
1.00  perfect separation
```

Very high scores should be investigated for leakage, especially when the five failure-mode indicators are included.

## Visualizations

The notebook creates or supports:

- Target-class distribution.
- Confusion matrices.
- ROC curves.
- Precision-recall curves.
- Decision Tree visualization.
- Decision Tree feature importance.
- Model-metric comparison plots.

Example confusion matrix:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Reproducibility

The main experiments use:

```python
random_state=42
```

This makes the train-test split and supported model components reproducible. Results can still vary when changing:

- The train-test split.
- Feature selection.
- Preprocessing.
- Model hyperparameters.
- scikit-learn version.
- Whether failure-mode columns are included.

## Recommended improvements

The current notebook is an educational model-comparison workflow. The following improvements are recommended for a stronger experiment:

1. **Run a leakage-safe experiment** without `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.
2. **Use stratified cross-validation** rather than relying only on one train-test split.
3. **Scale numeric features for KNN, SVC, FFNN, and Perceptron.** The current shared preprocessing is designed for the tree-based workflow; distance-based and neural models generally benefit from `StandardScaler`.
4. **Tune models using only the training data**, for example with `GridSearchCV` or `RandomizedSearchCV`.
5. **Keep the final test set untouched** until model selection is complete.
6. **Choose a classification threshold based on maintenance costs**, not only the default threshold of `0.50`.
7. **Report confidence intervals or cross-validation mean and standard deviation.**
8. **Use model explainability tools** such as permutation importance or SHAP for a production-oriented analysis.
9. **Document when every feature becomes available** in the real machine-monitoring process.
10. **Save the complete preprocessing and model pipeline** for deployment instead of saving only the classifier.

## Example of a leakage-safe feature list

```python
early_warning_features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
```

## Limitations

- The dataset is a static tabular dataset rather than a live time series.
- The rows should not automatically be interpreted as a continuous machine timeline.
- The model does not currently provide a deployment API or real-time sensor ingestion.
- The failure-mode columns can leak information about the target.
- A high test score does not by itself prove that the model will work on new machines or operating conditions.
- The model has not been validated against production sensor data.

## License and data attribution

Add the license and dataset attribution required by the source from which the AI4I 2020 dataset was obtained. Do not publish a license for the dataset unless its terms permit redistribution.

## Author

Add your name, GitHub profile, and contact information here.
