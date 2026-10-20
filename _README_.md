# E-Commerce Customer Behavior, Spending & Sales Segmentation

**Course:** 23CSE301 – Machine Learning  
**Project:** Machine Learning Case Study  
**Topics:** Classification, Regression, and Clustering

This project applies three machine learning approaches to e-commerce and sales data:

- **Classification:** Predict whether an online shopping session results in a purchase.
- **Regression:** Estimate customer spending.
- **Clustering:** Discover natural groups in sales transactions without predefined labels.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets and Tasks](#datasets-and-tasks)
- [Classification: Model Comparison](#classification-model-comparison)
- [Regression: Model Comparison](#regression-model-comparison)
- [Clustering: Sales Segmentation](#clustering-sales-segmentation)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Workflow](#workflow)
- [Team Members](#team-members)
- [Review Scope](#review-scope)
- [Reproducibility and Data Notes](#reproducibility-and-data-notes)

## Project Overview

| Notebook | Task | Target / Approach | Objective |
|---|---|---|---|
| `Classification.ipynb` | Classification | `Revenue` | Predict whether a shopping session results in a purchase |
| `23CSE301_Regression(2).ipynb` | Regression | `Monetary` | Estimate customer spending |
| `23CSE301_Clustering (1).ipynb` | Clustering | No target label | Group sales transactions into segments |

## Datasets and Tasks

### 1. Online Shoppers Purchasing Intention

**CSV filename:** `online_shoppers_intention.csv`

The dataset describes online shopping sessions, including browsing activity, session duration, bounce and exit rates, page values, and visitor type.

| Item | Description |
|---|---|
| Task | Binary classification |
| Target | `Revenue` |
| Prediction | Whether a session results in a purchase |
| Evaluation | Accuracy, weighted precision/recall/F1, ROC-AUC, classification report, and confusion matrix |

The notebook audits the data, explores numerical and categorical features, checks missing and duplicate values, preprocesses and encodes features, scales data where appropriate, and compares multiple classification algorithms.

**Classification models evaluated:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- AdaBoost
- Gradient Boosting
- Bagging with Decision Trees
- Multi-Layer Perceptron (MLP) Neural Network

### 2. E-Commerce Customer Spending

**CSV filename:** `ecommerce_user_segmentation.csv`

This dataset contains customer and shopping-behavior information used to estimate the `Monetary` target.

| Item | Description |
|---|---|
| Task | Regression |
| Target | `Monetary` |
| Prediction | Customer spending |
| Evaluation | R² score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) |

**Regression models evaluated:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

The notebook includes exploratory data analysis, feature preparation, model comparison, tuning, cross-validation, and prediction visualizations.

### 3. Sales Transaction Segmentation

**CSV filename:** `store.csv`

The clustering notebook uses sales transaction data with numerical features such as `qty` and `revenue`, and categorical features such as `reps`, `product`, and `region`. The goal is to discover groups of similar transactions for possible use in sales analysis, inventory planning, and targeted marketing.

| Item | Description |
|---|---|
| Task | Unsupervised clustering |
| Target | None; clusters are inferred from the data |
| Features | `qty`, `revenue`, `reps`, `product`, `region` |
| Algorithms | K-Means and Agglomerative Hierarchical Clustering |
| Evaluation | Silhouette score, Davies-Bouldin index, and Calinski-Harabasz index |
| Visualization | Elbow curve, dendrogram, PCA, and t-SNE |

The notebook handles missing values, removes duplicate rows, standardizes numerical features, one-hot encodes categorical features, selects a candidate cluster count using the elbow method, evaluates both clustering algorithms, and visualizes the cluster assignments.

## Classification: Model Comparison

The following values are recorded in `Classification.ipynb`.

| Model | Accuracy | Weighted F1-score | ROC-AUC |
|---|---:|---:|---:|
| Bagging (Decision Tree) | **0.8415** | **0.8018** | 0.7426 |
| Gradient Boosting | 0.8394 | 0.7919 | **0.7846** |
| Random Forest | 0.7714 | 0.7888 | 0.7793 |
| MLP Neural Network | 0.8386 | 0.7886 | 0.7551 |
| K-Nearest Neighbors | 0.8222 | 0.7821 | 0.6382 |
| AdaBoost | 0.8435 | 0.7801 | 0.7561 |
| Gaussian Naive Bayes | 0.6796 | 0.7205 | 0.7227 |
| Support Vector Machine | 0.6731 | 0.7168 | 0.7506 |
| Logistic Regression | 0.6448 | 0.6933 | 0.7560 |
| Decision Tree | 0.6022 | 0.6561 | 0.7188 |

**Best model by weighted F1-score: Bagging (Decision Tree)**, with a weighted F1-score of **0.8018** and accuracy of **0.8415** in the recorded comparison. Gradient Boosting has the highest ROC-AUC in this table (**0.7846**), so the preferred model depends on the metric and project objective.

The target classes are imbalanced: purchase sessions are less common than non-purchase sessions. Therefore, accuracy alone can be misleading. Check the purchase-class precision, recall, and F1-score in the notebook before drawing conclusions about purchase detection.

Five-fold cross-validation on the training data recorded the following weighted F1 results for the two top models:

| Model | Mean 5-fold CV weighted F1 | Standard Deviation |
|---|---:|---:|
| Bagging (Decision Tree) | 0.7994 | 0.0033 |
| Gradient Boosting | 0.7952 | 0.0051 |

## Regression: Model Comparison

The initial model comparison recorded these test-set results:

| Model | R² score | RMSE | MAE |
|---|---:|---:|---:|
| Random Forest Regressor | **0.8469** | **507.84** | **351.94** |
| Gradient Boosting Regressor | 0.8441 | 512.45 | 356.60 |
| KNN Regressor | 0.8352 | 527.00 | 360.31 |
| Polynomial Regression | 0.8179 | 553.89 | 380.43 |
| ElasticNet Regression | 0.7898 | 595.15 | 438.98 |
| Ridge Regression | 0.7893 | 595.75 | 439.52 |
| Linear Regression | 0.7893 | 595.75 | 439.53 |
| Lasso Regression | 0.7893 | 595.76 | 439.54 |
| Decision Tree Regressor | 0.7050 | 704.94 | 464.97 |
| Support Vector Regressor | 0.4174 | 990.74 | 630.68 |

The notebook then tunes the leading models:

| Tuned Model | Recorded Settings | R² score | RMSE | MAE |
|---|---|---:|---:|---:|
| Random Forest Regressor | `max_depth=10`, `n_estimators=100` | **0.8484** | **505.45** | **349.81** |
| Gradient Boosting Regressor | `learning_rate=0.05`, `max_depth=5`, `n_estimators=100` | 0.8473 | 507.21 | 353.25 |

**Best recorded regression model: Tuned Random Forest Regressor.** It has the highest test-set R² score and the lowest RMSE and MAE among the two tuned models shown. Its five-fold cross-validation mean R² score is approximately **0.8518**.

Higher R² is generally better; lower RMSE and MAE indicate smaller prediction errors. The units of RMSE and MAE depend on the units of the `Monetary` target.

## Clustering: Sales Segmentation

The clustering notebook compares K-Means with Agglomerative Hierarchical Clustering. The elbow-curve analysis selects **six clusters** as a reasonable candidate, although the bend is gradual rather than sharply defined.

| Algorithm | Number of Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|---:|---:|---:|---:|
| K-Means | 6 | **0.1940** | **1.8238** | **1450.2858** |
| Agglomerative Hierarchical | 6 | 0.1758 | 1.8924 | 1334.3137 |

**Preferred clustering model in this comparison: K-Means.** It has the higher Silhouette and Calinski-Harabasz scores and the lower Davies-Bouldin index. However, the Silhouette scores are low, which suggests that the clusters overlap and are not strongly separated. The groups should therefore be interpreted cautiously and checked for practical meaning in the sales data.

The notebook also includes:
- **Elbow method** to help select the number of clusters.
- **Dendrogram** to inspect the hierarchy of merged groups.
- **PCA** to visualize cluster assignments in two dimensions.
- **t-SNE** to explore local structure in a sample of the data.

For clustering metrics, a higher Silhouette score and Calinski-Harabasz index are generally preferred, while a lower Davies-Bouldin index is preferred. These metrics should be considered together rather than treated as proof that the segments are commercially meaningful.

## Technology Stack

- **Language:** Python
- **Notebook environment:** Jupyter Notebook
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Hierarchical clustering support:** SciPy

## Repository Structure

```text
.
├── README.md
├── Classification.ipynb
├── 23CSE301_Regression(2).ipynb
├── 23CSE301_Clustering (1).ipynb
└── datasets/
    ├── online_shoppers_intention.csv
    ├── ecommerce_user_segmentation.csv
    └── store.csv
```

This is a suggested layout based on the notebook filenames and expected CSV filenames. The clustering notebook currently reads `store.csv` from its working directory. Update the notebook paths if you keep data inside `datasets/`. Add CSV files to the public repository only when their sources and licenses permit redistribution.

## Getting Started

### 1. Clone the repository

Replace the placeholders with your actual GitHub repository URL and folder name.

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

Install the packages:

```bash
pip install -r requirements.txt
```

### 4. Place the datasets

Place each CSV file where its notebook expects it, or update the relevant `pd.read_csv(...)` path.

```python
pd.read_csv("online_shoppers_intention.csv")
pd.read_csv("ecommerce_user_segmentation.csv")
pd.read_csv("store.csv")
```

### 5. Run the notebooks

```bash
jupyter notebook
```

Open the required notebook and run the cells from top to bottom. Confirm that the results and visualizations are generated successfully from the data and code in the repository.

## Workflow

1. Load and audit the dataset.
2. Perform exploratory data analysis.
3. Handle missing values, duplicates, categorical features, and scaling as appropriate.
4. Prepare features and target variables for supervised learning.
5. Train and compare classification or regression models.
6. Evaluate using metrics appropriate to each task.
7. Tune selected models and use cross-validation where included.
8. Apply clustering to discover groups without target labels.
9. Evaluate cluster quality and visualize the results.
10. Interpret results and document limitations.

## Team Members

| No. | Team Member | Section | Registration Number |
|---:|---|---|---|
| 1 | Bikkam Nikhi Reddy | CSE-D | `CB.SC.U4CSE24309` |
| 2 | Kattumuri Santosh | CSE-D | `CB.SC.U4CSE24326` |
| 3 | Shaik Moyeen | CSE-D | `CB.SC.U4CSE24347` |

## Review Scope

The repository contains supervised-learning work for classification and regression, along with an unsupervised clustering notebook. The notebooks document data auditing, exploratory analysis, preprocessing, model comparison, evaluation, and visualizations; the regression notebook also includes tuning and cross-validation. Make sure the scope and results you present match the current committed notebooks and your course review requirements.

## Reproducibility and Data Notes

- Run notebook cells in order so that variables are defined before use.
- Keep the dataset filenames and paths consistent with the repository structure.
- Scores can change if the datasets, preprocessing, random seeds, or model settings change.
- Verify dataset sources, terms of use, and licenses before publishing raw data.
- Do not commit passwords, API keys, personal information, virtual environments, or unrelated generated files.
- Add a `LICENSE` file only after selecting a suitable license and confirming that dataset terms allow the intended use.
