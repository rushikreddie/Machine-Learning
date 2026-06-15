<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=200&section=header&text=Machine%20Learning&fontSize=60&fontColor=ffffff&fontAlignY=35&desc=A%20complete%20hands-on%20ML%20notebook%20collection&descAlignY=55&descSize=18&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)

<br/>

> **A structured collection of Jupyter notebooks** covering the full spectrum of classical machine learning —  
> from fitting your first line to stacking production-ready ensembles.

<br/>

[🚀 Get Started](#-quick-start) • [📚 Modules](#-modules) • [🧠 Algorithms](#-algorithms-covered) • [🛠 Setup](#%EF%B8%8F-installation) • [🤝 Contribute](#-contributing)

</div>

---

## 📁 Repository Structure

```
Machine-Learning/
│
├── 📈  1_Supervised_Learning_Regression/
│       └── Linear, Polynomial, Ridge, Lasso, Gradient Descent
│
├── 🔵  2_Supervised_Learning_Classification/
│       └── Logistic Regression, KNN, SVM, Naive Bayes, Decision Trees
│
├── 🌲  3_Ensemble_Learning/
│       └── Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking
│
├── 📊  4_model_Evaluation/
│       └── Cross-Validation, GridSearchCV, ROC-AUC, Learning Curves
│
└── ⚙️  5_Feature_engineering/
        └── Encoding, Scaling, Imputation, Pipelines, Feature Selection
```

---

## 📚 Modules

<table>
<tr>
<td width="50px" align="center">📈</td>
<td><strong>Module 01 — Supervised Learning: Regression</strong><br/>
Predict continuous outcomes. Covers linear and polynomial regression, gradient descent, cost functions, L1/L2 regularization, and metrics like RMSE and R².</td>
</tr>
<tr><td colspan="2"><code>LinearRegression</code> &nbsp; <code>PolynomialFeatures</code> &nbsp; <code>Ridge</code> &nbsp; <code>Lasso</code> &nbsp; <code>gradient-descent</code></td></tr>

<tr><td colspan="2"><br/></td></tr>

<tr>
<td width="50px" align="center">🔵</td>
<td><strong>Module 02 — Supervised Learning: Classification</strong><br/>
Classify data into discrete labels. Logistic regression, KNN, SVM, Naive Bayes, and decision trees — with confusion matrices, ROC-AUC, precision, recall.</td>
</tr>
<tr><td colspan="2"><code>LogisticRegression</code> &nbsp; <code>SVC</code> &nbsp; <code>KNeighborsClassifier</code> &nbsp; <code>DecisionTreeClassifier</code> &nbsp; <code>GaussianNB</code></td></tr>

<tr><td colspan="2"><br/></td></tr>

<tr>
<td width="50px" align="center">🌲</td>
<td><strong>Module 03 — Ensemble Learning</strong><br/>
Combine weak learners into powerful predictors. Bagging, boosting, stacking — with detailed comparisons of Random Forest, XGBoost, and AdaBoost.</td>
</tr>
<tr><td colspan="2"><code>RandomForestClassifier</code> &nbsp; <code>XGBClassifier</code> &nbsp; <code>GradientBoostingClassifier</code> &nbsp; <code>AdaBoostClassifier</code> &nbsp; <code>VotingClassifier</code></td></tr>

<tr><td colspan="2"><br/></td></tr>

<tr>
<td width="50px" align="center">📊</td>
<td><strong>Module 04 — Model Evaluation</strong><br/>
Know if your model actually works. Implements cross-validation, bias-variance analysis, learning curves, and hyperparameter tuning pipelines.</td>
</tr>
<tr><td colspan="2"><code>cross_val_score</code> &nbsp; <code>GridSearchCV</code> &nbsp; <code>RandomizedSearchCV</code> &nbsp; <code>learning_curve</code> &nbsp; <code>StratifiedKFold</code></td></tr>

<tr><td colspan="2"><br/></td></tr>

<tr>
<td width="50px" align="center">⚙️</td>
<td><strong>Module 05 — Feature Engineering</strong><br/>
Turn raw data into model fuel. One-hot encoding, imputation, scaling strategies, feature selection, and sklearn pipelines for production-ready preprocessing.</td>
</tr>
<tr><td colspan="2"><code>OneHotEncoder</code> &nbsp; <code>StandardScaler</code> &nbsp; <code>Pipeline</code> &nbsp; <code>SimpleImputer</code> &nbsp; <code>SelectKBest</code></td></tr>
</table>

---

## 🧠 Algorithms Covered

| Category | Algorithms |
|---|---|
| **Regression** | Linear · Polynomial · Ridge · Lasso · ElasticNet |
| **Classification** | Logistic Regression · KNN · SVM · Naive Bayes · Decision Tree |
| **Ensemble** | Random Forest · AdaBoost · Gradient Boosting · XGBoost · Stacking |
| **Evaluation** | K-Fold CV · Stratified CV · GridSearchCV · RandomizedSearchCV |
| **Preprocessing** | StandardScaler · MinMaxScaler · RobustScaler · OneHotEncoder · LabelEncoder |

---

## 🗝 Key Concepts

| Concept | What you'll learn |
|---|---|
| **Gradient Descent** | Iterative optimization to minimize loss by updating weights step by step |
| **Bias-Variance Tradeoff** | The balance between underfitting and overfitting, visualized via learning curves |
| **Regularization** | L1 (Lasso) and L2 (Ridge) penalties that prevent overfitting |
| **ROC-AUC** | Measuring classifier quality across all decision thresholds |
| **Bagging vs Boosting** | Parallel vs sequential ensemble strategies — when to use which |
| **Feature Scaling** | Why distance-based algorithms break without it and how to fix them |
| **Sklearn Pipelines** | Chaining preprocessing + modeling into leak-free production workflows |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Basic knowledge of NumPy & Pandas
- Familiarity with Jupyter Notebook

### Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/rushikreddie/Machine-Learning.git
cd Machine-Learning
```

**2. Create a virtual environment**
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

**4. Launch Jupyter**
```bash
jupyter notebook
```

---

## 🚀 Quick Start

Follow the modules in order for the best learning experience:

```
01 → Regression        ▸  Master the fundamentals
02 → Classification    ▸  Extend to categorical outputs
03 → Ensemble          ▸  Combine models for power
04 → Evaluation        ▸  Validate and tune rigorously
05 → Feature Eng.      ▸  Build production-ready pipelines
```

---

## 🤝 Contributing

Contributions are welcome! If you find a bug, want to add a notebook, or improve an explanation:

1. **Fork** the repository
2. **Create** a branch: `git checkout -b feature/your-idea`
3. **Commit** your changes: `git commit -m "Add: your description"`
4. **Push** and open a **Pull Request**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=100&section=footer" width="100%"/>

**Made with 💜 by [rushikreddie](https://github.com/rushikreddie)**

⭐ Star this repo if it helped you learn!

</div>
