# 🚢 Titanic Survival Prediction

An end-to-end machine learning project that predicts passenger survival on the Titanic using the classic Kaggle dataset. Includes full exploratory data analysis, feature engineering, and a multi-model comparison pipeline.

## 📌 Overview

This project walks through the complete ML workflow:
1. **Exploratory Data Analysis (EDA)** — understand the data and visualize survival patterns
2. **Feature Engineering** — clean and enrich the raw data
3. **Model Training & Comparison** — train multiple classifiers and pick the best one
4. **Prediction & Submission** — generate a Kaggle-ready `submission.csv`

## 📂 Repository Structure

```
├── train.csv                  # Training data (with 'Survived' labels)
├── test.csv                   # Test data (labels to be predicted)
├── titanic_eda.py             # Exploratory data analysis + visualizations
├── titanic_model.py           # Feature engineering + model training/prediction
├── titanic_eda.png            # 9-panel EDA dashboard (saved output)
├── titanic_correlation.png    # Feature correlation heatmap (saved output)
├── submission.csv             # Final predictions (PassengerId, Survived)
```

## 🔍 Exploratory Data Analysis (`titanic_eda.py`)

Generates a 9-panel dashboard and a correlation heatmap covering:
- Overall survival distribution
- Passenger class vs. survival
- Gender vs. survival
- Age distribution and age vs. survival
- Fare distribution
- Embarkation port vs. survival
- Siblings/spouses (SibSp) vs. survival
- Parents/children (Parch) vs. survival
- Correlation of all numeric features with survival

**Key findings:**
- Female passengers survived at a much higher rate than male passengers
- 1st class passengers had a notably higher survival rate than 3rd class
- Survivors paid a higher average fare (~$50) than non-survivors (~$24)
- Younger passengers had a slightly better chance of survival

## ⚙️ Feature Engineering & Modeling (`titanic_model.py`)

**Preprocessing:**
- Standardized column names, ensured `PassengerId` exists in the test set
- Encoded `sex` (male/female → 1/0) and `embarked` (S/C/Q → 0/1/2)
- Filled missing `age` and `fare` values with the median

**Engineered features:**
- `family_size` = siblings/spouses + parents/children + 1
- `is_alone` = whether the passenger was traveling solo
- `age_bin` = binned age groups (child, teen, adult, middle-aged, senior)
- `fare_bin` = fare quartiles

**Final feature set:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `family_size`, `is_alone`, `age_bin`, `fare_bin`

**Models trained & compared** (on an 80/20 stratified train/validation split, features scaled with `StandardScaler`):
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Each model is evaluated on **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**, and the model with the best **F1-Score** on the validation set is automatically selected to generate the final predictions.

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/golekhushal-cmd/Titanic-surviva-prediction.git
cd Titanic-surviva-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the EDA script (produces titanic_eda.png and titanic_correlation.png)
python titanic_eda.py

# Run the model training + prediction script (produces submission.csv)
python titanic_model.py
```

## 🛠️ Tech Stack

- **Python 3**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — preprocessing, modeling, evaluation

## 📈 Results

The pipeline prints a full metrics comparison (Accuracy, Precision, Recall, F1, ROC-AUC) for all three models and reports the best-performing one at runtime. See the terminal output when running `titanic_model.py` for exact scores on your machine.

## 🔮 Possible Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation instead of a single train/validation split
- Extracting titles (Mr., Mrs., Miss, etc.) from the `Name` column as a feature
- Trying additional models (XGBoost, SVM, ensembling/stacking)

## 👤 Author

**Khushal Gole**
[GitHub](https://github.com/golekhushal-cmd)

---
*This is an independent minor project built for learning purposes, using the well-known Titanic dataset from Kaggle.*
