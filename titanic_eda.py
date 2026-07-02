import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Make column names consistent with the files in this folder
train.columns = [col.strip().lower() for col in train.columns]
test.columns = [col.strip().lower() for col in test.columns]

# ============= BASIC INFO =============
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}\n")

print("First few rows:")
print(train.head())

print("\nDataset Info:")
print(train.info())

# ============= MISSING VALUES =============
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = train.isnull().sum()
missing_pct = (train.isnull().sum() / len(train)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing_Count'] > 0])

# ============= STATISTICAL SUMMARY =============
print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(train.describe())

# ============= SURVIVAL DISTRIBUTION =============
print("\n" + "=" * 50)
print("SURVIVAL STATISTICS")
print("=" * 50)
print(train['survived'].value_counts())
print(f"Survival rate: {train['survived'].mean():.2%}")

# ============= VISUALIZATIONS =============
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

# 1. Survival Distribution
ax1 = plt.subplot(3, 3, 1)
train['survived'].value_counts().plot(kind='bar', ax=ax1, color=['red', 'green'])
ax1.set_title('Survival Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Survived')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['No', 'Yes'], rotation=0)

# 2. Passenger Class vs Survival
ax2 = plt.subplot(3, 3, 2)
pd.crosstab(train['pclass'], train['survived']).plot(kind='bar', ax=ax2)
ax2.set_title('Passenger Class vs Survival', fontsize=12, fontweight='bold')
ax2.set_xlabel('Passenger Class')
ax2.set_ylabel('Count')
ax2.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
ax2.legend(['Did not survive', 'Survived'])

# 3. Gender vs Survival
ax3 = plt.subplot(3, 3, 3)
pd.crosstab(train['sex'], train['survived']).plot(kind='bar', ax=ax3)
ax3.set_title('Gender vs Survival', fontsize=12, fontweight='bold')
ax3.set_xlabel('Gender')
ax3.set_ylabel('Count')
ax3.set_xticklabels(['Female', 'Male'], rotation=0)
ax3.legend(['Did not survive', 'Survived'])

# 4. Age Distribution
ax4 = plt.subplot(3, 3, 4)
train['age'].hist(bins=30, ax=ax4, color='skyblue', edgecolor='black')
ax4.set_title('Age Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')

# 5. Age vs Survival
ax5 = plt.subplot(3, 3, 5)
train[train['survived'] == 0]['age'].hist(bins=30, ax=ax5, alpha=0.5, label='Did not survive', color='red')
train[train['survived'] == 1]['age'].hist(bins=30, ax=ax5, alpha=0.5, label='Survived', color='green')
ax5.set_title('Age vs Survival', fontsize=12, fontweight='bold')
ax5.set_xlabel('Age')
ax5.set_ylabel('Count')
ax5.legend()

# 6. Fare Distribution
ax6 = plt.subplot(3, 3, 6)
train['fare'].hist(bins=30, ax=ax6, color='skyblue', edgecolor='black')
ax6.set_title('Fare Distribution', fontsize=12, fontweight='bold')
ax6.set_xlabel('Fare')
ax6.set_ylabel('Count')

# 7. Embarked Port vs Survival
ax7 = plt.subplot(3, 3, 7)
pd.crosstab(train['embarked'], train['survived']).plot(kind='bar', ax=ax7)
ax7.set_title('Embarked Port vs Survival', fontsize=12, fontweight='bold')
ax7.set_xlabel('Port of Embarkation')
ax7.set_ylabel('Count')
ax7.legend(['Did not survive', 'Survived'])

# 8. SibSp vs Survival
ax8 = plt.subplot(3, 3, 8)
pd.crosstab(train['sibsp'], train['survived']).plot(kind='bar', ax=ax8)
ax8.set_title('Siblings/Spouses vs Survival', fontsize=12, fontweight='bold')
ax8.set_xlabel('Number of Siblings/Spouses')
ax8.set_ylabel('Count')
ax8.legend(['Did not survive', 'Survived'])

# 9. Parch vs Survival
ax9 = plt.subplot(3, 3, 9)
pd.crosstab(train['parch'], train['survived']).plot(kind='bar', ax=ax9)
ax9.set_title('Parents/Children vs Survival', fontsize=12, fontweight='bold')
ax9.set_xlabel('Number of Parents/Children')
ax9.set_ylabel('Count')
ax9.legend(['Did not survive', 'Survived'])

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=100, bbox_inches='tight')
print("\nOK: Visualizations saved as 'titanic_eda.png'")
plt.show()

# ============= CORRELATION ANALYSIS =============
print("\n" + "=" * 50)
print("FEATURE CORRELATIONS WITH SURVIVAL")
print("=" * 50)

# Prepare data for correlation (numeric only)
numeric_train = train.select_dtypes(include=[np.number]).copy()
correlations = numeric_train.corr()['survived'].sort_values(ascending=False)
print(correlations)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_train.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('titanic_correlation.png', dpi=100, bbox_inches='tight')
print("\nOK: Correlation heatmap saved as 'titanic_correlation.png'")
plt.show()

# ============= KEY INSIGHTS =============
print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
print(f"1. Survival rate by class:")
print(train.groupby('pclass')['survived'].mean())
print(f"\n2. Survival rate by gender:")
print(train.groupby('sex')['survived'].mean())
print(f"\n3. Average age by survival:")
print(train.groupby('survived')['age'].mean())
print(f"\n4. Average fare by survival:")
print(train.groupby('survived')['fare'].mean())
