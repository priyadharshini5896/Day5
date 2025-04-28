import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Basic Data Overview
print("\n--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Dataset Description ---")
print(df.describe())
print("\n--- Value Counts (Survived Column) ---")
print(df['Survived'].value_counts())

# Missing Values Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Histograms
plt.figure(figsize=(10, 6))
df['Age'].hist(bins=30, color='teal', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Countplot for Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Boxplot: Age vs Pclass
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age vs Passenger Class')
plt.show()

# Scatterplot: Fare vs Age
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Fare vs Age (Colored by Survival)')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['number'])  
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(df.dropna(), hue='Survived', vars=['Age', 'Fare', 'Pclass'])
plt.suptitle('Pairplot of Features Colored by Survival', y=1.02)
plt.show()

print("""
Summary of Key Findings:
- Females had a much higher survival rate compared to males.
- Passengers from higher classes (Pclass = 1) had higher survival rates.
- Younger passengers had slightly higher survival rates.
- Higher fare paid was loosely associated with higher survival chances.
- Missing values mostly in 'Age' and 'Cabin'; Embarked has very few missing entries.
- Sex and Pclass are strong indicators of survival.
""")
