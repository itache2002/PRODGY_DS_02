import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('train.csv')
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Cabin'].fillna('C123',inplace=True)
data['Embarked'].fillna('Q',inplace=True)
print(data.isnull().sum())  
print(data.duplicated().sum())

# 1. Histogram of Ages
sns.histplot(data['Age'].dropna(), kde=True)  # Dropping NaN values for the histogram
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Bar plot of Survived
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 3. Correlation Heatmap
# Select only numeric columns for the correlation heatmap
numeric_data = data.select_dtypes(include=[float, int])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()