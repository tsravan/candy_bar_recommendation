import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Reading the data
data = pd.read_csv("candy-data.csv")
data.columns

# null check
data.isna().sum()

# Subsetting the data
data_t = data.iloc[:,1:]
data_t.columns

# Correlation
plt.figure(figsize=(25, 20))
sns.set(font_scale=1)
sns.heatmap(data_t.corr(), annot=True)
plt.show()

# Preparing the training data
x = data.iloc[:,1:-1]
y = data['winpercent']

# Traning Regression Model
mlr = LinearRegression()  
mlr_model = mlr.fit(x, y)

# Intercept and model coefficients
print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))

# Display feature coefficients with feature names
feature_names = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 
                 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent']

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficients': mlr.coef_})
coef_df = coef_df.sort_values(by='Coefficients', ascending=False)


# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(coef_df['Feature'], coef_df['Coefficients'])
plt.xlabel('Features')
plt.ylabel('Weightage/coefficients')
plt.xticks(rotation=45)
plt.show()