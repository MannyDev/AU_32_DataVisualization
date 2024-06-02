import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

insuranceData = pd.read_csv("C:/Users/manue/Documents/University/Courses/3-2 Data Visualization/AU_32_DataVisualization/datasets/insurance.csv")

bins = [18, 36, 55, 12000]
labels = ['youth', 'middleAge', 'elder']
insuranceData['age_cat'] = pd.cut(insuranceData['age'], bins=bins, labels=labels, right=False)

# Correlation Matrix
correlation = insuranceData[['age', 'bmi', 'children', 'charges']].corr()
sns.heatmap(correlation, annot = True, fmt = '.2f', cmap = 'Blues')
plt.title("Correlation between numerical values")
plt.show()

# Regression Model
X = dataset[['Age', 'YearsExperience', 'Gender', 'Classification', 'Job']]
X= pd.get_dummies (data=X, drop_first=True)
Y = dataset[['Salary']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
predictionsTable = pd.DataFrame(predictions)
results = { 'MAE': mean_absolute_error(y_test, predictions),                      
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions))}
dataset = pd.DataFrame (results, index=[0])

