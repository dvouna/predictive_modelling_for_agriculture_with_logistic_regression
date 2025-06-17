# Load necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns 


# Load the dataset
crops = pd.read_csv("soil_measures.csv")
crops.head()


# Display dataset information
crops.info()


# Check missing values 
print(crops.isna().sum())


# Count the number of unique crops
print(crops['crop'].nunique())


# Count the occurrences of each crop
print(crops['crop'].value_counts())


# Visualize the relationship between nitrogen (N), potassium (K), and crop type
sns.relplot(x='N', y='K', data=crops, kind='scatter', hue='crop')
plt.show()


# Prepare the data for logistic regression
X = crops.drop(columns='crop')
y = crops['crop'] 


# Splitting the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state= 27) 


# Initialize a dictionary to store feature performance
feature_performance = {}


# Run logistic regression for each feature and calculate F1 score
for feature in ["N", "P", "K", "ph"]: 
    log_reg = LogisticRegression(multi_class='multinomial')
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]]) 
    
    # Calculating F1 score, 
    f1 = metrics.f1_score(y_test, y_pred, average='weighted') 
    
    # Add feature-f1 score paries to the dict 
    feature_performance[feature] = f1 
    print(f'F1-score for {feature}: {f1}') 


# K produces the best F1 score 
best_predictive_feature = {'K': feature_performance['K']}
best_predictive_feature 