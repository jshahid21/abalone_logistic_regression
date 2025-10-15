# Step 1 - Import all required binaries for applying logistic regression to abalone data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# Step 2 - Read data into a data frame
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df = pd.read_csv('/content/abalone.data', names=names, sep=',')

df.head()

# Step 3 - Preprocess the data
df.dtypes

df.isna().sum()

df = df.drop('Sex', axis=1)

df.columns

# Step 4 - Define X and y
X = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight']]
y = df['Rings']

# Step 5 - Split X and y into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)

# Step 6 - Create an object of the algorithm
lg = LogisticRegression(max_iter=300)

# Step 7 - Fit the data to the object
lg.fit(X_train, y_train)

lg.score(X_train,y_train)

# Step 8 - Evaluate the model
print(f"Training Accuracy: {lg.score(X_train,y_train):.4f}")
print("\nModel Coefficients:")
print(lg.coef_)
print("\nModel Intercept:")
print(lg.intercept_)
