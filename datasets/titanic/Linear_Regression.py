import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('train.csv')
# Drop rows with missing values in the 'Age' column
titanic_df_age = titanic_df.dropna(subset=['Age'])

# Define features (X) and target variable (y)
X = titanic_df_age[['Pclass', 'SibSp', 'Parch', 'Fare']]
y = titanic_df_age['Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)

# Calculate Root Mean Squared (RMS)
rms = np.sqrt(np.mean(np.square(y_test - y_pred)))
print("Root Mean Squared:", rms)

# Plot actual vs. predicted age
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs. Predicted Age (Linear Regression)')
plt.show()
