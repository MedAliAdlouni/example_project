import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data (e.g., a small dataset for linear regression)
data = {'Hours_Studied': [1, 2, 3, 4, 5],
        'Test_Score': [50, 55, 65, 70, 80]}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the data
plt.scatter(df['Hours_Studied'], df['Test_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Test Score vs Hours Studied')
plt.show()

# Prepare data for model training
X = df[['Hours_Studied']]
y = df['Test_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")

# Show the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Regression Line')
plt.show()
