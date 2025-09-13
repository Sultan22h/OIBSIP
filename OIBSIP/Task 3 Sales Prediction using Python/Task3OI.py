# Task 3: Sales Prediction using Python By Sultan Hussain

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Data
# -------------------------------
data = pd.read_excel(r"C:\Users\DELL\Desktop\OIBSIP\Task 3 Sales Prediction using Python\Advertising.xlsx")

print("Data Preview:\n", data.head())

# Drop SR.NO if exists
if 'SR.NO' in data.columns:
    data.drop(columns=['SR.NO'], inplace=True)

# -------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------
# Pairplot to see relationships
sns.pairplot(data)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 3. Prepare Data for Modeling
# -------------------------------
X = data[['TV', 'Radio', 'Newspaper']]  # features
y = data['Sales']                        # target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 4. Train Model (Linear Regression)
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:\n", comparison.head())

# -------------------------------
# 6. Visualize Predictions
# -------------------------------
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.show()

# -------------------------------
# 7. Test with Custom Input
# -------------------------------
def predict_sales(tv, radio, newspaper):
    return model.predict([[tv, radio, newspaper]])[0]

# Example
print("\nTest Prediction:")
print("Sales prediction for TV=150, Radio=40, Newspaper=30:")
print("Predicted Sales:", predict_sales(150, 40, 30))
