import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("student_data.csv")

print("Data:")
print(df)

# Visualization
plt.scatter(df["Hours"], df["Marks"])
plt.title("Hours vs Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()

# Features & Target
X = df[["Hours"]]
y = df["Marks"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization - Regression line
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Linear Regression Line")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.show()

# Test custom input
hours = [[7.5]]
predicted = model.predict(hours)

print("\nPredicted Marks for 7.5 hours:", predicted[0])