# sales-demand-forecasting
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("sales_data.csv")

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Create numerical feature
df['Day_Number'] = range(len(df))

X = df[['Day_Number']]
y = df['Sales']
# Train model
model = LinearRegression()
model.fit(X, y)

# Forecast next 7 days
future = pd.DataFrame({
    'Day_Number': range(len(df), len(df) + 7)
})

forecast = model.predict(future)

# Plot
plt.plot(df['Day_Number'], y, label='Actual Sales')
plt.plot(future['Day_Number'], forecast, label='Forecast')
plt.legend()
plt.title("Sales Forecasting")
plt.show()
