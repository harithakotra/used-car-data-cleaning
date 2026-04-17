# Explore and Identify issues
import pandas as pd
# Load dataset
df = pd.read_csv("used_cars_messy.csv")
# Basic exploration
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())
print("\nSample data:")
df.head()

# Clean the data
df = df.dropna(subset=['selling_price'])

# Fill numeric columns with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df['brand'] = df['brand'].str.strip().str.lower()

df['mileage'] = df['mileage'].str.extract('(\d+\.?\d*)').astype(float)

# Drop duplicates
df = df.drop_duplicates()

print("Cleaned Shape:", df.shape)
print(df.isnull().sum())

# Compute Baseline MAE
from sklearn.metrics import mean_absolute_error

# Mean of target
mean_price = df['selling_price'].mean()

# Create predictions
y_true = df['selling_price']
y_pred = [mean_price] * len(df)

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)

print("Baseline Mean Price:", mean_price)
print("Baseline MAE:", mae)
