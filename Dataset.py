import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# GLOBAL PLOT SETTINGS
# -------------------------------

plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["figure.dpi"] = 120

# -------------------------------
# 1. Load Dataset
# -------------------------------

df = pd.read_csv(r"C:\Users\2006r\Downloads\Dataset.csv")

print("\nShape:", df.shape)
print(df.head())

print("\nInfo:")
df.info()

print("\nSummary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# 2. Data Cleaning
# -------------------------------

df = df.dropna()

df["Date time"] = pd.to_datetime(df["Date time"])
df = df.sort_values("Date time")

df["Year"] = df["Date time"].dt.year
df["Month"] = df["Date time"].dt.month

# -------------------------------
# 3. Feature Engineering
# -------------------------------

df["MA7"] = df["Temperature"].rolling(7).mean()
df["ES"] = df["Temperature"].ewm(alpha=0.2).mean()
df["Volatility"] = df["Temperature"].rolling(30).std()

df["Z"] = (df["Temperature"] - df["Temperature"].mean()) / df["Temperature"].std()
df["Anomaly"] = df["Z"].abs() > 2

# -------------------------------
# 4. BAR PLOTS 
# -------------------------------

# Yearly Average Temperature
yearly_avg = df.groupby("Year")["Temperature"].mean()

plt.figure(figsize=(10,6))
yearly_avg.plot(kind='bar')

plt.title("Average Temperature per Year", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# Monthly Average Temperature
monthly_avg = df.groupby("Month")["Temperature"].mean()

plt.figure(figsize=(10,6))
monthly_avg.plot(kind='bar')

plt.title("Average Temperature per Month", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Correlation Heatmap
# -------------------------------

plt.figure(figsize=(12,10))

sns.heatmap(
    df.select_dtypes(include=['float64','int64']).corr(),
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Temperature Range Plot
# -------------------------------

plt.figure(figsize=(18,8))

plt.plot(df["Date time"], df["Temperature"], label="Avg Temp")

plt.fill_between(
    df["Date time"],
    df["Minimum Temperature"],
    df["Maximum Temperature"],
    alpha=0.2,
    label="Min-Max Range"
)

plt.title("Temperature Range Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Analytical Time Series
# -------------------------------

plt.figure(figsize=(18,8))

plt.plot(df["Date time"], df["Temperature"], label="Raw", alpha=0.4)
plt.plot(df["Date time"], df["MA7"], label="Trend (MA7)")
plt.plot(df["Date time"], df["ES"], label="Exp Smoothing")

plt.scatter(
    df["Date time"][df["Anomaly"]],
    df["Temperature"][df["Anomaly"]],
    label="Anomalies"
)

plt.title("Analytical Temperature View", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 8. Temperature vs Humidity
# -------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(x=df["Temperature"], y=df["Relative Humidity"])

plt.title("Temperature vs Humidity", fontsize=16)
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.tight_layout()
plt.show()

# -------------------------------
# 9. Heat Index vs Temperature
# -------------------------------

plt.figure(figsize=(18,8))

plt.plot(df["Date time"], df["Temperature"], label="Actual")
plt.plot(df["Date time"], df["Heat Index"], label="Feels Like")

plt.title("Actual vs Feels Like Temperature", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 10. Dew Point vs Temperature
# -------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(x=df["Temperature"], y=df["Dew Point"])

plt.title("Dew Point vs Temperature", fontsize=16)
plt.xlabel("Temperature")
plt.ylabel("Dew Point")
plt.tight_layout()
plt.show()

# -------------------------------
# 11. Wind Speed Over Time
# -------------------------------

plt.figure(figsize=(18,6))

plt.plot(df["Date time"], df["Wind Speed"])

plt.title("Wind Speed Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Wind Speed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 12. Wind Direction Distribution (Histogram)
# -------------------------------

plt.figure(figsize=(10,6))

sns.histplot(df["Wind Direction"], bins=36)

plt.title("Wind Direction Distribution", fontsize=16)
plt.xlabel("Direction (degrees)")
plt.tight_layout()
plt.show()

# -------------------------------
# 13. Monthly Temperature Boxplot
# -------------------------------

plt.figure(figsize=(12,6))

sns.boxplot(x="Month", y="Temperature", data=df)

plt.title("Monthly Temperature Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# -------------------------------
# 14. Multi-variable Trend
# -------------------------------

plt.figure(figsize=(18,8))

plt.plot(df["Date time"], df["Temperature"], label="Temp")
plt.plot(df["Date time"], df["Relative Humidity"], label="Humidity")
plt.plot(df["Date time"], df["Wind Speed"], label="Wind")

plt.title("Multi-variable Weather Trends", fontsize=16)
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 15. Save Processed Data
# -------------------------------

df.to_csv("processed_weather_data.csv", index=False)

print("\nAnalysis complete & file saved!")