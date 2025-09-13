# Unemployment Analysis in India By Sultan Hussain

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------------------------------
# 1. Load Data
# -------------------------------
india_data = pd.read_excel(
    r"C:\Users\DELL\Desktop\OIBSIP\Task 1 Unemployment_Analysis_Python\Unemployment in India.xlsx"
)
covid_data = pd.read_excel(
    r"C:\Users\DELL\Desktop\OIBSIP\Task 1 Unemployment_Analysis_Python\Unemployment_Rate_upto_11_2020.xlsx"
)

# Remove leading/trailing spaces from column names (important!)
india_data.columns = india_data.columns.str.strip()
covid_data.columns = covid_data.columns.str.strip()

print("India Data Preview:\n", india_data.head())
print("Covid Data Preview:\n", covid_data.head())

# -------------------------------
# 2. Data Cleaning
# -------------------------------
if "Date" in india_data.columns:
    india_data["Date"] = pd.to_datetime(india_data["Date"], errors="coerce")
else:
    print("⚠️ 'Date' column not found in India dataset")

if "Date" in covid_data.columns:
    covid_data["Date"] = pd.to_datetime(covid_data["Date"], errors="coerce")
else:
    print("⚠️ 'Date' column not found in Covid dataset")

india_data.rename(
    columns={
        "Estimated Unemployment Rate (%)": "UnemploymentRate",
        "Estimated Employed": "Employed",
        "Estimated Labour Participation Rate (%)": "LabourParticipation",
    },
    inplace=True,
)

covid_data.rename(
    columns={
        "Estimated Unemployment Rate (%)": "UnemploymentRate",
        "Estimated Employed": "Employed",
        "Estimated Labour Participation Rate (%)": "LabourParticipation",
    },
    inplace=True,
)

# -------------------------------
# 3. Visualizations
# -------------------------------

# Trend of unemployment by region (India dataset)
plt.figure(figsize=(12, 6))
sns.lineplot(data=india_data, x="Date", y="UnemploymentRate", hue="Region")
plt.title("Unemployment Rate Trend in India (by Region)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Rural vs Urban unemployment
plt.figure(figsize=(8, 5))
sns.barplot(data=india_data, x="Area", y="UnemploymentRate", ci=None)
plt.title("Rural vs Urban Unemployment Rate in India")
plt.tight_layout()
plt.show()

# Geo scatter map (Covid dataset)
if "longitude" in covid_data.columns and "latitude" in covid_data.columns:
    fig = px.scatter_geo(
        covid_data,
        lon="longitude",
        lat="latitude",
        color="UnemploymentRate",
        size="UnemploymentRate",
        hover_name="Region",
        animation_frame=covid_data["Date"].dt.strftime("%Y-%m"),
        title="Unemployment Rate During Covid-19",
    )
    fig.show()
else:
    print("⚠️ Covid dataset missing longitude/latitude columns")
