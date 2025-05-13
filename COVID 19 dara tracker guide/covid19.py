import pandas as pd

df = pd.read_csv('owid-covid-data.csv')
countries = ['Kenya', 'India', 'United States']
df = df[df['location'].isin(countries)]

df = df.drop(columns=['SNo', 'Last_Updated_Time'], errors='ignore')
print(df.columns)
print(df.head())
print(df.isnull().sum())
# 2. Drop rows with missing date or other critical values (like total_cases, total_deaths)
df = df.dropna(subset=['date', 'total_cases', 'total_deaths'])

# 3. Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# 4. Handle missing numeric values with forward-fill, backfill, or interpolation
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# Preview cleaned data
print(df.head())

# Get the latest date's data for each country
latest_data = df.sort_values('date').groupby('location').tail(1)

# Sort by total confirmed cases
top_cases = latest_data.sort_values(by='total_cases', ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns
# Plot
plt.figure(figsize=(10, 6))
plt.bar(top_cases['location'], top_cases['total_cases'], color='skyblue')
plt.xticks(rotation=45)
plt.title('Total Confirmed COVID-19 Cases (Latest Data)')
plt.ylabel('Confirmed Cases')
plt.tight_layout()
plt.show()


# Calculate death rate
df['death_rate'] = df['total_deaths'] / df['total_cases']

# Set up plotting style
sns.set(style="whitegrid")

# Plot 1: Total cases over time
plt.figure(figsize=(12, 6))
for country in countries:
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)
plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Total deaths over time
plt.figure(figsize=(12, 6))
for country in countries:
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['total_deaths'], label=country)
plt.title('Total COVID-19 Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: Daily new cases
plt.figure(figsize=(12, 6))
for country in countries:
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases'], label=country)
plt.title('Daily New COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[['total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'death_rate']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# --- 1. Line Chart: Cumulative Vaccinations Over Time ---
plt.figure(figsize=(12, 6))
for country in countries:
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)
plt.title('Cumulative COVID-19 Vaccinations Over Time')
plt.xlabel('Date')
plt.ylabel('Total Vaccinations')
plt.legend()
plt.tight_layout()
plt.show()

# --- 2. Bar Chart: % of Population Fully Vaccinated ---
# Get most recent data per country
latest = df.sort_values('date').groupby('location').tail(1)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=latest, x='location', y='people_fully_vaccinated_per_hundred', palette='Greens')
plt.title('Percentage of Population Fully Vaccinated')
plt.ylabel('% Fully Vaccinated')
plt.xlabel('Country')
plt.tight_layout()
plt.show()

# --- 3. (Optional) Pie Charts: Vaccinated vs. Unvaccinated ---
for country in countries:
    data = latest[latest['location'] == country]
    vaccinated = data['people_fully_vaccinated_per_hundred'].values[0]
    unvaccinated = 100 - vaccinated
    plt.figure(figsize=(5, 5))
    plt.pie([vaccinated, unvaccinated],
            labels=['Vaccinated', 'Unvaccinated'],
            colors=['green', 'lightgray'],
            autopct='%1.1f%%',
            startangle=140)
    plt.title(f'{country}: Vaccinated vs. Unvaccinated')
    plt.tight_layout()
    plt.show()

    # Keep only latest data per country
df = df[df['continent'].notna()]  # Filter out aggregates like 'World'
df['date'] = pd.to_datetime(df['date'])
latest_df = df.sort_values('date').groupby('location').tail(1)

import plotly.express as px

# Prepare data for choropleth
choropleth_df = latest_df[['iso_code', 'location', 'total_cases', 'total_cases_per_million',
                           'people_fully_vaccinated_per_hundred']].copy()

# Fill NaNs with 0
choropleth_df.fillna(0, inplace=True)

# --- 1. Choropleth: Total COVID-19 Cases ---
fig1 = px.choropleth(
    choropleth_df,
    locations="iso_code",
    color="total_cases",
    hover_name="location",
    color_continuous_scale="Reds",
    title="Total Confirmed COVID-19 Cases (Latest)"
)
fig1.write_html("choropleth_total_cases.html")


# --- 2. Choropleth: % Fully Vaccinated ---
fig2 = px.choropleth(
    choropleth_df,
    locations="iso_code",
    color="people_fully_vaccinated_per_hundred",
    hover_name="location",
    color_continuous_scale="Greens",
    title="Percentage of Population Fully Vaccinated"
)