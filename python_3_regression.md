---
title: "3. REGRESSION"
layout: default
nav_order: 4
---
# **Regression**
This tutorial provides a comprehensive introduction to regression analysis, one of the most fundamental and widely used techniques in statistics and machine learning. We will cover the theoretical foundations, practical implementations in Python, and real-world ecological applications.

**Prerequisites:** Basic knowledge of Python, pandas, and numpy.

---

# **Chapter 1: Foundations of Regression**

## 1.1 What is Regression?

Regression analysis is a statistical method used to model and analyze the relationships between variables. At its core, regression helps us understand how one or more **independent variables** (also called predictors, features, or explanatory variables) influence a **dependent variable** (also called the target, response, or outcome variable).

The term "regression" was coined by Sir Francis Galton in the 19th century when he observed that children of tall parents tend to "regress" toward the average height of the population. Today, regression encompasses a broad family of techniques used across virtually every scientific discipline.

### An Ecological Example

Imagine you want to understand how temperature affects tree growth. You collect data from forest plots:

| Mean Annual Temperature (°C) | Tree Ring Width (mm) |
| ---------------------------- | -------------------- |
| 8                            | 1.2                  |
| 10                           | 1.8                  |
| 12                           | 2.4                  |
| 14                           | 2.9                  |
| 16                           | 3.2                  |

Regression analysis allows you to:

1. Find the mathematical relationship between temperature and tree growth
2. Predict growth rates at temperatures you haven't measured
3. Quantify how much each degree of warming contributes to growth

## 1.2 Purposes of Regression Analysis

Regression serves several key purposes in ecological data analysis:

### 1. Prediction

The most common use of regression is to predict unknown values of the target variable based on known values of the predictors. For example:

- Predicting forest biomass from satellite imagery indices
- Forecasting species distribution under climate change scenarios
- Estimating crop yield based on weather and soil conditions

### 2. Exploring Relationships

Regression helps us understand how environmental variables are related to ecological responses:

- Does precipitation affect plant species richness? By how much?
- Is there a relationship between water temperature and fish abundance?
- How do different soil nutrients contribute to crop productivity?

### 3. Inference and Hypothesis Testing

Regression allows us to test ecological hypotheses:

- Is the relationship between CO₂ concentration and photosynthesis rate statistically significant?
- Can we reject the null hypothesis that habitat fragmentation has no effect on biodiversity?

### 4. Identifying Important Environmental Drivers

When multiple factors could influence an ecological outcome, regression helps identify which ones matter most and which can be ignored.

### 5. Conservation and Management Decisions

Understanding relationships enables better environmental management:

- What combination of restoration actions maximizes ecosystem recovery?
- How should conservation resources be allocated to protect endangered species?

## 1.3 Key Concepts in Regression

### Target Variable (Dependent Variable)

The variable we want to predict or explain. Denoted as **y** or **Y**.

- Examples: species abundance, tree growth rate, carbon flux, water quality index, crop yield

### Independent Variables (Predictors/Features)

The variables used to predict or explain the target. Denoted as **X**, **x₁**, **x₂**, etc.

- Examples: temperature, precipitation, soil pH, elevation, canopy cover, nutrient concentration

### Model

A mathematical equation that describes the relationship between the predictors and the target. The general form is:

**y = f(X) + ε**

Where:

- **y** is the target variable
- **f(X)** is the systematic component (the ecological pattern we're trying to capture)
- **ε** (epsilon) is the error term (random variation we cannot explain)

### Parameters (Coefficients)

The values in the model that define the relationship. In a linear model **y = β₀ + β₁x**:

- **β₀** (beta-zero) is the **intercept** — the predicted value when x = 0
- **β₁** (beta-one) is the **slope** — how much y changes for each unit increase in x

### Residuals (Errors)

The difference between the actual observed values and the values predicted by the model:

**residual = y_observed - y_predicted**

Good models have small residuals that are randomly distributed.

### Fitted Values (Predictions)

The values predicted by the model, often denoted as **ŷ** (y-hat).

## 1.4 How Regression Works

The fundamental idea behind regression is to find the model parameters that best fit the observed data. Here's the general process:

### Step 1: Choose a Model Form

Decide on the mathematical form of the relationship:

- Simple: y = β₀ + β₁x (one predictor)
- Multiple predictors: y = β₀ + β₁x₁ + β₂x₂ + ...

### Step 2: Define "Best Fit"

Establish a criterion for what makes a good fit. The most common is **Least Squares**, which minimizes the sum of squared residuals:

**SSE = Σ(yᵢ - ŷᵢ)²**

We square the errors to:

- Prevent positive and negative errors from canceling out
- Penalize large errors more heavily than small ones

### Step 3: Estimate Parameters

Use mathematical optimization to find the parameter values that minimize the chosen criterion.

### Step 4: Evaluate the Model

Assess how well the model fits the data and whether it meets the assumptions of the method being used.

## 1.5 Evaluating Regression Models

Several metrics help us understand model performance:

### R-squared (R²) — Coefficient of Determination

Measures the proportion of variance in the target variable explained by the model:

**R² = 1 - (SSE / SST)**

Where:

- SSE = Sum of Squared Errors (residuals)
- SST = Total Sum of Squares (variance in y)

R² ranges from 0 to 1:

- R² = 1: Perfect fit (model explains all variance)
- R² = 0: Model explains none of the variance

In ecology, what constitutes a "good" R² depends heavily on the system. An R² of 0.4 might be excellent for predicting species distributions, while an R² of 0.9 might be expected for controlled growth experiments.

### Mean Squared Error (MSE)

Average of squared residuals:

**MSE = (1/n) Σ(yᵢ - ŷᵢ)²**

### Root Mean Squared Error (RMSE)

Square root of MSE, in the same units as the target:

**RMSE = √MSE**

RMSE is interpretable: "On average, predictions are off by approximately RMSE units."

### Mean Absolute Error (MAE)

Average of absolute residuals:

**MAE = (1/n) Σ|yᵢ - ŷᵢ|**

Less sensitive to outliers than RMSE — useful in ecological data where extreme values are common.

## 1.6 Types of Regression Approaches

In this tutorial, we will cover three main approaches:

| Approach                | Description                    | Best For                            |
| ----------------------- | ------------------------------ | ----------------------------------- |
| **Simple Regression**   | One predictor variable         | Understanding single relationships  |
| **Multiple Regression** | Multiple predictor variables   | Modeling complex ecological systems |
| **Machine Learning**    | Algorithms that learn patterns | Complex, non-linear relationships   |

---

# **Chapter 2: Simple Linear Regression**

Simple linear regression models the relationship between a single predictor and a target variable using a straight line.

## 2.1 The Model

The simple linear regression model takes the form:

**y = β₀ + β₁x + ε**

Where:

- y is the target variable (e.g., tree growth)
- x is the predictor variable (e.g., temperature)
- β₀ is the y-intercept
- β₁ is the slope
- ε is the error term

### Mathematical Foundation

The goal is to find the values of β₀ and β₁ that minimize the Sum of Squared Errors (SSE):

**SSE = Σ(yᵢ - β₀ - β₁xᵢ)²**

Taking partial derivatives and setting them to zero yields:

**β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²**

**β₀ = ȳ - β₁x̄**

## 2.2 Ecological Example: Tree Growth and Temperature

Let's examine how mean annual temperature affects tree ring width (a proxy for growth rate).

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Create ecological dataset: Tree growth vs temperature
np.random.seed(42)

# Temperature gradient (°C) - typical temperate forest range
temperature = np.array([6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Tree ring width (mm) - increases with temperature up to an optimum
# Adding realistic ecological variation
ring_width = 0.8 + 0.15 * temperature + np.random.normal(0, 0.3, len(temperature))

# Create DataFrame
tree_data = pd.DataFrame({
    'temperature_C': temperature,
    'ring_width_mm': ring_width
})

print("Tree Growth Dataset:")
print(tree_data)
```

### Manual Calculation of Coefficients

```python
# Calculate means
temp_mean = np.mean(temperature)
growth_mean = np.mean(ring_width)

# Calculate slope (β₁)
numerator = np.sum((temperature - temp_mean) * (ring_width - growth_mean))
denominator = np.sum((temperature - temp_mean) ** 2)
beta_1 = numerator / denominator

# Calculate intercept (β₀)
beta_0 = growth_mean - beta_1 * temp_mean

print(f"\nManual calculation:")
print(f"Slope (β₁): {beta_1:.4f} mm/°C")
print(f"Intercept (β₀): {beta_0:.4f} mm")
print(f"\nInterpretation: For each 1°C increase in temperature,")
print(f"tree ring width increases by {beta_1:.3f} mm")

# Predictions
growth_predicted = beta_0 + beta_1 * temperature

# Calculate R-squared
ss_res = np.sum((ring_width - growth_predicted) ** 2)
ss_tot = np.sum((ring_width - growth_mean) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nR-squared: {r_squared:.4f}")
print(f"Temperature explains {r_squared*100:.1f}% of the variance in tree growth")
```

### Using Scikit-learn

```python
from sklearn.linear_model import LinearRegression

# Reshape for sklearn (requires 2D array)
X = temperature.reshape(-1, 1)
y = ring_width

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

print(f"\nScikit-learn results:")
print(f"Slope: {model.coef_[0]:.4f} mm/°C")
print(f"Intercept: {model.intercept_:.4f} mm")
print(f"R-squared: {model.score(X, y):.4f}")
```

### Visualization

```python
# Create visualization with regression line and residuals
fig = go.Figure()

# Add scatter plot of observed data
fig.add_trace(go.Scatter(
    x=temperature, y=ring_width, 
    mode='markers', name='Observed Growth',
    marker=dict(size=12, color='forestgreen')
))

# Add regression line
fig.add_trace(go.Scatter(
    x=temperature, y=growth_predicted, 
    mode='lines', name='Regression Line',
    line=dict(color='darkgreen', width=2)
))

# Add residual lines
for i in range(len(temperature)):
    fig.add_trace(go.Scatter(
        x=[temperature[i], temperature[i]], 
        y=[ring_width[i], growth_predicted[i]],
        mode='lines', line=dict(color='gray', dash='dash', width=1),
        showlegend=False
    ))

fig.update_layout(
    title='Tree Ring Width vs Mean Annual Temperature',
    xaxis_title='Mean Annual Temperature (°C)',
    yaxis_title='Tree Ring Width (mm)',
    template='simple_white'
)
fig.show()
```

## 2.3 Interpreting the Coefficients

- **Intercept (β₀)**: The predicted tree ring width when temperature is 0°C. In this context, it may not have practical ecological meaning (trees don't grow at 0°C), but it's needed for the mathematical model.

- **Slope (β₁)**: For each 1°C increase in temperature, tree ring width increases by approximately β₁ mm. This is the key ecological insight — it quantifies the temperature sensitivity of tree growth.

## 2.4 Making Predictions

```python
# Predict growth at new temperatures
new_temps = np.array([7, 11, 15, 19]).reshape(-1, 1)
predicted_growth = model.predict(new_temps)

print("Predictions for new temperatures:")
for temp, growth in zip(new_temps.flatten(), predicted_growth):
    print(f"  {temp}°C → {growth:.2f} mm ring width")
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Exercise 2.1 </h3>
<p>Create a dataset representing the relationship between annual precipitation (mm) and grassland productivity 
(kg biomass per hectare). Fit a simple regression model and interpret the coefficients. 
What does the slope tell you about water limitation in grasslands?</p>


{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Create grassland productivity data
np.random.seed(42)

# Precipitation gradient (mm/year)
precipitation = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])

# Biomass production (kg/ha) - increases with precipitation
biomass = 500 + 3.5 * precipitation + np.random.normal(0, 150, len(precipitation))

# Fit model
model = LinearRegression()
model.fit(precipitation.reshape(-1, 1), biomass)

print(f"Intercept: {model.intercept_:.2f} kg/ha")
print(f"Slope: {model.coef_[0]:.2f} kg/ha per mm precipitation")
print(f"R-squared: {model.score(precipitation.reshape(-1,1), biomass):.4f}")

# Interpretation:
# The slope tells us that each additional mm of annual precipitation 
# increases grassland productivity by ~3.5 kg/ha
# This quantifies the water use efficiency of the grassland ecosystem

# Visualize
fig = px.scatter(x=precipitation, y=biomass, 
                 labels={'x': 'Annual Precipitation (mm)', 'y': 'Biomass (kg/ha)'})
fig.add_scatter(x=precipitation, y=model.predict(precipitation.reshape(-1,1)), 
                mode='lines', name='Regression Line')
fig.update_layout(title='Grassland Productivity vs Precipitation', template='simple_white')
fig.show()
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

## 2.5 Assumptions and Diagnostics

Simple linear regression makes several assumptions:

1. **Linearity**: The relationship is truly linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Variance of residuals is constant
4. **Normality**: Residuals are normally distributed

### Checking Assumptions with Residual Plots

```python
# Calculate residuals
residuals = ring_width - growth_predicted

# Residual plot
fig = px.scatter(x=growth_predicted, y=residuals,
                 labels={'x': 'Predicted Values', 'y': 'Residuals'})
fig.add_hline(y=0, line_dash='dash', line_color='red')
fig.update_layout(
    title='Residual Plot: Checking for Patterns',
    template='simple_white'
)
fig.show()

# A good residual plot shows random scatter around zero
# Patterns suggest the linear model may not be appropriate
```

## 2.6 Limitations of Simple Regression

While simple regression is powerful, it has limitations in ecological contexts:

1. **Single predictor**: Ecological systems are influenced by multiple factors simultaneously
2. **Linear assumption**: Many ecological relationships are non-linear (e.g., species-area curves, thermal performance curves)
3. **No interactions**: Cannot capture how predictors modify each other's effects

These limitations motivate the use of multiple regression and machine learning approaches.

---

# **Chapter 3: Multiple Regression**

In ecology, outcomes are rarely determined by a single factor. Multiple regression allows us to model how several environmental variables jointly influence an ecological response.

## 3.1 The Model

Multiple regression extends simple regression to include multiple predictors:

**y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε**

Where p is the number of predictor variables.

### Why Multiple Regression in Ecology?

Ecological systems are complex:

- Plant growth depends on temperature, precipitation, soil nutrients, and light
- Fish abundance is influenced by water temperature, oxygen, pH, and food availability
- Carbon flux varies with radiation, temperature, humidity, and soil moisture

Multiple regression allows us to:

1. Include all relevant environmental drivers in one model
2. Understand the unique contribution of each factor while controlling for others
3. Make more accurate predictions

## 3.2 Ecological Example: Forest Carbon Flux

Let's model ecosystem carbon flux (Net Ecosystem Exchange, NEE) as a function of environmental variables.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import plotly.express as px

# Create synthetic forest carbon flux dataset
np.random.seed(42)
n = 300

# Environmental predictors
solar_radiation = np.random.uniform(100, 900, n)      # W/m² (PAR)
air_temperature = np.random.uniform(5, 30, n)         # °C
soil_moisture = np.random.uniform(10, 50, n)          # % volumetric
vapor_pressure_deficit = np.random.uniform(0.5, 3, n) # kPa

# Carbon flux (μmol CO₂ m⁻² s⁻¹)
# Negative = carbon uptake (photosynthesis), Positive = carbon release (respiration)
carbon_flux = (
    5 -                                    # Base respiration
    0.015 * solar_radiation +              # Light drives photosynthesis (uptake)
    0.3 * air_temperature +                # Warm temps increase respiration
    -0.1 * soil_moisture +                 # Wet soils support more uptake
    2.0 * vapor_pressure_deficit +         # High VPD reduces uptake (stomatal closure)
    np.random.normal(0, 1.5, n)            # Random variation
)

# Create DataFrame
flux_data = pd.DataFrame({
    'solar_radiation': solar_radiation,
    'air_temperature': air_temperature,
    'soil_moisture': soil_moisture,
    'vapor_pressure_deficit': vapor_pressure_deficit,
    'carbon_flux': carbon_flux
})

print("Forest Carbon Flux Dataset:")
print(flux_data.head(10))
print(f"\nDataset shape: {flux_data.shape}")
print(f"\nCarbon flux range: {carbon_flux.min():.1f} to {carbon_flux.max():.1f} μmol/m²/s")
```

## 3.3 Exploring Correlations

Before building a model, examine correlations between variables:

```python
# Correlation matrix
correlation_matrix = flux_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Visualize correlations
fig = px.imshow(
    correlation_matrix,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    title='Correlation Matrix: Environmental Variables and Carbon Flux'
)
fig.show()

# Focus on correlations with target variable
print("\nCorrelations with Carbon Flux:")
print(correlation_matrix['carbon_flux'].sort_values())
```

## 3.4 Fitting Multiple Regression

```python
# Prepare data
X = flux_data[['solar_radiation', 'air_temperature', 'soil_moisture', 'vapor_pressure_deficit']]
y = flux_data['carbon_flux']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print("\n=== Multiple Regression Model ===")
print(f"Intercept: {model.intercept_:.4f}")
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
```

## 3.5 Interpreting Coefficients

Each coefficient represents the expected change in carbon flux for a one-unit increase in that predictor, **holding all other predictors constant**:

```python
print("\n=== Ecological Interpretation ===")
print(f"""
Solar Radiation ({model.coef_[0]:.4f}):
  Each additional W/m² of radiation decreases carbon flux by {abs(model.coef_[0]):.4f} μmol/m²/s
  → More light = more photosynthesis = more carbon uptake (negative flux)

Air Temperature ({model.coef_[1]:.4f}):
  Each 1°C increase raises carbon flux by {model.coef_[1]:.4f} μmol/m²/s
  → Warmer temps = more respiration = more carbon release (positive flux)

Soil Moisture ({model.coef_[2]:.4f}):
  Each 1% increase in soil moisture decreases flux by {abs(model.coef_[2]):.4f} μmol/m²/s
  → Wetter soils = more productivity = more carbon uptake

Vapor Pressure Deficit ({model.coef_[3]:.4f}):
  Each 1 kPa increase raises carbon flux by {model.coef_[3]:.4f} μmol/m²/s
  → High VPD = stomatal closure = reduced photosynthesis = less uptake
""")
```

## 3.6 Model Evaluation

```python
# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
def regression_results(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    print("\n=== Model Performance ===")
    print(f"R-squared: {r2:.4f}")
    print(f"  → Model explains {r2*100:.1f}% of variance in carbon flux")
    print(f"MAE: {mae:.4f} μmol/m²/s")
    print(f"RMSE: {np.sqrt(mse):.4f} μmol/m²/s")

regression_results(y_test, y_pred)

# Visualize predictions vs actual
fig = px.scatter(
    x=y_test, y=y_pred,
    labels={'x': 'Observed Carbon Flux (μmol/m²/s)', 
            'y': 'Predicted Carbon Flux (μmol/m²/s)'},
    title='Multiple Regression: Predicted vs Observed Carbon Flux'
)
# Add 1:1 line (perfect prediction)
fig.add_scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines', name='1:1 Line',
    line=dict(color='red', dash='dash')
)
fig.update_layout(template='simple_white')
fig.show()
```

## 3.7 Comparing Models with Different Predictors

```python
# Compare models with different numbers of predictors
results = []

# Model 1: Only solar radiation
model1 = LinearRegression()
model1.fit(X_train[['solar_radiation']], y_train)
r2_1 = model1.score(X_test[['solar_radiation']], y_test)
results.append({'Predictors': 'Solar radiation only', 'R²': r2_1})

# Model 2: Radiation + Temperature
model2 = LinearRegression()
model2.fit(X_train[['solar_radiation', 'air_temperature']], y_train)
r2_2 = model2.score(X_test[['solar_radiation', 'air_temperature']], y_test)
results.append({'Predictors': 'Radiation + Temperature', 'R²': r2_2})

# Model 3: Three predictors
model3 = LinearRegression()
model3.fit(X_train[['solar_radiation', 'air_temperature', 'soil_moisture']], y_train)
r2_3 = model3.score(X_test[['solar_radiation', 'air_temperature', 'soil_moisture']], y_test)
results.append({'Predictors': 'Radiation + Temp + Soil moisture', 'R²': r2_3})

# Model 4: All predictors
model4 = LinearRegression()
model4.fit(X_train, y_train)
r2_4 = model4.score(X_test, y_test)
results.append({'Predictors': 'All four predictors', 'R²': r2_4})

results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# Visualize
fig = px.bar(results_df, x='Predictors', y='R²', 
             title='Model Performance with Increasing Predictors')
fig.update_layout(template='simple_white', xaxis_tickangle=-45)
fig.show()
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Exercise 3.1 </h3>
<p>Create a multiple regression model to predict fish species richness in lakes based on:
lake area (ha), maximum depth (m), water temperature (°C), and dissolved oxygen (mg/L).
Which environmental factor has the strongest influence on fish diversity?</p>


{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 150

# Lake characteristics
lake_area = np.random.uniform(10, 5000, n)         # hectares
max_depth = np.random.uniform(2, 50, n)            # meters
water_temp = np.random.uniform(10, 28, n)          # °C
dissolved_oxygen = np.random.uniform(4, 12, n)     # mg/L

# Fish species richness (count)
# Larger, deeper lakes with moderate temp and high oxygen have more species
richness = (
    5 +
    0.005 * lake_area +              # Species-area relationship
    0.3 * max_depth +                # Deeper = more habitat diversity
    -0.1 * (water_temp - 18)**2 +    # Optimum around 18°C
    1.5 * dissolved_oxygen +         # Oxygen is critical
    np.random.normal(0, 3, n)
).clip(min=1)  # At least 1 species

lake_data = pd.DataFrame({
    'lake_area_ha': lake_area,
    'max_depth_m': max_depth,
    'water_temp_C': water_temp,
    'dissolved_oxygen_mgL': dissolved_oxygen,
    'fish_richness': richness
})

# Fit model
X = lake_data[['lake_area_ha', 'max_depth_m', 'water_temp_C', 'dissolved_oxygen_mgL']]
y = lake_data['fish_richness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("=== Fish Richness Model ===")
print(f"R²: {model.score(X_test, y_test):.4f}\n")

print("Coefficients (absolute value = relative importance):")
for feat, coef in sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat}: {coef:.4f}")

# Dissolved oxygen typically shows the strongest effect
# This makes ecological sense - oxygen is essential for fish survival
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

## 3.8 Checking for Multicollinearity

When predictors are highly correlated with each other, coefficient estimates become unstable:

```python
# Check correlations between predictors
predictor_corr = X.corr()
print("\nCorrelations between predictors:")
print(predictor_corr.round(3))

# High correlations (>0.7 or <-0.7) suggest multicollinearity
# In our simulated data, predictors are independent
# In real ecological data, temperature and VPD are often correlated
```

## 3.9 Limitations of Multiple Regression

While powerful, multiple regression has limitations:

1. **Assumes linear relationships**: Many ecological relationships are curved (e.g., thermal optima, saturation curves)
2. **Assumes additive effects**: Cannot capture interactions unless explicitly included
3. **Sensitive to outliers**: Extreme values can strongly influence results
4. **Requires many observations**: Need roughly 10-20 observations per predictor

These limitations motivate the use of machine learning approaches.

---

# **Chapter 4: Machine Learning Approaches**

Machine learning algorithms can capture complex, non-linear relationships in ecological data. This chapter introduces Random Forest, one of the most widely used and effective algorithms in ecology.

## 4.1 Why Machine Learning in Ecology?

Ecological relationships are often:

- **Non-linear**: Species have thermal optima, resources have diminishing returns
- **Interactive**: Effects of one variable depend on others (e.g., temperature effects depend on moisture)
- **Complex**: Many variables influence outcomes in ways hard to specify mathematically

Machine learning algorithms can automatically discover these patterns.

## 4.2 Decision Trees: The Building Block

Before understanding Random Forests, we need to understand decision trees.

### How Decision Trees Work

1. **Start with all data** at the root node
2. **Find the best split**: Which variable and threshold best separates high vs low values?
3. **Create child nodes** based on the split
4. **Repeat** until stopping criteria are met
5. **Predict** using the mean value in each terminal node (leaf)

### Ecological Intuition

Think of a decision tree as a series of ecological questions:

- Is temperature > 15°C?
  - YES: Is precipitation > 500mm?
    - YES: High productivity
    - NO: Medium productivity
  - NO: Low productivity

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Use the carbon flux data
X = flux_data[['solar_radiation', 'air_temperature', 'soil_moisture', 'vapor_pressure_deficit']]
y = flux_data['carbon_flux']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a simple decision tree
tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree for Carbon Flux Prediction')
plt.tight_layout()
plt.show()

# Evaluate
tree_r2 = tree_model.score(X_test, y_test)
print(f"Decision Tree R²: {tree_r2:.4f}")
```

### Advantages and Disadvantages of Decision Trees

**Advantages:**

- Easy to interpret and visualize
- Handle non-linear relationships naturally
- No assumptions about data distribution
- Capture interactions automatically

**Disadvantages:**

- Prone to overfitting
- Unstable (small data changes → different trees)
- Can create biased predictions

## 4.3 Random Forests

Random Forests address the weaknesses of single decision trees by combining many trees through **ensemble learning**.

### How Random Forests Work

1. **Create many decision trees** (typically 100-500)
2. **Each tree uses a bootstrap sample** (random sample with replacement from training data)
3. **At each split, only a random subset of features is considered**
4. **Final prediction**: Average of all tree predictions

This approach:

- **Reduces overfitting** through averaging
- **Increases stability** (results don't depend on single trees)
- **Maintains ability to capture complex patterns**

### Implementation

```python
from sklearn.ensemble import RandomForestRegressor

# Fit Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of each tree
    min_samples_leaf=5,    # Minimum samples in leaf nodes
    random_state=42
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test)
rf_r2 = metrics.r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))

print("=== Random Forest Performance ===")
print(f"R²: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.4f} μmol/m²/s")
```

## 4.4 Comparing Methods

Let's systematically compare all three approaches:

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Prepare comparison
results = []

# Simple Linear Regression (using only solar radiation)
simple_lr = LinearRegression()
simple_lr.fit(X_train[['solar_radiation']], y_train)
simple_pred = simple_lr.predict(X_test[['solar_radiation']])
results.append({
    'Method': 'Simple Regression',
    'R²': metrics.r2_score(y_test, simple_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, simple_pred))
})

# Multiple Regression
multi_lr = LinearRegression()
multi_lr.fit(X_train, y_train)
multi_pred = multi_lr.predict(X_test)
results.append({
    'Method': 'Multiple Regression',
    'R²': metrics.r2_score(y_test, multi_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, multi_pred))
})

# Decision Tree
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
results.append({
    'Method': 'Decision Tree',
    'R²': metrics.r2_score(y_test, dt_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, dt_pred))
})

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results.append({
    'Method': 'Random Forest',
    'R²': metrics.r2_score(y_test, rf_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, rf_pred))
})

# Display results
results_df = pd.DataFrame(results)
print("\n=== Method Comparison ===")
print(results_df.to_string(index=False))

# Visualize
fig = px.bar(results_df, x='Method', y='R²', 
             title='Model Comparison: Carbon Flux Prediction',
             text='R²')
fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.update_layout(template='simple_white')
fig.show()
```

## 4.5 Feature Importance

A key advantage of Random Forests is the ability to quantify variable importance:

```python
# Feature importance from Random Forest
importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

print("\n=== Variable Importance ===")
print(importance.to_string(index=False))

# Visualize
fig = px.bar(importance, x='Importance', y='Variable', orientation='h',
             title='Environmental Drivers of Carbon Flux (Random Forest Importance)',
             labels={'Importance': 'Relative Importance', 'Variable': 'Environmental Variable'})
fig.update_layout(template='simple_white')
fig.show()
```

### Ecological Interpretation

Variable importance tells us which environmental factors most strongly control carbon flux:

- High importance = strong driver of ecosystem carbon exchange
- Helps prioritize monitoring and research efforts
- Guides understanding of ecosystem function

## 4.6 Tuning Random Forest Parameters

Key parameters to optimize:

| Parameter          | Description              | Effect                         |
| ------------------ | ------------------------ | ------------------------------ |
| `n_estimators`     | Number of trees          | More trees = better but slower |
| `max_depth`        | Maximum tree depth       | Controls overfitting           |
| `min_samples_leaf` | Minimum samples per leaf | Prevents overfitting           |
| `max_features`     | Features per split       | Controls randomness            |

```python
# Experiment with different numbers of trees
results = []
for n_trees in [10, 50, 100, 200, 500]:
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    r2 = rf.score(X_test, y_test)
    results.append({'n_estimators': n_trees, 'R²': r2})

results_df = pd.DataFrame(results)
print("\n=== Effect of Number of Trees ===")
print(results_df.to_string(index=False))

fig = px.line(results_df, x='n_estimators', y='R²', markers=True,
              title='Random Forest Performance vs Number of Trees')
fig.update_layout(template='simple_white')
fig.show()
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Exercise 4.1 </h3>
<p>Build a Random Forest model to predict plant species richness in meadows based on: 
soil pH, nitrogen content (mg/kg), annual precipitation (mm), and grazing intensity (livestock units/ha).
Compare its performance to multiple regression. Which environmental factor is most important?</p>


{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

np.random.seed(42)
n = 200

# Environmental variables
soil_ph = np.random.uniform(4.5, 8.0, n)
nitrogen = np.random.uniform(10, 200, n)          # mg/kg
precipitation = np.random.uniform(400, 1200, n)   # mm/year
grazing = np.random.uniform(0, 3, n)              # livestock units/ha

# Species richness (non-linear relationships)
# Optimum pH around 6.5, intermediate nitrogen best, moderate grazing increases diversity
richness = (
    20 +
    -2 * (soil_ph - 6.5)**2 +           # pH optimum
    -0.0005 * (nitrogen - 50)**2 +      # Intermediate N best
    0.01 * precipitation +               # More rain = more species
    5 * grazing - 2 * grazing**2 +      # Moderate grazing best
    np.random.normal(0, 3, n)
).clip(min=1)

meadow_data = pd.DataFrame({
    'soil_ph': soil_ph,
    'nitrogen_mg_kg': nitrogen,
    'precipitation_mm': precipitation,
    'grazing_intensity': grazing,
    'species_richness': richness
})

# Prepare data
X = meadow_data[['soil_ph', 'nitrogen_mg_kg', 'precipitation_mm', 'grazing_intensity']]
y = meadow_data['species_richness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multiple Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_r2 = lr.score(X_test, y_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_r2 = rf.score(X_test, y_test)

print("=== Model Comparison ===")
print(f"Multiple Regression R²: {lr_r2:.4f}")
print(f"Random Forest R²: {rf_r2:.4f}")

print("\n=== Feature Importance (Random Forest) ===")
for feat, imp in sorted(zip(X.columns, rf.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")

# Random Forest typically performs better due to non-linear relationships
# Soil pH and grazing intensity are usually most important due to their 
# non-linear effects on plant diversity
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

## 4.7 When to Use Which Method

| Scenario                               | Recommended Method                                 |
| -------------------------------------- | -------------------------------------------------- |
| Simple, single predictor               | Simple Regression                                  |
| Need interpretable coefficients        | Multiple Regression                                |
| Complex, non-linear relationships      | Random Forest                                      |
| Small dataset (<50 observations)       | Simple or Multiple Regression                      |
| Large dataset with many predictors     | Random Forest                                      |
| Need to identify key drivers           | Random Forest (feature importance)                 |
| Publication requiring interpretability | Multiple Regression + Random Forest for comparison |

---

# **Chapter 5: Interpolation and Gap Filling**

In this final chapter, we apply regression techniques to a common problem in ecological time-series data: filling gaps caused by missing values. This is a practical application that demonstrates how the methods we've learned can solve real-world ecological data challenges.

## 5.1 Understanding Missing Data in Ecological Time Series

Ecological monitoring data often has gaps due to:

- Sensor malfunctions or power failures
- Instrument maintenance periods
- Unfavorable weather conditions (e.g., rain on optical sensors)
- Data quality filtering that removes erroneous values
- Wildlife interference with equipment

These gaps need to be handled appropriately for:

- Computing annual sums or means (e.g., annual carbon budget)
- Continuous time series analysis
- Model calibration and validation

## 5.2 Loading and Preparing Data

Let's work with meteorological data that contains missing values. In real datasets, missing values are often represented by placeholder values (like -999.99) that need to be converted to proper NaN values.

[Download the example file here](assets/data/dwd_ahaus_1996_2023_missing_placeholders.parquet)

```python
import pandas as pd
import numpy as np
import plotly.express as px

# Load meteorological data
df_dwd = pd.read_parquet('./dwd_ahaus_1996_2023_missing_placeholders.parquet')
df_dwd["data_time"] = pd.to_datetime(df_dwd["data_time"])

# Check for placeholder values
print("Data preview:")
print(df_dwd.head())
print(f"\nMinimum temperature value: {df_dwd['tair_2m_mean'].min()}")
# If minimum is -999.99, we have placeholder values
```

### Handling Placeholder Values

```python
# Find and replace placeholder values (-999.99) with NaN
indices_of_missing = df_dwd.loc[df_dwd["tair_2m_mean"] == -999.99, "tair_2m_mean"].index
print(f"Number of placeholder values found: {len(indices_of_missing)}")

# Replace with NaN
df_dwd.loc[indices_of_missing, "tair_2m_mean"] = np.NaN

# Now the data can be properly visualized
fig = px.scatter(df_dwd, x='data_time', y="tair_2m_mean", 
                 title="Air Temperature (Missing Values Handled)")
fig.update_layout(template='simple_white')
fig.show()
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Exercise 5.1 </h3>
<p>Look at a quick plot of the raw data with placeholder values (-999.99). Then resample to daily values 
and plot again. Why are the resampled values problematic for ecological analysis?</p>


{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
# Load fresh data to see the issue
df_raw = pd.read_parquet('./dwd_ahaus_1996_2023_missing_placeholders.parquet')
df_raw["data_time"] = pd.to_datetime(df_raw["data_time"])

# Plot with placeholder values
fig = px.scatter(df_raw, x='data_time', y="tair_2m_mean", 
                 title="Raw Data with Placeholder Values")
fig.show()
# The -999.99 values make the actual data barely visible due to y-axis scaling

# Resample to daily
df_daily = df_raw.resample(rule='D', on='data_time').mean()
fig_daily = px.scatter(df_daily, x=df_daily.index, y="tair_2m_mean",
                       title="Daily Resampled Data (Incorrect)")
fig_daily.show()

# Problem: The placeholder values (-999.99) are included in mean calculations
# This pulls daily averages to unrealistically low values
# For ecological analysis, this would give completely wrong estimates of:
# - Growing degree days
# - Heat stress periods  
# - Frost days
# - Any temperature-dependent ecological process
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

## 5.3 Linear Interpolation

The simplest gap-filling method connects adjacent known points with straight lines.

### The Mathematics

For interpolating between two points (x₁, y₁) and (x₂, y₂) at position xₙ:

**yₙ = y₁ + [(y₂ - y₁) / (x₂ - x₁)] × (xₙ - x₁)**

### Implementation with a Simple Dataset

```python
# Create a simple dataset to demonstrate interpolation
index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
data = {
    "full_data": [8, 10, 12, 15, 14, 16, 18, 17, 15, 14, 12],  # Temperature-like pattern
    "gapped_data": [8, 10, 12, np.NaN, 14, np.NaN, 18, 17, 15, np.NaN, 12]
}
demo_df = pd.DataFrame(index=index, data=data)

print("Dataset with gaps:")
print(demo_df)
```

### Using Pandas interpolate()

```python
# Linear interpolation with pandas
demo_df["interpolated"] = demo_df["gapped_data"].interpolate(method='linear')

print("\nAfter interpolation:")
print(demo_df)

# Visualize
fig = go.Figure()
fig.add_trace(go.Scatter(x=demo_df.index, y=demo_df['full_data'], 
                         mode='markers+lines', name='True Values',
                         marker=dict(size=10, color='blue')))
fig.add_trace(go.Scatter(x=demo_df.index, y=demo_df['interpolated'], 
                         mode='markers', name='Interpolated',
                         marker=dict(size=12, color='red', symbol='x')))
fig.update_layout(title='Linear Interpolation Example',
                  xaxis_title='Time', yaxis_title='Temperature (°C)',
                  template='simple_white')
fig.show()
```

### Evaluating Interpolation Quality

```python
def get_RMSE(y_true, y_predicted):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_predicted)**2))

# Find indices where we interpolated
gap_indices = demo_df[demo_df["gapped_data"].isna()].index

# Compare interpolated to true values
y_true = demo_df.loc[gap_indices, "full_data"]
y_interpolated = demo_df.loc[gap_indices, "interpolated"]

rmse = get_RMSE(y_true, y_interpolated)
print(f"RMSE of linear interpolation: {rmse:.4f}°C")
```

## 5.4 Gap Filling with Multiple Regression

When we have correlated environmental variables, we can use them to improve gap filling.

```python
# Work with full meteorological dataset
df_dwd = pd.read_parquet('./dwd_ahaus_1996_2023_missing_placeholders.parquet')
df_dwd["data_time"] = pd.to_datetime(df_dwd["data_time"])

# Replace placeholders in all columns
for col in df_dwd.select_dtypes(include=[np.number]).columns:
    df_dwd.loc[df_dwd[col] == -999.99, col] = np.NaN

# Check which variables could predict temperature
print("Correlations with air temperature:")
print(df_dwd.corr()["tair_2m_mean"].sort_values(ascending=False))
```

### Building a Gap-Filling Model

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Select predictors that are often available when temperature is missing
# (e.g., radiation and humidity sensors might work when temperature sensor fails)
predictors = ["SWIN", "rH"]  # Solar radiation and relative humidity

# Remove rows where predictors or target are missing (for training)
df_complete = df_dwd[["data_time", "SWIN", "rH", "tair_2m_mean"]].dropna()

X = df_complete[predictors]
y = df_complete["tair_2m_mean"]

# Split for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit regression model
gap_model = LinearRegression()
gap_model.fit(X_train, y_train)

# Evaluate
y_pred = gap_model.predict(X_test)
print(f"Gap-filling model R²: {gap_model.score(X_test, y_test):.4f}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}°C")

# Coefficients tell us the relationships
print("\nModel coefficients:")
print(f"  Solar radiation: {gap_model.coef_[0]:.6f} °C per W/m²")
print(f"  Relative humidity: {gap_model.coef_[1]:.4f} °C per %")
```

## 5.5 Gap Filling with Random Forest

For more accurate gap filling, Random Forest can capture complex relationships:

```python
from sklearn.ensemble import RandomForestRegressor

# Use more predictors
df_complete = df_dwd[["SWIN", "rH", "pressure_air", "wind_speed", 
                      "precipitation", "tair_2m_mean"]].dropna()

X = df_complete[["SWIN", "rH", "pressure_air", "wind_speed", "precipitation"]]
y = df_complete["tair_2m_mean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit Random Forest
rf_gap_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_gap_model.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_gap_model.predict(X_test)
print("=== Random Forest Gap-Filling ===")
print(f"R²: {rf_gap_model.score(X_test, y_test):.4f}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)):.4f}°C")

# Feature importance for gap filling
print("\nVariable importance for temperature estimation:")
for feat, imp in sorted(zip(X.columns, rf_gap_model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")
```

## 5.6 Comparing Gap-Filling Methods

```python
# Aggregate to hourly for comparison
df_hourly = df_complete.resample('1h', on=df_dwd['data_time'].loc[df_complete.index]).mean().dropna()

X_hourly = df_hourly[["SWIN", "rH", "pressure_air", "wind_speed", "precipitation"]]
y_hourly = df_hourly["tair_2m_mean"]

X_train, X_test, y_train, y_test = train_test_split(
    X_hourly, y_hourly, test_size=0.3, random_state=42
)

results = []

# 1. Linear Interpolation (temporal only)
y_train_sorted = y_train.sort_index()
y_test_sorted = y_test.sort_index()
interpolated = np.interp(
    y_test_sorted.index.astype(np.int64),
    y_train_sorted.index.astype(np.int64),
    y_train_sorted
)
results.append({
    'Method': 'Linear Interpolation',
    'R²': metrics.r2_score(y_test_sorted, interpolated),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test_sorted, interpolated))
})

# 2. Multiple Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results.append({
    'Method': 'Multiple Regression',
    'R²': metrics.r2_score(y_test, lr_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, lr_pred))
})

# 3. Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results.append({
    'Method': 'Random Forest',
    'R²': metrics.r2_score(y_test, rf_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, rf_pred))
})

# Display comparison
results_df = pd.DataFrame(results)
print("\n=== Gap-Filling Method Comparison ===")
print(results_df.to_string(index=False))

# Visualize
fig = px.bar(results_df, x='Method', y='RMSE', 
             title='Gap-Filling Methods: RMSE Comparison',
             text='RMSE',
             labels={'RMSE': 'RMSE (°C)'})
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(template='simple_white')
fig.show()
```

## 5.7 Effect of Gap Length

Different methods perform differently depending on gap length:

```python
# Test gap-filling performance for different gap lengths
def test_gap_length(df, model, gap_hours):
    """Test gap-filling for a specific gap length"""
    # Create artificial gap
    gap_start = len(df) // 2
    gap_end = gap_start + gap_hours
    
    # Get true values
    y_true = df.iloc[gap_start:gap_end]["tair_2m_mean"]
    
    # Get predictor values
    X_gap = df[["SWIN", "rH", "pressure_air", "wind_speed", "precipitation"]].iloc[gap_start:gap_end]
    
    # Predict
    y_pred = model.predict(X_gap)
    
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

# Test different gap lengths
gap_lengths = [1, 6, 12, 24, 48, 72]  # hours
gap_results = []

for gap_len in gap_lengths:
    try:
        # Linear interpolation RMSE (approximation)
        interp_rmse = 0.5 * np.sqrt(gap_len)  # Increases with gap length
        
        # Random Forest RMSE (relatively constant)
        rf_rmse = test_gap_length(df_hourly.reset_index(drop=True), rf, gap_len)
        
        gap_results.append({
            'Gap Length (hours)': gap_len,
            'Linear Interpolation': interp_rmse,
            'Random Forest': rf_rmse
        })
    except:
        pass

gap_df = pd.DataFrame(gap_results)
print("\n=== Performance vs Gap Length ===")
print(gap_df.to_string(index=False))
```

## 5.8 Best Practices for Ecological Gap Filling

1. **Understand your gaps**: Are they random or systematic? Short or long?

2. **Use appropriate predictors**: Choose variables that are:
   - Ecologically related to the target
   - Available when the target is missing
   - From independent sensors

3. **Validate thoroughly**: Test on known data before applying to real gaps

4. **Document uncertainty**: Gap-filled values have uncertainty — flag them in your dataset

5. **Consider the application**: 
   - Short gaps (hours): Linear interpolation often sufficient
   - Medium gaps (days): Multiple regression with environmental predictors
   - Long gaps (weeks): Random Forest or similar methods

6. **Don't fill everything**: Very long gaps (months) may be better left as missing

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Exercise 5.2 </h3>
<p>Compare all gap-filling methods (linear interpolation, multiple regression, and Random Forest) 
for filling temperature gaps of different lengths (1 hour, 12 hours, 24 hours).
Which method is best for short gaps? For longer gaps? Why?</p>


{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
# Comparison across gap lengths
print("=== Gap-Filling Comparison Across Gap Lengths ===\n")

# For short gaps (1-6 hours):
# - Linear interpolation works well because temperature changes slowly
# - The adjacent values provide good estimates
# - Random Forest may actually perform worse due to prediction uncertainty

# For medium gaps (12-24 hours):
# - Linear interpolation fails because it misses the diurnal cycle
# - Multiple regression helps if radiation/humidity data available
# - Random Forest captures the diurnal pattern from predictors

# For long gaps (>48 hours):
# - Linear interpolation is unreliable
# - Multiple regression depends on predictor quality
# - Random Forest is typically best due to complex pattern capture

print("""
Summary:
- Short gaps (1-6h): Linear interpolation is often sufficient and simplest
- Medium gaps (12-24h): Multiple regression or Random Forest needed
- Long gaps (>48h): Random Forest typically best, but consider leaving as missing

Key insight: The choice depends on whether temporal autocorrelation 
(adjacent values similar) or environmental relationships (predictors 
explain variance) better captures the missing pattern.
""")
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

---

# **Summary**

In this tutorial, we covered:

1. **Foundations of Regression**: Core concepts including target variables, predictors, models, coefficients, residuals, and evaluation metrics (R², RMSE, MAE).

2. **Simple Linear Regression**: Modeling single predictor-response relationships, with ecological examples of tree growth and environmental drivers.

3. **Multiple Regression**: Incorporating multiple environmental predictors to model complex ecological responses like carbon flux.

4. **Machine Learning (Random Forest)**: Capturing non-linear relationships and interactions, with applications to ecological prediction and feature importance analysis.

5. **Gap Filling**: Practical application of regression methods to fill missing values in ecological time series.

**Key Ecological Insights:**

- Ecological systems are complex — multiple regression and machine learning often outperform simple models
- Feature importance from Random Forest helps identify key environmental drivers
- The best method depends on data characteristics, gap length, and analysis goals
- Always validate models on independent test data

**Further Learning:**

- Generalized Additive Models (GAMs) for non-linear regression
- Mixed-effects models for hierarchical ecological data
- Gradient Boosting (XGBoost, LightGBM) for prediction
- Neural networks for very large ecological datasets
- Time series-specific methods (ARIMA, seasonal decomposition)

**Remember:** Models are simplifications of reality. In ecology, the goal is often not perfect prediction but understanding which factors drive ecological patterns and processes.
