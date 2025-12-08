---
title: "3. REGRESSION"
layout: default
nav_order: 4
---
# **Regression**

## Table of Contents

- [1. What's Regression All About?](#chapter-1-whats-regression-all-about)
- [2. Simple Linear Regression](#chapter-2-simple-linear-regression)
- [3. Multiple Regression](#chapter-3-multiple-regression)
- [4. Machine Learning with Random Forests](#chapter-4-machine-learning-with-random-forests)
- [5. Filling Gaps in Time Series](#chapter-5-filling-gaps-in-time-series)

Welcome! This tutorial will walk you through regression analysis - one of the most useful tools you'll encounter for making sense of ecological data. We'll start from the basics and work our way up to more advanced machine learning methods.
Don't worry if statistics isn't your strong suit. We'll take it step by step, and by the end you should feel comfortable applying these techniques to your own data.


---

# **1. What's Regression All About?**

## The Basic Idea

Here's the thing: as ecologists, we're constantly trying to figure out what drives the patterns we observe. Why are there more species in some places than others? What makes trees grow faster? How does temperature affect animal behavior?

Regression gives us a way to quantify these relationships. Instead of just saying "warmer temperatures seem to increase growth," we can say "for every 1°C increase in temperature, tree ring width increases by 0.15 mm." That's powerful stuff.

At its core, regression asks: **how does one thing change when another thing changes?**

## A Quick Example

Let's say you're studying tree growth across a temperature gradient. You measure tree ring widths at different sites:

| Mean Annual Temperature (°C) | Tree Ring Width (mm) |
|------------------------------|----------------------|
| 8                            | 1.2                  |
| 10                           | 1.8                  |
| 12                           | 2.4                  |
| 14                           | 2.9                  |
| 16                           | 3.2                  |

You can see there's a pattern - warmer sites have wider rings. But how strong is this relationship? Can we predict growth at a site with 11°C mean temperature? Regression helps us answer these questions.

## What Can Regression Achieve?

In ecological research, regression is useful for:

**Making predictions** - You've measured carbon flux at 20 sites, but you want to estimate it across the whole landscape. Regression lets you predict values at unmeasured locations based on environmental variables you can get from satellite data.

**Understanding relationships** - Does nitrogen addition actually increase plant biomass? By how much? Is the effect statistically significant or could it just be noise?

**Figuring out what matters** - When you have 15 environmental variables that might explain species richness, regression helps you sort out which ones are actually important.

**Supporting management decisions** - If you know how much habitat area affects population size, you can make informed recommendations about reserve design.

## Terminology

Before we dive in, let's get our vocabulary straight. Different fields use different terms for the same things, which can be confusing.

**Target variable**
- You can also call it response variable, dependent variable, outcome
- The variable you're trying to predict or explain
- We usually call it **y**
- Examples: species richness, biomass, survival rate, carbon flux

**independent variables**
- Can also be termed predictors, features, independent variables, explanatory variables
- The variable you use to make predictions or explain target variables 
- We call these **x** (or x₁, x₂, etc. when there are several)
- Examples: temperature, precipitation, soil pH, elevation

**The model:**
- This is the mathematical equation that describes how x relates to y
- General form: y = f(x) + error
- The "error" part is important - it acknowledges that our model won't be perfect

**Coefficients:**
- These are the numbers in our model that define the relationship
- In a simple model like y = 3 + 2x, the "3" is the intercept and "2" is the slope
- We estimate these from our data

**Residuals:**
- The difference between what we observed and what our model predicted
- Small residuals = good model fit
- Patterns in residuals = something's wrong with our model

## How Does Regression Actually Work?

The basic process goes like this:

1. **Choose a model type.** Are you assuming a straight line relationship? A curve? Multiple predictors?

2. **Fit the model to your data.** This means finding the coefficient values that make your predictions as close to the observations as possible.

3. **Check if it worked.** Look at how well the model fits, whether the assumptions are met, and whether the results make ecological sense.

The most common approach for step 2 is called "least squares" - we find the coefficients that minimize the sum of squared differences between observed and predicted values. We square the differences so that positive and negative errors don't cancel out.

## Evaluating Your Model

How do you know if your model is any good? A few key metrics:

**R² (R-squared)**: This tells you what fraction of the variation in your data is explained by the model. An R² of 0.7 means your model explains 70% of the variance. What's "good" depends entirely on your system - in controlled experiments 0.9 might be expected, while in field ecology 0.3 might be excellent.
<div>$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} $$</div>
Where:

$y_i$ is the observed value

$\hat{y}_i$ is the predicted value  

$\bar{y}$ is the mean of observed values

$SS_{res}$ is the sum of squared residuals

$SS_{tot}$ is the total sum of squares


**RMSE (Root Mean Square Error)**: This is the average size of your prediction errors, in the same units as your response variable. An RMSE of 2.5°C for a temperature model means your predictions are typically off by about 2.5 degrees.
<div>$$ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} $$</div>

**MAE (Mean Absolute Error)**: Similar to RMSE but less sensitive to outliers. Useful when you have some weird extreme values in your data.
<div>$$ MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| $$</div>
The difference between RMSE and MAE? RMSE penalizes large errors more heavily because of the squaring. If you have a few really bad predictions, RMSE will be much higher than MAE. This can be useful for detecting outliers or problematic predictions.

---

# **Chapter 2: Simple Linear Regression**

Alright, let's actually do some regression. We'll start with the simplest case: one predictor, one response, straight line relationship.

## The Model

Simple linear regression fits this equation:

**y = β₀ + β₁x**

Where:

- β₀ is the intercept (value of y when x is zero)
- β₁ is the slope (how much y changes for each unit increase in x)

That's it. We're just fitting a line through our data points.

## Let's Try It: Penguin Body Mass and Flipper Length

We'll use the Palmer Penguins dataset - real measurements collected by Dr. Kristen Gorman at Palmer Station, Antarctica.

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the penguins dataset
# You can install it with: pip install palmerpenguins
from palmerpenguins import load_penguins
penguins = load_penguins()

# Take a look at what we have
print("Dataset shape:", penguins.shape)
print("\nFirst few rows:")
print(penguins.head())

# Check for missing values and drop them for now
penguins_clean = penguins.dropna(subset=['flipper_length_mm', 'body_mass_g'])
print(f"\nComplete cases: {len(penguins_clean)}")
```

Now let's explore the relationship between flipper length and body mass:

```python
# Quick visualization
fig = px.scatter(penguins_clean, x='flipper_length_mm', y='body_mass_g',
                 color='species',
                 title='Penguin Body Mass vs Flipper Length')
fig.update_layout(template='simple_white')
fig.show()
```

You'll see there's clearly a positive relationship - longer flippers go with heavier birds. Let's quantify it.

### Fitting the Regression

```python
from sklearn.linear_model import LinearRegression

# Prepare the data
X = penguins_clean[['flipper_length_mm']].values  # sklearn wants 2D array
y = penguins_clean['body_mass_g'].values

# Fit the model
model = LinearRegression()
model.fit(X, y)

print(f"Slope: {model.coef_[0]:.2f} g per mm")
print(f"Intercept: {model.intercept_:.2f} g")
print(f"R-squared: {model.score(X, y):.3f}")
```

### Visualizing the Fit

```python
# Get predictions for the regression line
X_line = np.linspace(penguins_clean['flipper_length_mm'].min(), 
                     penguins_clean['flipper_length_mm'].max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)

fig = go.Figure()

# Data points
fig.add_trace(go.Scatter(
    x=penguins_clean['flipper_length_mm'], 
    y=penguins_clean['body_mass_g'],
    mode='markers', name='Observations',
    marker=dict(size=8, opacity=0.6)
))

# Regression line
fig.add_trace(go.Scatter(
    x=X_line.flatten(), y=y_line,
    mode='lines', name='Regression line',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title='Penguin Body Mass vs Flipper Length',
    xaxis_title='Flipper Length (mm)',
    yaxis_title='Body Mass (g)',
    template='simple_white'
)
fig.show()
```

## What Do These Numbers Mean?

With real data, you should get something like:

- **Slope ≈ 49.7 g/mm**: For every 1 mm increase in flipper length, body mass increases by about 50 grams
- **Intercept ≈ -5781 g**: This would be the predicted mass at flipper length = 0, which makes no biological sense (negative mass!), but it's needed mathematically to position the line
- **R² ≈ 0.76**: Flipper length explains about 76% of the variation in body mass

That R² is pretty good for biological data! It means if you only know a penguin's flipper length, you can predict its mass reasonably well.

### Making Predictions

```python
# What's the predicted mass for a penguin with 200mm flippers?
new_flipper = np.array([[200]])
predicted_mass = model.predict(new_flipper)
print(f"Predicted mass for 200mm flipper: {predicted_mass[0]:.0f} g")

# What about 180mm?
new_flipper = np.array([[180]])
predicted_mass = model.predict(new_flipper)
print(f"Predicted mass for 180mm flipper: {predicted_mass[0]:.0f} g")
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Try It Yourself </h3>
<p>Using the Palmer Penguins dataset, fit a simple regression predicting bill length from bill depth. 
What do you find? Is the relationship positive or negative? How does R² compare to the flipper-mass relationship?</p>


{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
from palmerpenguins import load_penguins
from sklearn.linear_model import LinearRegression

penguins = load_penguins().dropna(subset=['bill_length_mm', 'bill_depth_mm'])

X = penguins[['bill_depth_mm']].values
y = penguins['bill_length_mm'].values

model = LinearRegression()
model.fit(X, y)

print(f"Slope: {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"R-squared: {model.score(X, y):.3f}")

# Surprise! You'll find a NEGATIVE relationship and very low R²
# This is Simpson's paradox - when you combine the species,
# the overall trend is negative, but within each species 
# the relationship is positive. This is because Gentoo penguins
# have long bills but shallow depth, while Adelie have 
# shorter bills but deeper depth.

# Try it by species:
for species in penguins['species'].unique():
    subset = penguins[penguins['species'] == species]
    X_sp = subset[['bill_depth_mm']].values
    y_sp = subset['bill_length_mm'].values
    model_sp = LinearRegression().fit(X_sp, y_sp)
    print(f"{species}: slope = {model_sp.coef_[0]:.2f}, R² = {model_sp.score(X_sp, y_sp):.3f}")
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

## A Word of Caution

The exercise above reveals something important: simple regression can be misleading when you have groups in your data. The overall flipper-mass relationship also differs among species - Gentoo penguins are bigger than Adelie and Chinstrap. 

This is one reason we need multiple regression - to account for additional factors that might be confounding our results.

---

# **Chapter 3: Multiple Regression**

In the real world, ecological responses depend on many factors simultaneously. Multiple regression lets us include all of them in one model.

## Why Go Multiple?

Looking at the penguin data, body mass depends on more than just flipper length:

- Species differ in overall body size
- Males are larger than females
- Bill dimensions correlate with mass too

If we only model mass against flipper length, we're missing important information. Multiple regression lets us ask: "what's the effect of flipper length, *after accounting for* species and sex?"

## The Model

We extend our simple model to include multiple predictors:

**y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ...**

Each coefficient now tells us the effect of that variable *while holding the others constant*. This is crucial for interpretation.

## Example: Predicting Penguin Body Mass

Let's build a more complete model for penguin body mass.

```python
from palmerpenguins import load_penguins
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

# Load and prepare data
penguins = load_penguins().dropna()

# Encode categorical variables (species and sex) as numbers
le_species = LabelEncoder()
le_sex = LabelEncoder()
penguins['species_code'] = le_species.fit_transform(penguins['species'])
penguins['sex_code'] = le_sex.fit_transform(penguins['sex'])

print("Species encoding:", dict(zip(le_species.classes_, range(len(le_species.classes_)))))
print("Sex encoding:", dict(zip(le_sex.classes_, range(len(le_sex.classes_)))))

# Check what we have
print(f"\nDataset: {len(penguins)} penguins")
print(penguins[['species', 'sex', 'flipper_length_mm', 'bill_length_mm', 
                'bill_depth_mm', 'body_mass_g']].head())
```

### Check Correlations First

```python
# Which variables correlate with body mass?
numeric_cols = ['flipper_length_mm', 'bill_length_mm', 'bill_depth_mm', 
                'body_mass_g', 'species_code', 'sex_code']
                
print("\nCorrelations with body mass:")
print(penguins[numeric_cols].corr()['body_mass_g'].sort_values(ascending=False))
```

### Fitting the Multiple Regression

```python
# Prepare features and target
X = penguins[['flipper_length_mm', 'bill_length_mm', 'bill_depth_mm', 
              'species_code', 'sex_code']]
y = penguins['body_mass_g']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train)} penguins")
print(f"Test set: {len(X_test)} penguins")

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Look at coefficients
print("\nModel coefficients:")
print(f"  Intercept: {model.intercept_:.1f}")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.2f}")
```

## Interpreting the Results

What do these coefficients actually mean?

```python
print("""
Interpreting the coefficients:

flipper_length_mm: Each additional mm of flipper length adds about 
  17g to body mass, after controlling for other variables. Note this 
  is smaller than in simple regression (~50g) because some of that 
  effect was actually due to species differences.

bill_length_mm: Longer bills are associated with slightly higher mass,
  holding other variables constant.

bill_depth_mm: Deeper bills are associated with higher mass. This makes
  sense - it's a measure of overall head size.

species_code: The coefficient shows average difference between species
  (encoded as 0, 1, 2). Interpretation is tricky with encoded categories.

sex_code: Males (coded as 1) are heavier than females (coded as 0) by
  about this many grams, controlling for body measurements.
""")
```

### How Good Is Our Model?

```python
# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
r2 = metrics.r2_score(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
mae = metrics.mean_absolute_error(y_test, y_pred)

print(f"R-squared: {r2:.3f}")
print(f"RMSE: {rmse:.1f} g")
print(f"MAE: {mae:.1f} g")

# Plot predicted vs observed
fig = px.scatter(x=y_test, y=y_pred,
                 labels={'x': 'Observed Mass (g)', 'y': 'Predicted Mass (g)'})
fig.add_scatter(x=[y_test.min(), y_test.max()], 
                y=[y_test.min(), y_test.max()],
                mode='lines', name='1:1 line',
                line=dict(dash='dash', color='red'))
fig.update_layout(template='simple_white',
                  title=f'Multiple Regression: R² = {r2:.3f}')
fig.show()
```

## Does Adding Variables Help?

Let's compare models with different numbers of predictors:

```python
results = []

# Just flipper length
m1 = LinearRegression().fit(X_train[['flipper_length_mm']], y_train)
results.append({
    'Model': 'Flipper only',
    'R²': m1.score(X_test[['flipper_length_mm']], y_test)
})

# Flipper + species
m2 = LinearRegression().fit(X_train[['flipper_length_mm', 'species_code']], y_train)
results.append({
    'Model': 'Flipper + Species',
    'R²': m2.score(X_test[['flipper_length_mm', 'species_code']], y_test)
})

# Flipper + species + sex
m3 = LinearRegression().fit(X_train[['flipper_length_mm', 'species_code', 'sex_code']], y_train)
results.append({
    'Model': 'Flipper + Species + Sex',
    'R²': m3.score(X_test[['flipper_length_mm', 'species_code', 'sex_code']], y_test)
})

# All predictors
m4 = LinearRegression().fit(X_train, y_train)
results.append({
    'Model': 'All predictors',
    'R²': m4.score(X_test, y_test)
})

print(pd.DataFrame(results))
```

You should see R² improve as you add relevant predictors - species and sex both contribute meaningful information about body mass.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Try It Yourself </h3>
<p>Build a multiple regression model predicting bill length from bill depth, flipper length, species, and sex. 
Which predictors have the strongest effects? Does the bill depth coefficient change from the simple regression?</p>


{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
from palmerpenguins import load_penguins
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

penguins = load_penguins().dropna()

# Encode categoricals
le_species = LabelEncoder()
le_sex = LabelEncoder()
penguins['species_code'] = le_species.fit_transform(penguins['species'])
penguins['sex_code'] = le_sex.fit_transform(penguins['sex'])

# Prepare data
X = penguins[['bill_depth_mm', 'flipper_length_mm', 'species_code', 'sex_code']]
y = penguins['bill_length_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Full model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"R²: {model.score(X_test, y_test):.3f}\n")
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.3f}")

# Compare to simple regression
simple = LinearRegression()
simple.fit(penguins[['bill_depth_mm']], penguins['bill_length_mm'])
print(f"\nSimple regression bill_depth coefficient: {simple.coef_[0]:.3f}")
print(f"Multiple regression bill_depth coefficient: {model.coef_[0]:.3f}")

# The bill_depth coefficient changes dramatically - in simple regression
# it's negative (Simpson's paradox), but in multiple regression 
# controlling for species, it becomes positive as expected.
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

## Limitations

Multiple regression is powerful but has some important limitations:

**It assumes linear relationships.** If the relationship between flipper length and mass is curved (it often is at extremes), a linear model won't capture that properly.

**It assumes additive effects.** The model says the effect of flipper length is the same for all species. In reality, species might differ in their flipper-mass scaling.

**Categorical encoding matters.** We used simple numeric codes for species, but this assumes equal "distances" between categories, which doesn't make biological sense.

These limitations bring us to machine learning approaches, which can handle more complexity.

---

# **Chapter 4: Machine Learning with Random Forests**

Alright, now we're getting to the fun stuff. Machine learning sounds fancy, but the basic idea is simple: let the algorithm figure out the patterns in your data, rather than you specifying them in advance.

## Why Machine Learning?

Ecological relationships are often messy:

- Relationships might be non-linear
- Effects of one variable depend on another (interactions)
- There might be thresholds we didn't anticipate

Machine learning algorithms can discover these patterns automatically. You don't have to know the shape of the relationship beforehand.

## Decision Trees: The Building Block

Before we get to Random Forests, we need to understand decision trees. They're intuitive once you see how they work.

A decision tree is basically a flowchart of questions:

- Is flipper length > 206mm?
  - Yes → Probably a Gentoo, predict ~5000g
  - No → Is bill depth > 18mm?
    - Yes → Probably Adelie, predict ~3700g
    - No → Probably Chinstrap, predict ~3500g

The algorithm figures out which questions to ask and what thresholds to use by looking at your data.

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Using our penguin data
X = penguins[['flipper_length_mm', 'bill_length_mm', 'bill_depth_mm', 
              'species_code', 'sex_code']]
y = penguins['body_mass_g']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a simple tree
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Visualize it
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree for Penguin Body Mass')
plt.tight_layout()
plt.show()

print(f"Decision Tree R²: {tree.score(X_test, y_test):.3f}")
```

Decision trees are easy to interpret - you can literally see the rules. But they have a problem: they tend to overfit. A deep tree can memorize the training data perfectly but fail miserably on new data.

## Random Forests: Many Trees Are Better Than One

Random Forests solve the overfitting problem by building many trees and averaging their predictions. Each tree is a bit different because:

1. Each tree is trained on a random subset of the data (with replacement)
2. At each split, only a random subset of variables is considered

This randomness means individual trees might make mistakes, but averaging over hundreds of trees gives robust predictions.

```python
from sklearn.ensemble import RandomForestRegressor

# Fit a random forest
rf = RandomForestRegressor(
    n_estimators=100,    # number of trees
    max_depth=10,        # how deep each tree can go
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(f"Random Forest R²: {metrics.r2_score(y_test, y_pred):.3f}")
print(f"Random Forest RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.1f} g")
```

## Comparing All Our Methods

Let's see how everything stacks up on the penguin data:

```python
results = []

# Simple regression
simple = LinearRegression()
simple.fit(X_train[['flipper_length_mm']], y_train)
pred = simple.predict(X_test[['flipper_length_mm']])
results.append({
    'Method': 'Simple regression',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

# Multiple regression
multi = LinearRegression()
multi.fit(X_train, y_train)
pred = multi.predict(X_test)
results.append({
    'Method': 'Multiple regression',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

# Decision tree
tree = DecisionTreeRegressor(max_depth=10, random_state=42)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
results.append({
    'Method': 'Decision tree',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

# Random forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
results.append({
    'Method': 'Random forest',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

print(pd.DataFrame(results).to_string(index=False))
```

## What's Driving the Patterns?

One of the nicest things about Random Forests is that they tell you which variables matter most:

```python
importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nVariable importance:")
print(importance.to_string(index=False))

# Plot it
fig = px.bar(importance, x='Importance', y='Variable', orientation='h',
             title='What drives penguin body mass?')
fig.update_layout(template='simple_white', yaxis={'categoryorder': 'total ascending'})
fig.show()
```

For the penguin data, you'll probably find that sex and flipper length are the most important predictors - which makes biological sense!

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}


<h3> Try It Yourself </h3>
<p>Use Random Forest to predict penguin species (as a classification problem) from the morphological 
measurements. Which measurements are most useful for distinguishing species?</p>


<p><strong>Hint:</strong> Use <code>RandomForestClassifier</code> instead of <code>RandomForestRegressor</code></p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

penguins = load_penguins().dropna()

# Prepare data - predict species from measurements
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = penguins['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate
y_pred = rf_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nFeature importance for species classification:")
for name, imp in sorted(zip(X.columns, rf_clf.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.3f}")

# Bill length and bill depth are typically most important -
# they differ most clearly between species
```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

</div>

---

# **5. Gap-filling in Time Series**

Now let's apply what we've learned to a practical problem: dealing with missing data in ecological time series.

## The Problem

If you've worked with field data, you know this frustration. Your sensor died for a week. The battery ran out during the coldest part of winter. Someone accidentally unplugged the datalogger.

Missing data is annoying because:
- You can't calculate annual totals or means
- It messes up time series analyses
- Some statistical methods can't handle NaN values

## Loading Messy Data

Real data often has placeholder values instead of proper missing data markers. Let's see an example:

```python
import pandas as pd
import numpy as np

# Load meteorological data
df = pd.read_parquet('./dwd_ahaus_1996_2023_missing_placeholders.parquet')
df["data_time"] = pd.to_datetime(df["data_time"])

print(f"Temperature range: {df['tair_2m_mean'].min():.1f} to {df['tair_2m_mean'].max():.1f}")
```

If you see -999.99 as the minimum, that's a placeholder for missing data - not an actual temperature! We need to fix this:

```python
# Replace placeholder with NaN
df.loc[df["tair_2m_mean"] == -999.99, "tair_2m_mean"] = np.NaN

# Now check
print(f"Missing values: {df['tair_2m_mean'].isna().sum()}")
```

## Method 1: Linear Interpolation

The simplest approach - just draw a straight line between known values:

```python
# Pandas makes this easy
df['temp_interp'] = df['tair_2m_mean'].interpolate(method='linear')
```

This works fine for short gaps. If temperature was 10°C at noon and 14°C at 2pm, it's reasonable to guess 12°C at 1pm.

But it fails badly for longer gaps. It can't capture the daily temperature cycle - if you have a 24-hour gap, linear interpolation will give you a flat line right through where the daily max and min should be.

## Method 2: Regression-Based Gap Filling

If we have other variables that were measured continuously, we can use them to estimate the missing temperatures:

```python
from sklearn.linear_model import LinearRegression

# Solar radiation and humidity are often available when temperature fails
# (different sensors)
predictors = ['SWIN', 'rH']

# Get data where everything is present (for training)
df_complete = df[['data_time', 'SWIN', 'rH', 'tair_2m_mean']].dropna()

X = df_complete[predictors]
y = df_complete['tair_2m_mean']

# Fit model
model = LinearRegression()
model.fit(X, y)

print(f"Gap-filling model R²: {model.score(X, y):.3f}")
```

Then we can predict temperature wherever we have radiation and humidity data:

```python
# Find rows where temp is missing but predictors exist
mask = df['tair_2m_mean'].isna() & df['SWIN'].notna() & df['rH'].notna()
df.loc[mask, 'temp_regression'] = model.predict(df.loc[mask, predictors])
```

## Method 3: Random Forest Gap Filling

For better accuracy, especially with complex patterns, Random Forest often wins:

```python
from sklearn.ensemble import RandomForestRegressor

# Use more predictors
all_predictors = ['SWIN', 'rH', 'pressure_air', 'wind_speed', 'precipitation']
df_complete = df[all_predictors + ['tair_2m_mean']].dropna()

X = df_complete[all_predictors]
y = df_complete['tair_2m_mean']

rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X, y)

print(f"Random Forest R²: {rf.score(X, y):.3f}")

# Check which predictors matter most for estimating temperature
for name, imp in sorted(zip(all_predictors, rf.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.3f}")
```

## Which Method When?

After working with a lot of gap-filled data, here's what I've found:

**Short gaps (a few hours):** Linear interpolation is usually fine. Temperature doesn't change that fast.

**Medium gaps (a day or two):** Regression with environmental predictors. This captures the daily cycle if you have radiation data.

**Long gaps (weeks+):** Random Forest or similar, but honestly... consider whether you should be filling such long gaps at all. Sometimes it's better to acknowledge the data is missing.

**General advice:**
- Always validate your gap-filling on data where you know the truth
- Flag gap-filled values in your final dataset
- Report the uncertainty or error in your gap-filled values
- Don't over-fill - sometimes missing data should stay missing

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Try It Yourself </h3>
<p>Compare linear interpolation vs. Random Forest for filling a 24-hour gap in temperature data.
Which method captures the daily cycle better?</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
# The key insight is that linear interpolation can't capture 
# diurnal patterns, while Random Forest (using radiation as 
# a predictor) can.

# For a 24-hour gap:
# - Linear interpolation draws a flat line
# - Random Forest predicts warm during day, cool at night
#   (because it learned that high radiation = high temp)

# In my experience, Random Forest reduces RMSE by 30-50% 
# compared to linear interpolation for day-long gaps.

# But for gaps under 3-6 hours, the methods are often similar
# because temperature hasn't changed much anyway.
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

---

# **Wrapping Up**

We've covered a lot of ground here. Let me leave you with the key takeaways:

**Simple regression** is your starting point. It's easy to understand, easy to explain, and often good enough for straightforward questions.

**Multiple regression** lets you account for multiple drivers at once. The coefficients tell you the effect of each variable while controlling for the others.

**Random Forests** can capture complex patterns that regression misses. They're particularly good when you don't know the shape of the relationships in advance.

**For gap-filling**, match your method to your gap length. Simple interpolation for short gaps, model-based methods for longer ones.

**Most importantly:** always plot your data, check your assumptions, and validate on independent test data. No amount of fancy statistics can fix bad data or inappropriate models.

Good luck with your analyses!

## Where to Go From Here

If you want to dig deeper:
- **Generalized Additive Models (GAMs)** let you fit smooth curves instead of straight lines
- **Mixed-effects models** handle hierarchical data (e.g., measurements nested within sites)
- **Gradient Boosting (XGBoost)** often outperforms Random Forests for prediction
- **Time series methods** (ARIMA, etc.) are specifically designed for temporal data

But honestly, you can get surprisingly far with just regression and Random Forests. Master these first before moving on to fancier tools.
