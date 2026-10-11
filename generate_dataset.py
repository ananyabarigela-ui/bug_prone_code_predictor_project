import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Number of samples
n = 500

# Generate code complexity features
loc = np.random.randint(5, 150, n)
functions = np.random.randint(1, 10, n)
loops = np.random.randint(0, 10, n)
conditions = np.random.randint(0, 15, n)
variables = np.random.randint(1, 20, n)
comment_ratio = np.round(np.random.uniform(0.0, 0.5, n), 2)
nesting = np.random.randint(0, 8, n)

# Calculate an educational complexity score
complexity_score = (
    0.02 * loc
    + 0.25 * functions
    + 0.40 * loops
    + 0.35 * conditions
    + 0.15 * variables
    + 0.80 * nesting
    - 1.5 * comment_ratio
)

# Add some randomness
noise = np.random.normal(0, 1.5, n)

score = complexity_score + noise

# Use median as the classification threshold
threshold = np.median(score)

# 0 = Not Bug-Prone
# 1 = Bug-Prone
bug = (score > threshold).astype(int)

# Create dataset
data = pd.DataFrame({
    "loc": loc,
    "functions": functions,
    "loops": loops,
    "conditions": conditions,
    "variables": variables,
    "comment_ratio": comment_ratio,
    "nesting": nesting,
    "bug": bug
})

# Save dataset
data.to_csv("data/bug_dataset.csv", index=False)

# Display information
print("======================================")
print("DATASET CREATED SUCCESSFULLY!")
print("======================================")

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nClass distribution:")
print(data["bug"].value_counts())

print("\nDataset saved to:")
print("data/bug_dataset.csv")