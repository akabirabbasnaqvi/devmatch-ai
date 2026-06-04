# Step 1 — Explore & Clean the Dataset

import pandas as pd

# ── 1. Load the data ──────────────────────────────────────────
df = pd.read_csv('survey_results_public.csv')
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# ── 2. See all column names ───────────────────────────────────
print("\nAll columns:")
for col in df.columns:
    print(" -", col)

# ── 3. Check the columns we need ─────────────────────────────
important_cols = [
    'DevType',                    # Target label (role)
    'LanguageHaveWorkedWith',     # Python, JS, etc.
    'FrameworkHaveWorkedWith',    # React, Django, etc.
    'DatabaseHaveWorkedWith',     # SQL, MongoDB, etc.
    'YearsCodePro',               # Experience (years)
    'EdLevel',                    # Education
    'Country'                     # Country
]

print("\n--- Sample values from important columns ---")
for col in important_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].dropna().head(3).tolist())
    else:
        print(f"\n{col}: *** NOT FOUND — check schema CSV ***")

# ── 4. Check missing values ───────────────────────────────────
print("\n--- Missing values (%) ---")
missing = df[important_cols].isnull().mean() * 100
print(missing.round(1))

# ── 5. Check DevType (your KNN target label) ──────────────────
print("\n--- Top 15 Developer Roles (DevType) ---")
roles = df['DevType'].dropna().str.split(';').explode()
print(roles.value_counts().head(15))