#Step 2 — Clean & Build Your Feature Matrix



import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv('survey_results_public.csv')
print(f"Loaded: {df.shape}")

# ── 2. Fix column name (2024 survey rename) ───────────────────
# The correct framework column in 2024 is:
df['FrameworkHaveWorkedWith'] = df['WebframeHaveWorkedWith']

# ── 3. Keep only rows where DevType is not empty ──────────────
df = df[df['DevType'].notna()].copy()
print(f"After removing empty DevType: {len(df)} rows")

# ── 4. Simplify DevType — take only the FIRST role listed ─────
df['role'] = df['DevType'].apply(lambda x: x.split(';')[0].strip())

# ── 5. Keep only the top 8 most common roles ─────────────────
top_roles = df['role'].value_counts().head(8).index.tolist()
df = df[df['role'].isin(top_roles)].copy()
print(f"After filtering top 8 roles: {len(df)} rows")
print("\nRoles kept:")
print(df['role'].value_counts())

# ── 6. Parse skill columns into lists ────────────────────────
def parse_skills(val):
    if pd.isna(val):
        return []
    return [s.strip() for s in val.split(';')]

df['lang_list']  = df['LanguageHaveWorkedWith'].apply(parse_skills)
df['db_list']    = df['DatabaseHaveWorkedWith'].apply(parse_skills)
df['frame_list'] = df['WebframeHaveWorkedWith'].apply(parse_skills)

# Combine all skills into one list per developer
df['all_skills'] = df['lang_list'] + df['db_list'] + df['frame_list']

# ── 7. Convert skills to binary columns (0/1 per skill) ───────
mlb = MultiLabelBinarizer()
skill_matrix = pd.DataFrame(
    mlb.fit_transform(df['all_skills']),
    columns=mlb.classes_,
    index=df.index
)
print(f"\nTotal unique skills found: {len(mlb.classes_)}")
print("Sample skills:", list(mlb.classes_)[:15])

# ── 8. Add experience as numeric feature ─────────────────────
df['exp'] = pd.to_numeric(df['YearsCodePro'], errors='coerce').fillna(0)
skill_matrix['experience'] = df['exp'].values

# ── 9. Final dataset ──────────────────────────────────────────
X = skill_matrix                  # Features (input to KNN)
y = df['role'].values             # Labels (what KNN predicts)

print(f"\nFinal X shape: {X.shape}")
print(f"Final y shape: {y.shape}")

# ── 10. Save cleaned data ─────────────────────────────────────
skill_matrix['role'] = y
skill_matrix.to_csv('devmatch_clean.csv', index=False)
print("\nSaved: devmatch_clean.csv")
print("Step 2 complete! Ready for KNN training.")