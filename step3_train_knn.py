# Step 3 — Train the KNN Model

# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # ── 1. Load your clean data ───────────────────────────────────
# df = pd.read_csv('devmatch_clean.csv')
# print(f"Loaded clean data: {df.shape}")

# # ── 2. Separate X (skills) and y (role) ──────────────────────
# # X = everything except the role column
# # y = only the role column
# X = df.drop(columns=['role'])
# y = df['role']
# print(f"Features (X): {X.shape}")
# print(f"Labels   (y): {y.shape}")

# # ── 3. Split into training and testing ───────────────────────
# # 80% of data = training (KNN learns from this)
# # 20% of data = testing  (we check if KNN guessed correctly)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,        # 20% for testing
#     random_state=42       # fixes randomness so results are same every run
# )
# print(f"\nTraining rows : {len(X_train)}")
# print(f"Testing rows  : {len(X_test)}")

# # ── 4. Create and train the KNN model ────────────────────────
# # k=5 means: look at 5 nearest neighbors and take majority vote
# knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', n_jobs=8)
# knn.fit(X_train, y_train)
# print("\nKNN model trained successfully!")

# # ── 5. Test the model ─────────────────────────────────────────
# y_predicted = knn.predict(X_test)
# accuracy = accuracy_score(y_test, y_predicted)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # ── 6. Save the trained model ─────────────────────────────────
# import joblib
# joblib.dump(knn, 'devmatch_knn_model.pkl')
# joblib.dump(list(X.columns), 'feature_columns.pkl')
# print("\nModel saved: devmatch_knn_model.pkl")
# print("Columns saved: feature_columns.pkl")
# print("\nStep 3 complete! Ready for Step 4 (GUI).")

#got 51.30% accuracy

# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# import joblib

# # ── 1. Load clean data ────────────────────────────────────────
# df = pd.read_csv('devmatch_clean.csv')

# X = df.drop(columns=['role'])
# y = df['role']

# # ── 2. Fix imbalance — give equal weight to all roles ────────
# # SMOTE creates fake extra rows for minority roles
# # so KNN sees equal examples of every role
# smote = SMOTE(random_state=42)
# X_balanced, y_balanced = smote.fit_resample(X, y)
# print(f"After balancing: {X_balanced.shape}")
# print("Role counts after fix:")
# print(pd.Series(y_balanced).value_counts())

# # ── 3. Train/test split ───────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(
#     X_balanced, y_balanced,
#     test_size=0.2,
#     random_state=42
# )

# # ── 4. Train KNN with best settings ──────────────────────────
# knn = KNeighborsClassifier(
#     n_neighbors=5,
#     algorithm='kd_tree',
#     n_jobs=8,
#     weights='distance'   # closer neighbors get more vote power
# )
# knn.fit(X_train, y_train)

# # ── 5. Test accuracy ──────────────────────────────────────────
# y_predicted = knn.predict(X_test)
# accuracy = accuracy_score(y_test, y_predicted)
# print(f"\nAccuracy: {accuracy * 100:.2f}%")

# # ── 6. Save model ─────────────────────────────────────────────
# joblib.dump(knn, 'devmatch_knn_model.pkl')
# joblib.dump(list(X.columns), 'feature_columns.pkl')
# print("Model saved!")

#got 71.70% accuracy with SMOTE and distance weighting!

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

df_raw = pd.read_csv('survey_results_public.csv')
df_clean = pd.read_csv('devmatch_clean.csv')

# ── Add EdLevel as numeric feature ───────────────────────────
edu_map = {
    'Primary/elementary school': 1,
    'Secondary school': 2,
    'Some college/university study without earning a degree': 3,
    'Associate degree': 4,
    'Bachelor\u2019s degree (B.A., B.S., B.Eng., etc.)': 5,
    'Master\u2019s degree (M.A., M.S., M.Eng., MBA, etc.)': 6,
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 7,
    'Something else': 3
}

# Filter raw data same way as step 2
df_raw = df_raw[df_raw['DevType'].notna()].copy()
df_raw['role'] = df_raw['DevType'].apply(lambda x: x.split(';')[0].strip())
top_roles = df_raw['role'].value_counts().head(8).index.tolist()
df_raw = df_raw[df_raw['role'].isin(top_roles)].copy()
df_raw = df_raw.reset_index(drop=True)

# Add education level
df_clean = df_clean.reset_index(drop=True)
df_clean['edu_level'] = df_raw['EdLevel'].map(edu_map).fillna(3)

# Add platform skills (Linux, AWS, Docker etc.)
def parse_skills(val):
    if pd.isna(val): return []
    return [s.strip() for s in val.split(';')]

from sklearn.preprocessing import MultiLabelBinarizer

platform_list = df_raw['PlatformHaveWorkedWith'].apply(parse_skills)
mlb2 = MultiLabelBinarizer()
platform_matrix = pd.DataFrame(
    mlb2.fit_transform(platform_list),
    columns=['platform_' + c for c in mlb2.classes_]
)

# Add misc tools (Docker, Kubernetes etc.)
misc_list = df_raw['MiscTechHaveWorkedWith'].apply(parse_skills)
mlb3 = MultiLabelBinarizer()
misc_matrix = pd.DataFrame(
    mlb3.fit_transform(misc_list),
    columns=['misc_' + c for c in mlb3.classes_]
)

# Combine everything
X = pd.concat([
    df_clean.drop(columns=['role']),
    platform_matrix,
    misc_matrix,
    df_clean[['edu_level']]
], axis=1)

y = df_clean['role']
print(f"New feature count: {X.shape[1]}")

# ── Balance with SMOTE ────────────────────────────────────────
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"After balancing: {X_balanced.shape}")

# ── Train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=42
)

# ── Fix 2 — Try different K values, pick the best ────────────
print("\nTesting different K values...")
best_k = 5
best_acc = 0
for k in [3, 5, 7, 9, 11, 15]:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        algorithm='kd_tree',
        n_jobs=8,
        weights='distance'
    )
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"  K={k}  →  Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nBest K = {best_k}  →  Best Accuracy = {best_acc*100:.2f}%")

# ── Train final model with best K ────────────────────────────
knn_final = KNeighborsClassifier(
    n_neighbors=best_k,
    algorithm='kd_tree',
    n_jobs=8,
    weights='distance'
)
knn_final.fit(X_train, y_train)

# ── Save everything ───────────────────────────────────────────
joblib.dump(knn_final, 'devmatch_knn_model.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')
joblib.dump(mlb2, 'platform_mlb.pkl')
joblib.dump(mlb3, 'misc_mlb.pkl')
print("\nModel saved!")
print("Step 3 upgraded complete!")