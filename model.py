import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb

# ---------------- Load data ----------------
data = pd.read_csv("internship.csv", encoding='latin1')

# --- Clean stipend ---
data['stipend'] = data['stipend'].apply(lambda x: '0' if 'Unpaid' in str(x) else x)
data['stipend'] = data['stipend'].str.replace("/month", "").str.replace("lump sum", "").str.replace("₹", "")

data['Other Benifits'] = data['stipend'].str.split('+').apply(lambda x: x[-1].strip() if len(x) > 1 else None)
data['stipend'] = data['stipend'].str.split('+').str[0].str.strip()

data['min stipend'] = data['stipend'].str.split("-").str[0].str.extract(r'([\d,]+)', expand=False).str.replace(",", "")
data['max stipend'] = data['stipend'].str.split("-").str[-1].str.extract(r'([\d,]+)', expand=False).str.replace(",", "")
data[['min stipend','max stipend']] = data[['min stipend','max stipend']].apply(pd.to_numeric, errors='coerce')

# --- Convert duration to months ---
def weeks_to_months(duration):
    if isinstance(duration, str) and 'Weeks' in duration:
        weeks = int(duration.split()[0])
        return weeks / 4.33
    try:
        return float(duration.split()[0])
    except:
        return None

data['duration'] = data['duration'].apply(weeks_to_months)

# ---------------- Feature encoding ----------------
categorical_cols = ['Sector', 'location']
encoders = {col: LabelEncoder().fit(data[col].astype(str)) for col in categorical_cols}
for col, encoder in encoders.items():
    data[col] = encoder.transform(data[col].astype(str))

# ---------------- Label creation ----------------
if "applied" in data.columns:
    # ✅ Real labels available
    print("Using real user-applied labels...")
    data['label'] = data['applied']
else:
    # ⚡ Simulated labels
    print("Using simulated stipend+duration labels...")
    median_stipend = data['max stipend'].median()
    median_duration = data['duration'].median()
    data['label'] = ((data['max stipend'] >= median_stipend) & 
                     (data['duration'] >= median_duration)).astype(int)

# ---------------- Model training ----------------
features = ['Sector', 'location', 'min stipend', 'max stipend', 'duration']
X = data[features].fillna(0)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100)
model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred)

print("\n📊 Model Evaluation")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("Classification Report:\n", report)

# ---------------- User query ----------------
Education = input("Enter your education: ")
Skills = input("Enter your skills: ")
sector = input("Enter sector: ")
location = input("Enter location: ")

# Encode inputs (fallback to 0 if unseen)
user_sector = encoders['Sector'].transform([sector])[0] if sector in encoders['Sector'].classes_ else 0
user_location = encoders['location'].transform([location])[0] if location in encoders['location'].classes_ else 0

# ---------------- Scoring internships ----------------
data['score'] = model.predict_proba(X)[:, 1]
top_internships = data.sort_values(by='score', ascending=False).head(3)

# ---------------- Map back encoded values ----------------
# Reverse mapping for categorical columns
for col, encoder in encoders.items():
    reverse_map = {i: label for i, label in enumerate(encoder.classes_)}
    top_internships[col] = top_internships[col].map(reverse_map)

# ---------------- Save & print ----------------
columns_to_show = ['Sector', 'location', 'min stipend', 'max stipend', 'duration', 'score']
best_internships_json = top_internships[columns_to_show].to_dict(orient="records")

with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(best_internships_json, indent=4, ensure_ascii=False))

print("\n🎯 Recommended Internships:")
print(json.dumps(best_internships_json, indent=4, ensure_ascii=False))

