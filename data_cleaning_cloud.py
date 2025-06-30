import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# S3 paths
# -------------------------------
bucket = "your-bucket-name"
input_path = f"s3://{bucket}/data/diabetic_data.csv"
output_path = f"s3://{bucket}/data/cleaned_diabetes_data.csv"

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(input_path)

# -------------------------------
# Check initial missing values
# -------------------------------
print("Initial Missing Values:")
print(df.isnull().sum())

# -------------------------------
# Convert range strings to midpoints
# -------------------------------
def range_to_midpoint(value):
    if pd.isna(value) or value == '?':
        return np.nan
    value = value.strip('[]()').replace('+', '').split('-')
    try:
        return (int(value[0]) + int(value[1])) / 2
    except:
        return np.nan

df['age_mid'] = df['age'].apply(range_to_midpoint)
df['weight_mid'] = df['weight'].apply(range_to_midpoint)

# -------------------------------
# Encode categorical features
# -------------------------------
df['gender_enc'] = pd.factorize(df['gender'])[0]
df['race_enc'] = pd.factorize(df['race'])[0]

# -------------------------------
# Impute missing weight
# -------------------------------
df_train = df[(~df['age_mid'].isna()) & (~df['weight_mid'].isna())]
X_train = df_train[['age_mid', 'gender_enc', 'race_enc']]
y_train = df_train['weight_mid']

rf_weight = RandomForestRegressor(n_estimators=100, random_state=42)
rf_weight.fit(X_train, y_train)

df_missing_weight = df[(df['weight'] == '?') & (~df['age_mid'].isna())].copy()
X_missing_weight = df_missing_weight[['age_mid', 'gender_enc', 'race_enc']]
df_missing_weight['weight_mid_predicted'] = rf_weight.predict(X_missing_weight)

def midpoint_to_range(midpoint):
    bins = list(range(0, 200, 10))
    for i in range(len(bins)-1):
        if bins[i] <= midpoint < bins[i+1]:
            return f"({bins[i]}-{bins[i+1]})"
    return f"({bins[-1]}+)"

df_missing_weight['weight_imputed'] = df_missing_weight['weight_mid_predicted'].apply(midpoint_to_range)

df['weight_imputed'] = df['weight']
df.loc[df_missing_weight.index, 'weight_imputed'] = df_missing_weight['weight_imputed']

# -------------------------------
# Impute medical_specialty
# -------------------------------
top_specialties = df['medical_specialty'].value_counts().nlargest(15).index.tolist()
df['medical_specialty_simplified'] = df['medical_specialty'].apply(
    lambda x: x if x in top_specialties else 'Other'
)

le_specialty = LabelEncoder()
df['specialty_label'] = le_specialty.fit_transform(df['medical_specialty_simplified'])

df_known_spec = df[df['medical_specialty'] != '?'].copy()
X_spec_train = df_known_spec[['age_mid', 'gender_enc', 'race_enc', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']]
y_spec_train = df_known_spec['specialty_label']

rf_spec = RandomForestClassifier(n_estimators=50, random_state=42)
rf_spec.fit(X_spec_train, y_spec_train)

df_missing_spec = df[df['medical_specialty'] == '?'].copy()
X_spec_missing = df_missing_spec[['age_mid', 'gender_enc', 'race_enc', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']]
y_spec_pred = rf_spec.predict(X_spec_missing)

df.loc[df_missing_spec.index, 'medical_specialty_imputed'] = le_specialty.inverse_transform(y_spec_pred)
df.loc[df_known_spec.index, 'medical_specialty_imputed'] = df.loc[df_known_spec.index, 'medical_specialty_simplified']

# -------------------------------
# Clean-up: drop helpers and rename
# -------------------------------
cols_to_drop = [
    'weight', 'medical_specialty', 'age_mid', 'weight_mid',
    'gender_enc', 'race_enc', 'medical_specialty_simplified'
]
df.drop(columns=cols_to_drop, inplace=True)

df.rename(columns={
    'weight_imputed': 'weight',
    'medical_specialty_imputed': 'medical_specialty'
}, inplace=True)

# -------------------------------
# Final filters
# -------------------------------
df = df.dropna(subset=['A1Cresult'])
df = df.drop('max_glu_serum', axis=1)

# -------------------------------
# Final missing check
# -------------------------------
print("Final Missing Values:")
print(df.isnull().sum())

# -------------------------------
# Save cleaned data to S3
# -------------------------------
df.to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to {output_path}")
