import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
df = pd.read_csv("data/diabetic_data.csv")  # change path as needed

# Step 2: Helper to convert ranges like '[30-40)' to midpoint
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

# Step 3: Encode categorical features
df['gender_enc'] = pd.factorize(df['gender'])[0]
df['race_enc'] = pd.factorize(df['race'])[0]

# ----------------------------------------------------------------
# 🎯 PART 1: IMPUTE MISSING WEIGHT BASED ON AGE, RACE, GENDER
# ----------------------------------------------------------------

# Training data (valid age + weight)
df_train = df[(~df['age_mid'].isna()) & (~df['weight_mid'].isna())]
X_train = df_train[['age_mid', 'gender_enc', 'race_enc']]
y_train = df_train['weight_mid']

# Train RandomForest to predict weight
rf_weight = RandomForestRegressor(n_estimators=100, random_state=42)
rf_weight.fit(X_train, y_train)

# Predict missing weights
df_missing_weight = df[(df['weight'] == '?') & (~df['age_mid'].isna())].copy()
X_missing_weight = df_missing_weight[['age_mid', 'gender_enc', 'race_enc']]
df_missing_weight['weight_mid_predicted'] = rf_weight.predict(X_missing_weight)

# Map midpoints to range
def midpoint_to_range(midpoint):
    bins = list(range(0, 200, 10))
    for i in range(len(bins)-1):
        if bins[i] <= midpoint < bins[i+1]:
            return f"({bins[i]}-{bins[i+1]})"
    return f"({bins[-1]}+)"

df_missing_weight['weight_imputed'] = df_missing_weight['weight_mid_predicted'].apply(midpoint_to_range)

# Add weight_imputed column to original dataset
df['weight_imputed'] = df['weight']
df.loc[df_missing_weight.index, 'weight_imputed'] = df_missing_weight['weight_imputed']

# ----------------------------------------------------------------
# 🎯 PART 2: IMPUTE medical_specialty USING SIMPLIFIED LABELS
# ----------------------------------------------------------------

# Simplify specialty to top 15 classes + "Other"
top_specialties = df['medical_specialty'].value_counts().nlargest(15).index.tolist()
df['medical_specialty_simplified'] = df['medical_specialty'].apply(
    lambda x: x if x in top_specialties else 'Other'
)

# Encode simplified labels
le_specialty = LabelEncoder()
df['specialty_label'] = le_specialty.fit_transform(df['medical_specialty_simplified'])

# Train classifier on known specialties
df_known_spec = df[df['medical_specialty'] != '?'].copy()
X_spec_train = df_known_spec[['age_mid', 'gender_enc', 'race_enc', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']]
y_spec_train = df_known_spec['specialty_label']

rf_spec = RandomForestClassifier(n_estimators=50, random_state=42)
rf_spec.fit(X_spec_train, y_spec_train)

# Predict for missing specialties
df_missing_spec = df[df['medical_specialty'] == '?'].copy()
X_spec_missing = df_missing_spec[['age_mid', 'gender_enc', 'race_enc', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']]
y_spec_pred = rf_spec.predict(X_spec_missing)
df.loc[df_missing_spec.index, 'medical_specialty_imputed'] = le_specialty.inverse_transform(y_spec_pred)

# For known values, retain original
df.loc[df_known_spec.index, 'medical_specialty_imputed'] = df.loc[df_known_spec.index, 'medical_specialty_simplified']

# ----------------------------------------------------------------
# 🧹  CLEAN-UP: drop helper/original columns & rename imputed cols
# ----------------------------------------------------------------
cols_to_drop = [
    'weight',                       
    'medical_specialty',            
    'age_mid', 'weight_mid',        
    'gender_enc', 'race_enc',       
    'medical_specialty_simplified' 
]

#CLEAN DATASET AND EXPORT FILE 

# 1) Drop the columns we no longer need
df.drop(columns=cols_to_drop, inplace=True)

# 2) Rename imputed columns to take the place of the originals
df.rename(
    columns={
        'weight_imputed': 'weight',
        'medical_specialty_imputed': 'medical_specialty'
    },
    inplace=True
)

df.to_csv("data/cleaned_diabetes_data.csv", index=False)

#df.to_excel("diabetic_data_cleaned_with_imputations1.xlsx",index=False)