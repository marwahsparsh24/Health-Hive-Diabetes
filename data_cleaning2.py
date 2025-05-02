import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load your data
df = pd.read_csv('diabetic_data.csv')

# Step 1: Clean and separate
df_clean = df.copy()
df.drop(columns=['max_glu_serum','A1Cresult'], inplace = True) 
df_known = df_clean[df_clean['weight'] != '?'].copy()
df_unknown = df_clean[df_clean['weight'] == '?'].copy()

# Step 2: Encode the target variable (weight categories)
le_weight = LabelEncoder()
df_known['weight_encoded'] = le_weight.fit_transform(df_known['weight'])

# Step 3: Define features to use for prediction
features = [
    'age', 'gender', 'race', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_medications'
]

# Step 4: Encode categorical features
X_train = df_known[features].copy()
for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))

X_unknown = df_unknown[features].copy()
for col in X_unknown.select_dtypes(include='object').columns:
    X_unknown[col] = LabelEncoder().fit_transform(X_unknown[col].astype(str))

# Step 5: Train Decision Tree Classifier
y_train = df_known['weight_encoded']
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Step 6: Predict missing weights
predicted_weights = clf.predict(X_unknown)
predicted_labels = le_weight.inverse_transform(predicted_weights)

# Step 7: Assign predictions and merge
df_unknown['weight'] = predicted_labels
df_final = pd.concat([df_known.drop(columns='weight_encoded'), df_unknown], ignore_index=True)

# Step 8: Save result
df_final.to_csv('diabetic_data_imputed.csv', index=False)
print("✅ Imputation complete. File saved as 'diabetic_data_imputed.csv'")

