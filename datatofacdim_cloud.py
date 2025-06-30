import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Define S3 paths
bucket = "health-hive-diabetes-dataset"
input_data_path = f"s3://{bucket}/data/"
output_dim_path = f"s3://{bucket}/star_schema/dimension_tables/"
output_fact_path = f"s3://{bucket}/star_schema/fact_tables/"

# -------------------------------
# Load cleaned dataset and mapping
# -------------------------------
df = pd.read_csv(input_data_path + "cleaned_diabetes_data.csv")
df.columns = df.columns.str.strip()

mapping_raw = pd.read_csv(input_data_path + "IDS_mapping.csv", header=None)

# ----------------------------
# Extract ID mappings
# ----------------------------
admission_type_start = mapping_raw[mapping_raw[0] == "admission_type_id"].index[0] + 1
discharge_start = mapping_raw[mapping_raw[0] == "discharge_disposition_id"].index[0]

admission_type = mapping_raw.iloc[admission_type_start:discharge_start].dropna()
admission_type.columns = ['admission_type_id', 'admission_type_description']
admission_type['admission_type_id'] = pd.to_numeric(admission_type['admission_type_id'], errors='coerce').astype('Int64')

discharge_start += 1
admission_source_start = mapping_raw[mapping_raw[0] == "admission_source_id"].index[0]

discharge = mapping_raw.iloc[discharge_start:admission_source_start].dropna()
discharge.columns = ['discharge_disposition_id', 'discharge_description']
discharge['discharge_disposition_id'] = pd.to_numeric(discharge['discharge_disposition_id'], errors='coerce').astype('Int64')

admission_source_start += 1
admission_source = mapping_raw.iloc[admission_source_start:].dropna()
admission_source.columns = ['admission_source_id', 'source_description']
admission_source['admission_source_id'] = pd.to_numeric(admission_source['admission_source_id'], errors='coerce').astype('Int64')

df['admission_type_id'] = pd.to_numeric(df['admission_type_id'], errors='coerce').astype('Int64')
df['discharge_disposition_id'] = pd.to_numeric(df['discharge_disposition_id'], errors='coerce').astype('Int64')
df['admission_source_id'] = pd.to_numeric(df['admission_source_id'], errors='coerce').astype('Int64')

# -------------------------------
# DIMENSION TABLES
# -------------------------------
dim_patient = df[['patient_nbr', 'race', 'gender', 'age']].drop_duplicates().rename(columns={
    'patient_nbr': 'patient_id'
})

dim_admission_type = df[['admission_type_id']].drop_duplicates()
dim_admission_type = dim_admission_type.merge(admission_type, on='admission_type_id', how='left')

dim_discharge_disposition = df[['discharge_disposition_id']].drop_duplicates()
dim_discharge_disposition = dim_discharge_disposition.merge(discharge, on='discharge_disposition_id', how='left')

dim_admission_source = df[['admission_source_id']].drop_duplicates()
dim_admission_source = dim_admission_source.merge(admission_source, on='admission_source_id', how='left')

df.rename(columns={'medical_specialty_imputed': 'medical_specialty'}, inplace=True)
dim_medical_specialty = df[['medical_specialty']].drop_duplicates().dropna().reset_index(drop=True)
dim_medical_specialty['medical_specialty_id'] = dim_medical_specialty.index + 1
df = df.merge(dim_medical_specialty, on='medical_specialty', how='left')

drug_columns = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone"
]

drug_data = df[drug_columns].copy()
dim_drug = drug_data.melt(var_name="drug_name", value_name="response").drop_duplicates().dropna().reset_index(drop=True)
dim_drug.insert(0, "drug_id", range(1, len(dim_drug) + 1))

# -------------------------------
# FACT TABLES
# -------------------------------
fact_hospital_encounter = df[[
    'encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'medical_specialty_id', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'A1Cresult', 'change', 'diabetesMed', 'readmitted'
]].rename(columns={'patient_nbr': 'patient_id'})

drug_df = df[["encounter_id"] + drug_columns].copy()
fact_drug_response = drug_df.melt(id_vars="encounter_id", value_vars=drug_columns, var_name="drug_name", value_name="response")
fact_drug_response = fact_drug_response.dropna().reset_index(drop=True)

# -------------------------------
# EXPORT TO S3 (CSV)
# -------------------------------
dim_patient.to_csv(output_dim_path + "dim_patient.csv", index=False)
dim_admission_type.to_csv(output_dim_path + "dim_admission_type.csv", index=False)
dim_discharge_disposition.to_csv(output_dim_path + "dim_discharge_disposition.csv", index=False)
dim_admission_source.to_csv(output_dim_path + "dim_admission_source.csv", index=False)
dim_medical_specialty.to_csv(output_dim_path + "dim_medical_specialty.csv", index=False)
dim_drug.to_csv(output_dim_path + "dim_drug.csv", index=False)

fact_hospital_encounter.to_csv(output_fact_path + "fact_hospital_encounter.csv", index=False)
fact_drug_response.to_csv(output_fact_path + "fact_drug_response.csv", index=False)

print("✅ All dimension and fact tables exported to S3.")
