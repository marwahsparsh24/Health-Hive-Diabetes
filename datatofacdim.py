import pandas as pd
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_diabetes_data.csv")
df.columns = df.columns.str.strip()

# Load raw mapping file (stacked vertical format)
mapping_raw = pd.read_csv("data/IDS_mapping.csv", header=None)
#mapping_raw.columns = mapping_raw.columns.str.strip()

# ----------------------------
# Extract admission_type mapping
# ----------------------------
admission_type_start = mapping_raw[mapping_raw[0] == "admission_type_id"].index[0] + 1
discharge_start = mapping_raw[mapping_raw[0] == "discharge_disposition_id"].index[0]

admission_type = mapping_raw.iloc[admission_type_start:discharge_start].dropna()
admission_type.columns = ['admission_type_id', 'admission_type_description']
admission_type['admission_type_id'] = pd.to_numeric(admission_type['admission_type_id'], errors='coerce').astype('Int64')

# ----------------------------
# Extract discharge_disposition mapping
# ----------------------------
discharge_start += 1
admission_source_start = mapping_raw[mapping_raw[0] == "admission_source_id"].index[0]

discharge = mapping_raw.iloc[discharge_start:admission_source_start].dropna()
discharge.columns = ['discharge_disposition_id', 'discharge_description']
discharge['discharge_disposition_id'] = pd.to_numeric(discharge['discharge_disposition_id'], errors='coerce').astype('Int64')

# ----------------------------
# Extract admission_source mapping
# ----------------------------
admission_source_start += 1
admission_source = mapping_raw.iloc[admission_source_start:].dropna()
admission_source.columns = ['admission_source_id', 'source_description']
admission_source['admission_source_id'] = pd.to_numeric(admission_source['admission_source_id'], errors='coerce').astype('Int64')

# ----------------------------
# Ensure main df ID columns are also Int64
# ----------------------------
df['admission_type_id'] = pd.to_numeric(df['admission_type_id'], errors='coerce').astype('Int64')
df['discharge_disposition_id'] = pd.to_numeric(df['discharge_disposition_id'], errors='coerce').astype('Int64')
df['admission_source_id'] = pd.to_numeric(df['admission_source_id'], errors='coerce').astype('Int64')

# -------------------------------
# DIMENSION TABLES
# -------------------------------

# dim_patient
dim_patient = df[['patient_nbr', 'race', 'gender', 'age']].drop_duplicates().rename(columns={
    'patient_nbr': 'patient_id'
})

# dim_admission_type
dim_admission_type = df[['admission_type_id']].drop_duplicates()
dim_admission_type = dim_admission_type.merge(admission_type, on='admission_type_id', how='left')

# dim_discharge_disposition
dim_discharge_disposition = df[['discharge_disposition_id']].drop_duplicates()
dim_discharge_disposition = dim_discharge_disposition.merge(discharge, on='discharge_disposition_id', how='left')

# dim_admission_source
dim_admission_source = df[['admission_source_id']].drop_duplicates()
dim_admission_source = dim_admission_source.merge(admission_source, on='admission_source_id', how='left')

# dim_medical_specialty
dim_medical_specialty = df[['medical_specialty']].drop_duplicates().dropna().reset_index(drop=True)
dim_medical_specialty['medical_specialty_id'] = dim_medical_specialty.index + 1
df = df.merge(dim_medical_specialty, on='medical_specialty', how='left')

# -------------------------------
# FACT TABLE
# -------------------------------

fact_hospital_encounter = df[[
    'encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'medical_specialty_id', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'readmitted'
]].rename(columns={'patient_nbr': 'patient_id'})

# -------------------------------
# EXPORT TO CORRECT FOLDERS
# -------------------------------

os.makedirs("data/dimension_tables", exist_ok=True)
os.makedirs("data/fact_table", exist_ok=True)

dim_patient.to_csv("data/dimension_tables/dim_patient.csv", index=False)
dim_admission_type.to_csv("data/dimension_tables/dim_admission_type.csv", index=False)
dim_discharge_disposition.to_csv("data/dimension_tables/dim_discharge_disposition.csv", index=False)
dim_admission_source.to_csv("data/dimension_tables/dim_admission_source.csv", index=False)
dim_medical_specialty.to_csv("data/dimension_tables/dim_medical_specialty.csv", index=False)
fact_hospital_encounter.to_csv("data/fact_table/fact_hospital_encounter.csv", index=False)

print("✅ All dimension and fact tables saved in 'data/' folder.")
