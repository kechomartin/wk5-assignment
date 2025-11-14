# Data Dictionary

## Patient Demographics
- `patient_id`: Unique patient identifier (string)
- `age`: Patient age in years (int, 18-120)
- `gender`: Patient gender (categorical: Male, Female, Other)
- `insurance_type`: Insurance coverage (categorical: Medicare, Medicaid, Private, None)

## Clinical Features
- `admission_type`: Type of admission (categorical: Emergency, Urgent, Elective)
- `diagnosis_primary`: Primary diagnosis code (ICD-10 code)
- `charlson_score`: Comorbidity index (int, 0-10+)
- `num_medications`: Number of medications prescribed (int, 0-30+)
- `length_of_stay`: Days in hospital (int, 1-30+)

## Laboratory Values
- `glucose_level`: Blood glucose (float, mg/dL)
- `creatinine`: Serum creatinine (float, mg/dL)
- `hemoglobin`: Hemoglobin level (float, g/dL)

## Historical Features
- `previous_admissions`: Count of admissions in past year (int, 0-20)
- `days_since_last_admission`: Days since previous admission (int, 0-365+)

## Target Variable
- `readmitted_30days`: Readmitted within 30 days (binary: 0=No, 1=Yes)