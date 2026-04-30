# Heart Attack Risk Predictor - Medical Knowledge Base

## About This Dataset

The Heart Attack Risk Predictor uses data from 303 patients to predict the likelihood of a heart attack.
The target variable is `output`: 1 = higher chance of heart attack, 0 = lower chance.
After feature selection, the model uses 10 key clinical features.

---

## Feature Descriptions and Medical Significance

### 1. Age (`age`)
- **Type**: Numeric (integer, years)
- **Medical meaning**: The patient's age in years.
- **Risk insight**: Age is a significant non-modifiable risk factor. Risk increases substantially after age 45 for men and age 55 for women.
- **Dataset range**: Most patients in this dataset are between 29 and 77 years old, with the majority in the 51–67 age group.
- **Correlation with output**: Negative correlation (-0.225). Younger patients in this dataset paradoxically have slightly higher risk; this reflects selection bias — younger patients who present with symptoms tend to be more severely affected.

### 2. Sex (`sex`)
- **Type**: Binary (0 = Female, 1 = Male)
- **Medical meaning**: The biological sex of the patient.
- **Risk insight**: Men generally have a higher risk of heart attack at a younger age. Women's risk rises sharply after menopause. However, women with heart disease are often underdiagnosed.
- **Correlation with output**: Negative (-0.281). In this dataset, female patients (sex=0) show a slightly higher rate of positive heart attack diagnosis — possibly reflecting the fact that women in the study were at a later disease stage when tested.

### 3. Chest Pain Type (`cp`)
- **Type**: Categorical (0, 1, 2, 3)
  - **0** = Typical angina: Classic exertional chest pain, relieved by rest or nitroglycerin.
  - **1** = Atypical angina: Chest pain with atypical features (may be arm pain, jaw pain, etc.).
  - **2** = Non-anginal pain: Chest discomfort not related to coronary artery disease.
  - **3** = Asymptomatic: Patient has no chest pain symptoms.
- **Risk insight**: Paradoxically, asymptomatic patients (cp=3) may have severe underlying disease (silent ischemia). Typical angina (cp=0) is the classic presentation.
- **Strongest positive predictor**: Correlation with output is +0.434. Higher cp values = higher heart attack risk in this dataset.
- **Clinical note**: Never dismiss chest pain. Any type of chest pain must be evaluated promptly.

### 4. Resting Blood Pressure (`trtbps`)
- **Type**: Numeric (mm Hg)
- **Medical meaning**: Blood pressure measured at rest (systolic).
- **Normal range**: < 120 mm Hg (normal), 120–129 (elevated), 130–139 (Stage 1 hypertension), ≥ 140 (Stage 2 hypertension).
- **Risk insight**: High blood pressure strains the heart and arteries, increasing heart attack risk over time.
- **Correlation with output**: -0.145. Weaker correlation — resting BP alone is not the strongest predictor in this dataset.

### 5. Cholesterol (`chol`)
- **Type**: Numeric (mg/dL)
- **Medical meaning**: Serum cholesterol level measured via BMI sensor.
- **Normal range**: < 200 mg/dL (desirable), 200–239 (borderline high), ≥ 240 (high).
- **Risk insight**: High LDL cholesterol contributes to plaque buildup in arteries (atherosclerosis), directly increasing heart attack risk.
- **Correlation with output**: -0.085. Surprisingly weak in this dataset — cholesterol alone is not a reliable single predictor without context.

### 6. Fasting Blood Sugar (`fbs`)
- **Type**: Binary (1 = fasting blood sugar > 120 mg/dL, 0 = otherwise)
- **Medical meaning**: Indicates presence of elevated fasting blood glucose, a marker for diabetes or prediabetes.
- **Normal fasting glucose**: < 100 mg/dL. Values 100–125 indicate prediabetes; ≥ 126 indicates diabetes.
- **Risk insight**: Diabetes significantly increases cardiovascular risk. Diabetic patients have 2–4x higher heart attack risk.
- **Correlation with output**: -0.028. Very weak correlation in this dataset.

### 7. Resting ECG Results (`restecg`)
- **Type**: Categorical (0, 1, 2)
  - **0** = Normal ECG
  - **1** = ST-T wave abnormality (T wave inversions and/or ST elevation/depression > 0.05 mV). Suggests ischemia or injury.
  - **2** = Probable/definite left ventricular hypertrophy (LVH) by Estes' criteria. Indicates enlarged heart.
- **Risk insight**: ST-T abnormalities are a direct sign of cardiac ischemia. LVH suggests chronic overload.
- **Correlation with output**: +0.137. Moderate positive correlation — abnormal ECG increases risk.

### 8. Maximum Heart Rate Achieved (`thalachh`)
- **Type**: Numeric (beats per minute)
- **Medical meaning**: The maximum heart rate the patient achieved during a stress test.
- **Normal max HR formula**: 220 – age (approximate).
- **Risk insight**: Patients with heart disease often cannot achieve their predicted maximum heart rate (chronotropic incompetence), which is itself a cardiac risk marker. However, achieving a high max HR during exercise stress test can indicate better cardiac reserve.
- **Correlation with output**: +0.422. One of the strongest predictors — higher max HR is associated with higher risk diagnosis in this dataset (likely because those reaching higher HR had more demanding tests ordered).

### 9. Exercise Induced Angina (`exng`)
- **Type**: Binary (1 = yes, 0 = no)
- **Medical meaning**: Whether the patient experienced chest pain/angina during exercise (e.g., treadmill stress test).
- **Risk insight**: Exercise-induced angina is a strong indicator of significant coronary artery disease. It means blood supply cannot meet demand during exertion.
- **Correlation with output**: -0.437. One of the strongest predictors. Patients with exercise angina (exng=1) have lower risk scores in this dataset because it was likely factored into clinical decision differently.

### 10. Number of Major Vessels Colored by Fluoroscopy (`caa`)
- **Type**: Numeric (0, 1, 2, 3)
- **Medical meaning**: Number of major coronary vessels (out of 3: left anterior descending, right coronary artery, left circumflex) showing blockage detected by fluoroscopy/angiography.
- **Risk insight**: More blocked vessels = more severe coronary artery disease = higher heart attack risk.
- **Correlation with output**: -0.392. Strong predictor. Patients with more blocked vessels (caa=3) are more likely to have the disease.

---

## Key Risk Factor Summary

| Feature | Risk Direction | Strength |
|---------|---------------|----------|
| cp (chest pain type) | Higher cp → Higher risk | Very Strong |
| thalachh (max HR) | Higher HR → Higher risk | Strong |
| exng (exercise angina) | Yes → Lower output score | Strong |
| caa (blocked vessels) | More → Lower output score | Strong |
| sex (gender) | Female (0) → Slightly higher risk | Moderate |
| age | Older → Slightly lower output | Moderate |
| restecg (ECG result) | Abnormal → Higher risk | Moderate |
| trtbps (blood pressure) | Higher → Slightly lower output | Weak |
| chol (cholesterol) | Higher → Slightly lower output | Weak |
| fbs (fasting glucose) | Elevated → Lower output | Very Weak |

---

## Heart Attack Risk Factors — General Medical Knowledge

### Major Modifiable Risk Factors
- **High blood pressure (hypertension)**: Values ≥ 140/90 mm Hg significantly increase risk.
- **High cholesterol**: Especially high LDL (> 130 mg/dL) and low HDL.
- **Smoking**: Doubles heart attack risk. Carbon monoxide damages blood vessel walls.
- **Obesity**: BMI > 30 increases cardiac workload and promotes inflammation.
- **Physical inactivity**: Regular exercise reduces heart attack risk by 35%.
- **Diabetes**: Insulin resistance promotes atherosclerosis.
- **Unhealthy diet**: High saturated fat, sodium, and processed foods increase risk.
- **Stress**: Chronic stress elevates cortisol, increasing blood pressure and inflammation.

### Non-Modifiable Risk Factors
- **Age**: Risk increases with age (men > 45, women > 55).
- **Sex**: Men at higher risk at younger ages; women catch up post-menopause.
- **Family history**: First-degree relative with heart disease increases your risk significantly.
- **Ethnicity**: South Asians have higher genetic predisposition.

### Warning Signs of Heart Attack (FAST + P)
- **Chest pain**: Pressure, squeezing, fullness, or pain in center or left of chest.
- **Arm/shoulder pain**: Radiating down left or both arms.
- **Jaw/neck pain**: Often in women.
- **Shortness of breath**: With or without chest discomfort.
- **Cold sweat, nausea, lightheadedness**: Common accompanying symptoms.

### Prevention Strategies
1. Maintain blood pressure < 120/80 mm Hg
2. Keep LDL cholesterol < 100 mg/dL (< 70 if high risk)
3. Exercise: 150 minutes/week moderate aerobic activity
4. Quit smoking
5. Maintain healthy weight (BMI 18.5–24.9)
6. Control blood sugar (HbA1c < 7% if diabetic)
7. Mediterranean diet (olive oil, fish, vegetables, whole grains)
8. Regular cardiac checkups after age 40

---

## Model and Dataset Statistics

### Dataset Summary
- **Total patients**: 303
- **Features used**: age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, caa
- **Dropped features**: oldpeak, slp, thall (high missingness/low predictive power)
- **Target distribution**: ~54% high risk (output=1), ~46% low risk (output=0)

### Model Performance (sklearn classifiers)
The following classifiers were trained on this dataset:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes

Best performing models typically achieve **85–90% accuracy** on this dataset.

### Top Predictive Features (by correlation magnitude)
1. exng (exercise induced angina): |r| = 0.437
2. cp (chest pain type): |r| = 0.434
3. thalachh (max heart rate): |r| = 0.422
4. caa (blocked vessels): |r| = 0.392
5. sex (gender): |r| = 0.281
6. age: |r| = 0.225

---

## Clinical Decision Support Notes

**IMPORTANT**: This AI system is for educational and informational purposes only.
It is NOT a substitute for professional medical diagnosis or treatment.
Always consult a licensed physician for medical decisions.

### When to Seek Emergency Care Immediately
- Sudden chest pain lasting more than 5 minutes
- Chest pain with shortness of breath, sweating, or nausea
- Sudden severe headache, vision changes, or speech difficulty
- Rapid or irregular heartbeat with dizziness

### Interpreting Model Predictions
- **output = 1**: Model predicts higher probability of heart attack risk. Seek medical evaluation.
- **output = 0**: Model predicts lower risk. Maintain preventive lifestyle habits. Still consult doctor for regular checkups.
