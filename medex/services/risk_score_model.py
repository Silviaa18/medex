
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Load data from CSV file
data = pd.read_csv('../../examples/diabetes_prediction_dataset.csv')

# Determine categories of categorical columns
gender_categories = data['gender'].unique()
smoking_history_categories = data['smoking_history'].unique()

# One-hot encode categorical columns
encoder = OneHotEncoder(categories=[gender_categories, smoking_history_categories])
encoder.fit(data[['gender', 'smoking_history']])


def to_features(df):
    onehot = encoder.transform(df[['gender', 'smoking_history']])
    df_onehot = pd.DataFrame(onehot.toarray(), columns=encoder.get_feature_names_out(['gender', 'smoking_history']))

    # Concatenate numerical columns with one-hot encoded columns
    columns = list(df.columns)
    columns.remove('gender')
    columns.remove('smoking_history')
    df = pd.concat([df_onehot, df[columns]], axis=1)

    print(df)

    return df


data = to_features(data)
# Split data into features (X) and target (y)
X = data.drop(['diabetes'], axis=1)
y = data['diabetes']

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the logistic regression model using the scaled data
model = LogisticRegression()
model.fit(X_scaled, y)


# Print the model coefficients
coef_dict = {}
for coef, feat in zip(model.coef_[0], X.columns):
    coef_dict[feat] = coef
print(coef_dict)

# Calculate risk score for a new patient
new_patient = {'gender': 'Male', 'age': 44.0, 'hypertension': 1, 'heart_disease': 1,
               'smoking_history': 'current', 'bmi': 30, 'HbA1c_level': 6.5, 'blood_glucose_level': 200}

new_patient_df = pd.DataFrame(new_patient, index=[0])

new_patient_X = to_features(new_patient_df)

# Scale the input features using the same scaler as used in training
new_patient_x_scaled = scaler.transform(new_patient_X)

risk_score = model.predict_proba(new_patient_x_scaled)
print('Risk score:', risk_score[0, 1])

# Evaluate model accuracy
y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print('Accuracy:', accuracy)
