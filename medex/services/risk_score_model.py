import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib

from medex.services.old_risk_score_model import to_features, scaler, model, X_scaled, y


def train_diabetes_prediction_model(train_data):
    # Load data from CSV file
    data = pd.read_csv(f'../../examples/{train_data}')

    # Determine categories of categorical columns
    gender_categories = data['gender'].unique()
    smoking_history_categories = data['smoking_history'].unique()

    # One-hot encode categorical columns
    encoder = OneHotEncoder(categories=[gender_categories, smoking_history_categories])
    encoder.fit(data[['gender', 'smoking_history']])

    # Define a function to preprocess the data
    def to_features(df):
        onehot = encoder.transform(df[['gender', 'smoking_history']])
        df_onehot = pd.DataFrame(onehot.toarray(), columns=encoder.get_feature_names_out(['gender', 'smoking_history']))

        # Concatenate numerical columns with one-hot encoded columns
        columns = list(df.columns)
        columns.remove('gender')
        columns.remove('smoking_history')
        df = pd.concat([df_onehot, df[columns]], axis=1)

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

    # Print the model coefficients - I don't think we need that anymore TODO
    # coef_dict = {}
    # for coef, feat in zip(model.coef_[0], X.columns):
    #   coef_dict[feat] = coef
    # print(coef_dict)

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

    # Save the model and preprocessing components - same issue like in the loading part TODO
    joblib.dump(model, 'diabetes_prediction_model.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Define a function to calculate risk score using the saved model and preprocessing components
    def calculate_risk_score(new_patient):
        model = joblib.load('diabetes_prediction_model.pkl')
        encoder = joblib.load('encoder.pkl')
        scaler = joblib.load('scaler.pkl')

        new_patient_df = pd.DataFrame(new_patient, index=[0])
        new_patient_X = to_features(new_patient_df)

        # Scale the input features using the same scaler as used in training
        new_patient_x_scaled = scaler.transform(new_patient_X)

        risk_score = model.predict_proba(new_patient_x_scaled)
        return risk_score[0, 1]


new_patient = {'gender': 'Male', 'age': 44.0, 'hypertension': 1, 'heart_disease': 1,
               'smoking_history': 'current', 'bmi': 30, 'HbA1c_level': 6.5, 'blood_glucose_level': 200}

new_patient_df = pd.DataFrame(new_patient, index=[0])
new_patient_X = to_features(new_patient_df)

# Scale the input features using the same scaler as used in training
new_patient_x_scaled = scaler.transform(new_patient_X)

risk_score = model.predict_proba(new_patient_x_scaled)
print('Risk score:', risk_score[0, 1])
y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print('Accuracy:', accuracy)

## This part I added to check if the preprocesses data is actually saved- How to handle the names?  TODO
loaded_model = joblib.load('diabetes_prediction_model.pkl')
loaded_encoder = joblib.load('encoder.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Check if the loaded model and preprocessing components are working
new_patient = {'gender': 'Male', 'age': 44.0, 'hypertension': 1, 'heart_disease': 1,
               'smoking_history': 'current', 'bmi': 30, 'HbA1c_level': 6.5, 'blood_glucose_level': 200}
new_patient_df = pd.DataFrame(new_patient, index=[0])
new_patient_X = to_features(new_patient_df)
new_patient_x_scaled = loaded_scaler.transform(new_patient_X)
risk_score = loaded_model.predict_proba(new_patient_x_scaled)
print('Risk score:', risk_score[0, 1])
