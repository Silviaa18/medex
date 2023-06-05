class DiabetesPlugin(PluginInterface):
    @classmethod
    def get_name(cls) -> str:
        return "calculate"

    @classmethod
    def required_parameters(cls) -> List[str]:
        pass

    @classmethod
    def get_entity_type(cls) -> EntityType:
        return EntityType.CATEGORICAL

    def convert_to_features(df, enc, categorical_columns: list = ['Sex', 'Tobacco smoking']):
        assert isinstance(categorical_columns, list)

        onehot = enc.transform(df[categorical_columns])
        df_onehot = pd.DataFrame(onehot.toarray(), columns=enc.get_feature_names_out(categorical_columns))

        # Concatenate numerical columns with one-hot encoded columns
        columns = list(df.columns)
        for col in categorical_columns:
            columns.remove(col)
        df = pd.concat([df_onehot, df[columns]], axis=1)

        return df

    def load_model(self):
        model = joblib.load('diabetes_prediction_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        encoder = joblib.load('diabetes_encoder.pkl')
        return model, encoder, scaler

    def get_entities(self) -> tuple[list, list]:
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Delta0", "Delta2"]
        # actual database entries
        # cat_entities = ["Diagnoses - ICD10", "Sex", "Tobacco smoking"]
        # num_entities = ["year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]

        return cat_entities, num_entities

    def convert_to_features(df, enc, categorical_columns: list = ['Sex', 'Tobacco smoking']):
        assert isinstance(categorical_columns, list)

        onehot = enc.transform(df[categorical_columns])
        df_onehot = pd.DataFrame(onehot.toarray(), columns=enc.get_feature_names_out(categorical_columns))

        # Concatenate numerical columns with one-hot encoded columns
        columns = list(df.columns)
        for col in categorical_columns:
            columns.remove(col)
        df = pd.concat([df_onehot, df[columns]], axis=1)

        return df

    def calculate(self, params: Dict[str, Any]) -> Any:
        df = params['df']
        model, encoder, scaler = self.load_model()
        if isinstance(params, dict):
            df = pd.DataFrame(df, index=[0])
            has_disease, risk_score = self.calculate(df)
            return has_disease[0], risk_score[0]

        df = self.convert_to_features(df, encoder)
        # Scale the input features using the same scaler as used in training
        new_patient_x_scaled = scaler.transform(df)

        risk_score = model.predict_proba(new_patient_x_scaled)

        print('Risk score:', risk_score[:, 1])

        has_disease = risk_score[:, 1] >= 0.5

        return has_disease, risk_score[:, 1]
