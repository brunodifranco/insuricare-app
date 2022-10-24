import pickle
import pandas as pd

class Insurance:
    def __init__(self):
        self.path = 'C:/Users/bruno/OneDrive/Documentos/repos/pa-health-insurance-cross-sell/'
        self.age_scaler   = pickle.load(open(self.home_path + 'scalers/age_scaler.pkl', 'rb'))
        self.annual_premium_scaler   = pickle.load(open(self.home_path + 'scalers/annual_premium_scaler.pkl', 'rb'))
        self.policy_sales_channel_scaler   = pickle.load(open(self.home_path + 'scalers/policy_sales_channel_scaler.pkl', 'rb'))
        self.region_code_scaler   = pickle.load(open(self.home_path + 'scalers/region_code_scaler.pkl', 'rb'))
        self.vehicle_age_mms_scaler   = pickle.load(open(self.home_path + 'scalers/vehicle_age_mms_scaler.pkl', 'rb'))
        self.vehicle_age_oe_scaler   = pickle.load(open(self.home_path + 'scalers/vehicle_age_oe_scaler.pkl', 'rb'))
        self.vintage_scaler   = pickle.load(open(self.home_path + 'scalers/vintage_scaler.pkl', 'rb'))

    def data_cleaning(self, data):
        return data # No data cleaning transformations were applied

    def feature_engineering(self, data):
        data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 'below_1_year' if x=='< 1 Year'
                                              else 'between_1_2_years' if x=='1-2 Year'
                                              else 'above_2_years')

        data['vehicle_damage'] = data['vehicle_damage'].apply(lambda x: 0 if x=='No' 
                                                                else 1)
        return data

    def data_preparation(self, data):
        # age - MinMaxScaler
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # vintage - MinMaxScaler
        data['vintage'] = self.vintage_scaler.transform(data[['vintage']].values)

        # RobustScaler - annual_premium
        data['annual_premium'] = self.annual_premium_scaler.transform(data[['annual_premium']].values)

        # gender
        data = pd.get_dummies(data, prefix=['gender'], columns=['gender']).rename(columns={'gender_Female': 'female', 'gender_Male': 'male'})    

        # region_code
        data['region_code'] = data['region_code'].astype(str)
        data['region_code'] = self.region_code_scaler.transform(X=data[['region_code']].values, y=data[['response']].values)

        # policy_sales_channel 
        data['policy_sales_channel'] = data['policy_sales_channel'].astype(str)
        data['policy_sales_channel'] = self.policy_sales_channel_scaler.transform(X=data[['policy_sales_channel']].values, y=data[['response']].values)

        # vehicle_age
        mapping = {'below_1_year': 1, 'between_1_2_years': 2, 'above_2_years': 3} # mapping ordered values
        data['vehicle_age'] = self.vehicle_age_oe_scaler.transform(X=data[['vehicle_age']].values, y=data[['response']].values, mapping=mapping)

        # Now rescaling new vehicle_age - MinMaxScaler
        data['vehicle_age'] = self.vehicle_age_mms_scaler.transform(data[['vehicle_age']].values)

        # Feature Selection
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'policy_sales_channel', 'vehicle_damage', 'previously_insured']

        return data[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # Prediction
        pred = model.predict_proba(test_data)
        
        # Prediction as a column in the original data
        original_data['score'] = pred
        
        return original_data.to_json(orient='records', date_format='iso') # Converts to json