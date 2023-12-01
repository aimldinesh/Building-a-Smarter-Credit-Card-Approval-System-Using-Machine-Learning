import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import tempfile
import json
import requests
from streamlit_lottie import st_lottie_spinner


train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

cc_full_data = pd.concat([train_data, test_data], axis=0)
cc_full_data = cc_full_data.sample(frac=1).reset_index(drop=True)
def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

cc_train_data, cc_test_data = data_split(cc_full_data, 0.2)


# Data Cleaning
# 1..Impute Missing Value
class Handle_Missing_Values(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols=['GENDER', 'Type_Occupation'], numerical_cols=['Annual_income', 'Age']):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
    
    def fit(self, df):
        # Store the mode for categorical columns and the mean for numerical columns
        self.imputation_values = {}
        for col in self.categorical_cols:
            self.imputation_values[col] = df[col].mode()[0]
        for col in self.numerical_cols:
            self.imputation_values[col] = df[col].mean()
        return self
    
    def transform(self, df):
        # Replace missing values with the stored imputation values
        for col in self.categorical_cols:
            df[col].fillna(self.imputation_values[col], inplace=True)
        for col in self.numerical_cols:
            df[col].fillna(self.imputation_values[col], inplace=True)
        return df 

#2.calculate the count of each class in a feature with its frequency (normalized on a scale of 100)  
def value_cnt_freq(df, feature):
    # we get the value counts of each feature
    ftr_value_cnt = df[feature].value_counts()
    # we normalize the value counts on a scale of 100
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    # we concatenate the value counts with normalized value count column wise
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    # give it a column name
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    # return the dataframe
    return ftr_value_cnt_concat

# 3.Outlier Remover Function  
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_outliers = ['Family_Members','Annual_income', 'Employed_days']):
        self.feat_with_outliers = feat_with_outliers
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_outliers).issubset(df.columns)):
            # 25% quantile
            Q1 = df[self.feat_with_outliers].quantile(.25)
            # 75% quantile
            Q3 = df[self.feat_with_outliers].quantile(.75)
            IQR = Q3 - Q1
            # keep the data within 3 IQR
            df = df[~((df[self.feat_with_outliers] < (Q1 - 3 * IQR)) |(df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)]
            return df
        else:
            print("Warning: One or more specified features for outlier removal are not present in the dataframe. No outliers removed.")
            return df           

# 4.Drop Features
class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['Ind_ID','Mobile_phone','CHILDREN','Type_Occupation']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print("Warning: One or more specified features for dropping are not present in the dataframe. No features dropped.")
            return df           

# 5.Convert Time 
class Handle_time_conversion(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days=['Employed_days', 'Age']):
        self.feat_with_days = feat_with_days

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if set(self.feat_with_days).issubset(X.columns):
            # We convert days to absolute value
            X[['Employed_days', 'Age']] = np.abs(X[['Employed_days', 'Age']])
            return X
        else:
            print("Error: The specified features for time conversion are not present in the dataframe.")
            return X 

# 6.Handle Retiree
class Handle_Retiree(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        # Check if 'Employed_days' is in the dataframe columns
        if 'Employed_days' in df.columns:
            # Select rows with 'Employed_days' equal to 365243, corresponding to retirees
            df_ret_idx = df['Employed_days'][df['Employed_days'] == 365243].index
            # Change 'Employed_days' value from 365243 to 0 for retirees
            df.loc[df_ret_idx, 'Employed_days'] = 0
            return df
        else:
            # Print an error message if 'Employed_days' is not in the dataframe
            print("Error: 'Employed_days' is not in the dataframe.")
            return df

# 7.Handle Skewness
class Handle_Skewness(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=['Annual_income', 'Age']):
        self.feat_with_skewness = feat_with_skewness
    
    def fit(self, df):
        # This transformer doesn't require fitting, so it returns self.
        return self
    
    def transform(self, df):
        # Check if features with skewness are present in the dataframe columns
        if set(self.feat_with_skewness).issubset(df.columns):
            # Apply cubic root transformation to handle skewness
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            # Print an error message if features are not in the dataframe
            print("Error: One or more features are not in the dataframe.")
            return df

# 8.Binary Number to Y and N
class BinningNum_To_YN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=['Work_Phone', 'Phone', 'EMAIL_ID']):
        self.feat_with_num_enc = feat_with_num_enc

    def fit(self, df):
        # No fitting is needed, return self
        return self

    def transform(self, df):
        # Check if all features in feat_with_num_enc are present in the dataframe
        if set(self.feat_with_num_enc).issubset(df.columns):
            # Change 0 to 'N' and 1 to 'Y' for all features in feat_with_num_enc
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1: 'Y', 0: 'N'})
            return df
        else:
            # Print a message if one or more features are not in the dataframe
            print("One or more features are not in the dataframe")
            return df

# 9.One Hot Encoding
class OneHot_Encoding_with_Features(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=['GENDER', 'Car_Owner', 'Propert_Owner', 'Marital_status', 'Housing_type', 'Type_Income', 'Work_Phone', 'Phone', 'EMAIL_ID']):
        self.one_hot_enc_ft = one_hot_enc_ft
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        if set(self.one_hot_enc_ft).issubset(df.columns):
            # Function to one-hot encode the features in one_hot_enc_ft
            def one_hot_enc(df, one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])
                # Get the result of the one-hot encoding column names
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
                # Change the array of the one-hot encoding to a dataframe with the column names
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(), columns=feat_names_one_hot_enc, index=df.index)
                return df
            
            # Function to concatenate the one-hot encoded features with the rest of features that were not encoded
            def concat_with_rest(df, one_hot_enc_df, one_hot_enc_ft):
                # Get the rest of the features
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                # Concatenate the rest of the features with the one-hot encoded features
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]], axis=1)
                return df_concat
            
            # One-hot encoded dataframe
            one_hot_enc_df = one_hot_enc(df, self.one_hot_enc_ft)
            # Returns the concatenated dataframe
            full_df_one_hot_enc = concat_with_rest(df, one_hot_enc_df, self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("Warning: One or more specified features are not present in the dataframe. Returning the original dataframe.")
            return df  

# 10.Ordinal Encoding
class Ordinal_Encoding_with_Feature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=['EDUCATION']):
        self.ordinal_enc_ft = ordinal_enc_ft
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        if 'EDUCATION' in df.columns:
            # Instantiate the OrdinalEncoder object
            ordinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print("Warning: 'EDUCATION' feature is not present in the dataframe. Returning the original dataframe.")
            return df                                  

# 11.Min-Max Scaling
class MinMax_Scaling_With_Feature(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_scale=['Age', 'Annual_income', 'Employed_days']):
        self.features_to_scale = features_to_scale
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        if set(self.features_to_scale).issubset(df.columns):
            min_max_scaler = MinMaxScaler()
            df[self.features_to_scale] = min_max_scaler.fit_transform(df[self.features_to_scale])
            return df
        else:
            print("Warning: One or more features not found in the dataframe. Returning the original dataframe.")
            return df

# 12.Balanced the Imbalanced[Oversampling]
class Target_Oversampling(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Is_high_risk' in df.columns:
            # SMOTE function to oversample the minority class and balance the dataset
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'Is_high_risk'], df['Is_high_risk'])
            df_bal = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
            return df_bal
        else:
            print("Warning: 'Is_high_risk' not found in the dataframe. Returning the original dataframe.")
            return df

# Data Preprocessing Pipeline
def full_pipeline(df):
    pipeline = Pipeline([
        ('handle_missing_values',Handle_Missing_Values()),
        ('outlier_remover',OutlierRemover()),
        ('drop_feature', DropFeatures()),
        ('handle_time_conversion',Handle_time_conversion()),
        ('handle_retiree', Handle_Retiree()),
        ('handle_skewness', Handle_Skewness()),
        ('binnin_gnum_to_yn', BinningNum_To_YN()),
        ('one_hot_encoding_with_features', OneHot_Encoding_with_Features()),
        ('ordinal_encoding_with_feature', Ordinal_Encoding_with_Feature()),
        ('minmax_scaling_with_feature', MinMax_Scaling_With_Feature()),
        ('target_oversampling', Target_Oversampling())         
    ])
    df_pipeline_pre = pipeline.fit_transform(df)
    return df_pipeline_pre        

# Build Streamlit App
import streamlit as st
import numpy as np

# Assume cc_full_data is defined and value_cnt_freq function is available

# Build Streamlit App
st.set_page_config(page_title="Credit Card Approval Predictor", page_icon="🔮", layout="wide")

# Header
st.markdown(
    """
    # 🌟 PREDICT CREDIT CARD APPROVAL 🌟
    *This app predicts whether an application for a credit card will be approved or not by using the applicant's data.*
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Applicant Information")

# Sidebar Layout
with st.sidebar.expander("Personal Information"):
    # Gender
    input_gender = st.radio('Choose your gender', ['Male', 'Female'], index=0)

    # Age
    input_age = np.negative(st.slider('Choose your Age', value=42, min_value=18, max_value=70, step=1) * 365.25)

    # Marital Status
    marital_status_values = list(value_cnt_freq(cc_full_data, 'Marital_status').index)
    marital_status_key = ['Married', 'Single/not married', 'Civil marriage', 'Separated', 'Widow']
    marital_status_dict = dict(zip(marital_status_key, marital_status_values))
    input_marital_status_key = st.selectbox('Select your marital status', marital_status_key)
    input_marital_status_val = marital_status_dict.get(input_marital_status_key)

    # Family Members
    family_member_count = float(st.selectbox('Choose your family member length', [1, 2, 3, 4, 5, 6]))

    # Housing Type
    housing_type_values = list(value_cnt_freq(cc_full_data, 'Housing_type').index)
    housing_type_key = ['House / apartment', 'With parents', 'Municipal apartment ', 'Rented apartment', 'Office apartment', 'Co-op apartment']
    housing_type_dict = dict(zip(housing_type_key, housing_type_values))
    input_housing_type_key = st.selectbox('Select the type of housing type you live in', housing_type_key)
    input_housing_type_val = housing_type_dict.get(input_housing_type_key)

    # Car Owner
    input_car_owner = st.radio('Are you a car owner?', ['Yes', 'No'], index=0)

    # Property Owner
    input_property_owner = st.radio('Are you a property owner?', ['Yes', 'No'])

    # Annual Income
    input_Annual_income = int(st.text_input('Enter your income (in USD)', 0))

    # Type Income
    type_income_values = list(value_cnt_freq(cc_full_data, 'Type_Income').index)
    type_income_key = ['Working', 'Commercial associate', 'Pensioner', 'State servant']
    type_income_dict = dict(zip(type_income_key, type_income_values))
    input_type_income_key = st.selectbox('Select your type income', type_income_key)
    input_type_income_val = type_income_dict.get(input_type_income_key)

    # Education Level
    edu_level_values = list(value_cnt_freq(cc_full_data, 'EDUCATION').index)
    edu_level_key = ['Secondary / secondary special', 'Higher education', 'Incomplete higher ', 'Lower secondary', 'Academic degree']
    edu_level_dict = dict(zip(edu_level_key, edu_level_values))
    input_edu_level_key = st.selectbox('Select your education level', edu_level_key)
    input_edu_level_val = edu_level_dict.get(input_edu_level_key)

    # Employed Days
    input_employed_length = np.negative(st.slider('Choose your employment length in year', value=6, min_value=0, max_value=30, step=1) * 365.25)
    
    # Work Phone
    input_work_phone = st.radio('You have a work phone?', ['Yes', 'No'], index=0)
    work_phone_dict = {'Yes': 1, 'No': 0}
    work_phone_val = work_phone_dict.get(input_work_phone)

    # Phone
    input_phone = st.radio('You have a phone?', ['Yes', 'No'], index=0)
    phone_dict = {'Yes': 1, 'No': 0}
    phone_val = phone_dict.get(input_phone)
    
    # Email ID
    input_email_id = st.radio('You have an email id?', ['Yes', 'No'], index=0)
    email_id_dict = {'Yes': 1, 'No': 0}
    email_id_val = email_id_dict.get(input_email_id)

# Display selected inputs
st.write("## Applicant's Information")
st.write(f"**Gender:** {input_gender}")
st.write(f"**Age:** {input_age / 365.25:.2f} years")
st.write(f"**Marital Status:** {input_marital_status_key}")
st.write(f"**Family Members:** {family_member_count}")
st.write(f"**Housing Type:** {input_housing_type_key}")
st.write(f"**Car Owner:** {input_car_owner}")
st.write(f"**Property Owner:** {input_property_owner}")
st.write(f"**Annual Income:** ${input_Annual_income:,}")
st.write(f"**Type Income:** {input_type_income_key}")
st.write(f"**Education Level:** {input_edu_level_key}")
st.write(f"**Employed Days:** {input_employed_length / 365.25:.2f} years")
st.write(f"**Work Phone:** {'Yes' if input_work_phone == 'Yes' else 'No'}")
st.write(f"**Phone:** {'Yes' if input_phone == 'Yes' else 'No'}")
st.write(f"**Email ID:** {'Yes' if input_email_id == 'Yes' else 'No'}")

# Predict Button
predict_button = st.button('PREDICT')

# Add some styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stSidebar {
            background-color: #e6e6e6;
            padding: 20px;
            border-radius: 10px;
        }
        .stContent {
            padding: 20px;
        }
        .stButton {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton:hover {
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

predict_profile = [
    0, # Ind_ID
    input_gender[:1], 
    input_car_owner[:1], 
    input_property_owner[:1], 
    0, # CHILDREN
    input_Annual_income, 
    input_type_income_val, 
    input_edu_level_val, 
    input_marital_status_val, 
    input_housing_type_val, 
    input_age, 
    input_employed_length,
    1,# Mobile_phone
    work_phone_val, 
    phone_val, 
    email_id_val, 
    'to be droped', # Type_Occupation
    family_member_count, 
    0,
    
]


predict_profile_df = pd.DataFrame([predict_profile], columns = cc_train_data.columns)

# add the profile to predict as a last row in the train data
cc_train_data_with_predict_profile = pd.concat([cc_train_data,predict_profile_df],ignore_index = True)

# preprocessing data
cc_train_data_with_predict_profile_prep = full_pipeline(cc_train_data_with_predict_profile)
#print(cc_train_data_with_predict_profile_prep.columns)


# Check if 'Ind_ID' is present in the dataframe before dropping columns
if 'Ind_ID' in cc_train_data_with_predict_profile_prep.columns:
    # Drop 'Ind_ID' and 'Is_high_risk' only if 'Ind_ID' is present
    predict_profile_prep = cc_train_data_with_predict_profile_prep[cc_train_data_with_predict_profile_prep['Ind_ID'] == 0].drop(columns=['Ind_ID', 'Is_high_risk'], errors='ignore')
else:
    # If 'Ind_ID' is not present, drop only 'Is_high_risk'
    predict_profile_prep = cc_train_data_with_predict_profile_prep.iloc[[-1]].drop(columns=['Is_high_risk'], errors='ignore')



# Load the model
model = joblib.load('saved_models/gradient_boosting/gradient_boosting_model.sav')

#Function to make predictions
def make_prediction():
    return model.predict(predict_profile_prep)

#Animation function
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_loading_an = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json')

# Display loading animation while making prediction
if predict_button:
    with st_lottie_spinner(lottie_loading_an, quality='high', height='200px', width='200px'):
        final_pred = make_prediction()

    # Stop displaying the loading animation and show results based on the prediction
    if final_pred[0] == 0:
        st.success('### Congratulations! Your credit card application has been approved.')
        st.balloons()
    elif final_pred[0] == 1:
        st.error('### Your credit card application has not been approved.')

