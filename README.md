# Predict Credit Card Approval
In today's digital age, access to credit is crucial, but manually analyzing credit card applications can be time-consuming and error-prone. In this project, I used machine learning to build a model that predicts credit card approval with high accuracy, potentially helping lenders make faster and more informed decisions."

## Goals and Approach:
My goal was to build a model that could predict whether an applicant would be approved for a credit card based on their data. I started by exploring the data, and identifying patterns and trends. Then, I preprocessed the data by handling missing values, and outliers, and performing necessary transformations. Next, I trained various machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting. Finally, I evaluated the models based on their performance metrics and selected the best one."

## Achievements and Challenges:
One key finding from my data analysis was that annual income, family members, and employment duration were the most important features for predicting approval. I used techniques like SMOTE to address imbalanced data, where applicants with high risk were much fewer. Interestingly, during the economic recession, prioritizing recall (catching good applicants) was more important than precision (avoiding bad applicants), so I chose Gradient Boosting as the best model based on its recall score."
## Impact and Skills:
My model achieved an accuracy of 88.48%, potentially improving lenders' ability to assess creditworthiness efficiently. This project allowed me to showcase my skills in data analysis, machine learning, and feature engineering. I also learned valuable lessons about ethical considerations in AI models and the importance of tailoring them to specific business contexts."



## Here We follow the Following Steps:
+ We will start with exploratory data analysis(EDA)
+ Feature engineering
+ Feature selection
+ Data preprocessing
+ Model training
+ Model selection
+ Build a web app interface for the model using Streamlit.
+ Deploy the model

### Summary of EDA
#### Gender Distribution:
+ Two unique classes, Female (F) and Male (M), with 778 and 454 applicants, respectively.
+ 63.14% are female, and 36.85% are male.

#### Marital Status:
+ Five unique classes, with most applicants (68.50%) being married.
+ Widows have a higher risk compared to separated individuals.

#### Family Members:
+ Numerical feature with a median of 2 family members.
+ Most applicants have 2 family members (51.85%).

#### Children:
+ Most applicants don't have children.
+ Three outliers with 5, 6, and 15 children.

#### Housing Type:
+ 89.42% of applicants live in a house/apartment.

#### Annual Income:
+ The average income is 194,770.84, with outliers.
+ Most people make 172,125, ignoring outliers.
+ Positively skewed distribution.

#### Type Occupation:
+ Laborers (25.06%) and Core staff (16.83%) are the most common occupations.
+ 32.31% missing data.

#### Type Income:
+ Most applicants are working (51.21%), followed by commercial associates (23.59%).

#### Education:
+ Most applicants completed Secondary/Secondary Special (66.24%).

#### Employed Days:
+ Most applicants have been working between 5 to 7 years on average.
+ Outliers with employment durations over 20 years.

#### Car Ownership and Property Ownership:
+ Most applicants don't have a car (70.36%).
+ Most applicants own property (65.02%).

#### Phone and Work Phone:
+ 78.35% of applicants don't have a work phone.
+ More than 90% don't have an email ID.
+ All applicants have a mobile phone.
+ 69.39% don't have a phone, and 30.61% have a phone.

#### Is High Risk (Target Variable):
+ 88.29% have no risk, and 11.71% have risk, indicating imbalanced data.
+ Imbalance needs to be addressed using SMOTE before model training.

#### Age and Is High Risk:
+ There is no significant difference in age between high-risk and low-risk applicants.
+ The mean age for both groups is around 43 years, and there is no correlation between age and risk factors.

#### Correlations:{HeatMap]
+ Positive linear correlation between Family Members and Children, introducing multicollinearity.
+ Similar pattern observed between Employed Days and Age, indicating longer employment correlates with older age.
+ No features are strongly correlated with the target variable.
+ Positive correlation between having a Phone and a Work_phone.
+ Negative correlation between Employed_days and Age.
+ Positive correlation between Age and Work_phone.

#### Demographic Observations:
+ Female applicants are older than male applicants.
+ Non-car owners tend to be older.
+ Property owners are generally older.
+ Pensioners are older than working individuals, with some outliers.
+ Widows are generally older.
+ Individuals living with parents are younger, with some outliers.
+ Security staff tends to be older, while IT staff tends to be younger


### Deployed Application link
+ https://preditct-credit-card-appoval.onrender.com/
+ https://deploypredictcreditcardapproval.streamlit.app/

