# Building-a-Smarter-Credit-Card-Approval-System-Using-Machine-Learning
In today's digital age, access to credit is crucial, but manually analyzing credit card applications can be time-consuming and error-prone. In this project, I used machine learning to build a model that predicts credit card approval with high accuracy, potentially helping lenders make faster and more informed decisions."

## Goals and Approach:
My goal is to build a model that could predict whether an applicant would be approved for a credit card based on their data. I started by exploring the data, and identifying patterns and trends. Then, I preprocessed the data by handling missing values, and outliers, and performing necessary transformations. Next, I trained various machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting. Finally, I evaluated the models based on their performance metrics and selected the best one."

## Achievements and Challenges:
One key finding from this data analysis was that annual income, employment duration, Age, and family members were the most important features for predicting approval. I used techniques like SMOTE to address imbalanced data, where applicants with high risk were much fewer. Interestingly, during the economic recession, prioritizing recall (catching good applicants) was more important than precision (avoiding bad applicants), so I chose Gradient Boosting as the best model based on its recall score."
## Impact and Skills:
This model achieved an accuracy of 88.48%, potentially improving lenders' ability to assess creditworthiness efficiently. This project allowed me to showcase my skills in data analysis, machine learning, and feature engineering. "

![Alt text](https://github.com/dsml917/Deploy_predict_credit_card_approval_app/blob/main/Images/Capture1.PNG)
![Alt text](https://github.com/dsml917/Deploy_predict_credit_card_approval_app/blob/main/Images/Capture2.PNG)
![Alt text](https://github.com/dsml917/Deploy_predict_credit_card_approval_app/blob/main/Images/gb_roc_curve.PNG)
![ALt text](https://github.com/dsml917/Deploy_predict_credit_card_approval_app/blob/main/Images/gbconfusion_matrix.PNG)
![Alt text](https://github.com/dsml917/Deploy_predict_credit_card_approval_app/blob/main/Images/gbfeatures.PNG)
![Alt text](https://github.com/dsml917/Deploy_predict_credit_card_approval_app/blob/main/Images/heatmap.PNG)

## Here We follow the Following Steps:
+ Exploratory data analysis(EDA)
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

## Preparing the Data:
### Here is a list of all the transformations that need to be applied to each feature
#### Ind_ID:
+ Drop the feature

#### GENDER:
+ One-hot Encoding
+ impute missing value

#### Car_Owner:
+ Change it numerical
+ One-hot encoding

#### Propert_Owner:
+ Change it numerical
+ One-hot encoding

#### CHILDREN:
+ Fix outlier
+ Drop Feature

#### Annual_income:
+ Remove outlier
+ Fix Skewness
+ Min-Max Scaling
+ impute the missing value

#### Type_Income:
+ One-hot encoding

#### EDUCATION:
+ Ordinal encoding

#### Marital_status:
+ One-hot encoding

#### Housing_type:
+ One-hot encoding

#### Age:
+ Min-Max Scaling
+ Fix Skewness
+ Abs value and divide 365.25
+ impute missing value

#### Employed_days:
+ Min-Max Scaling
+ Remove outlier
+ Abs value and divide 365.25

#### Mobile_phone:
+ Drop feature

#### Work_Phone:
+ One-hot encoding

#### Phone:
+ One-hot encoding

#### EMAIL_ID:
+ One-hot encoding

#### Type_Occupation:
+ One-hot encoding
+ impute the missing value

#### Family_Members:
+ Fix outlier

#### Is_high_risk:
+ Balance the Imbalance Data with SMOTE

## Feature Selection:
### Drop Features:
+ We dropped Ind_ID because it is not useful for prediction. It was helpful during the merging of the two datasets, but afterward, it became unnecessary.
+ We dropped Mobile_phone because everyone has a mobile phone, and thus, this feature did not provide any useful information.
+ We dropped CHILDREN because it is highly correlated with Family_Members. To avoid multicollinearity, it was necessary to remove this feature.
+ We dropped Type_Occupation due to the presence of many missing values.
  
## Feature Engineering:
### Convert Time:
+ This class converts features that use days (Employed days, Age) to their absolute values because negative days of employment are not valid.
### Handle Retiree:
+ We will convert the employed days of retirees to 0 so that it is not considered an outlier
### Handle Skewness:
+ We employ cubic root transformation; this class aims to reduce the skewness in the distribution of Annual income and age. Skewed features can adversely impact the performance of our predictive model, and machine learning models generally benefit from normally distributed data.
### One Hot Encoding:
+ one-hot encoding on the specified categorical features and retains the feature names. Feature names are preserved instead of using the default array without names because they will be used for feature importance analysis
### Ordinal Encoding:
+ will convert the education level to an ordinal encoding. Here we use ordinal encoding instead of one-hot encoding because we know that the education level is ranked
Min-Max Scaling:
+ performs Min-Max scaling on specified numerical features while preserving the feature names. Scaling is essential because some numerical features have a wide range, and without scaling, machine learning algorithms might give more weight to features with larger values. Scaling ensures that all numerical features are on a similar scale (0 to 1), addressing this issue
### Balanced the Imbalanced[Oversampling]:
+ Here we can see that 88.29% have No risk, which means good applicants, the credit cards will be approved and 11.71% have risk, which means bad applicants, credit cards will not be approved.
+ have imbalanced data that need to balanced using SMOT sampling before training our model+
+ Here 0 means No and 1 means Yes

## Data Preprocessing Pipeline:
+ Creating the data preprocessing pipeline using the built-in sklearn function Pipeline.
+ The pipeline calls each class sequentially, starting from the outlier remover to the Target oversampling class.
+ The dataset will be transformed consecutively from the first class to the next one until the end.
+ The pipeline will be stored in a variable called 'pipeline.'
+ We will then call 'fit_transform' on that variable, passing our dataframe that we want to transform and return the result.

## Model Training:
+ First, we create a dictionary of models and their corresponding names. This dictionary will be utilized to loop through all the models and train them without the need to repeatedly write them.
+ We  designed a function  for plotting the feature importance of the model. Feature importance ranks the features based on their contribution, whether it be more or less, to the model prediction
+ This function is used to obtain the predicted values (y predictions) of the model using cross-validation with k folds, where k is set to 10

## Confusion Matrix Function:
+ function for plotting the confusion matrix of each algorithm. The confusion matrix visualizes the performance of a classification algorithm by showing the counts of true positive, true negative, false positive, and false negative predictions.

## Score Function:
+ This additional function is designed to print the classification report, which provides a comprehensive summary of the performance of a classification model. The report includes metrics such as precision, recall, f1-score, support, and accuracy

## How to choose best model?
+ We choose the best model based on precision,recall and ROC curve of the trained model
### When recall considerd?
+ When the economy is thriving, people experience a sense of prosperity and employment. Typically, money is inexpensive, and the risk of default is minimal due to economic stability and low unemployment. Financial institutions can manage the risk of default, and therefore, they are not overly stringent in granting credit. The institution can accommodate some high-risk clients as long as the majority of credit card holders are reliable clients, namely those who consistently repay their credit on time and in full. In such a scenario, achieving a high recall (sensitivity) is considered.
### When precision considered?
+ When the economy is in decline, individuals face job losses and financial setbacks, often stemming from the stock market and other investment platforms. Many people find it challenging to fulfill their financial responsibilities. Consequently, financial institutions adopt a more conservative approach in extending credit or loans. The institution cannot risk providing credit to numerous clients who may struggle to repay. In this situation, the financial institution prioritizes having a smaller but more reliable clientele, even if it results in denying credit to some creditworthy clients. In such cases, achieving good precision (specificity) is considered.
## Best Model
+ After analyzing the ROC curve and recall of all the trained models, it is apparent that our best-performing model is Gradient Boosting with accuracy 88.48

## More Important and Less Important Features
+ Based on this project’s analysis, Annual_income, familymember headcount , and employment_days are the three most predictive features in determining whether an applicant will be approved for a credit card. Other features like age and working type_income are also helpful. The least useful features are the type of housing_type  and car ownership.

## Deployment on streamlit:
### To deploy this project on streamlit share, follow these steps:
+ First, We  upload files on Github, including a requirements.txt file
+  Go to Streamlit share
+ Login with Github, Google, etc.
+ Click on the new app button
+ Select the Github repo name, branch, python file with the streamlit codes
+ Then save and deploy!


### Deployed Application link
+ https://preditct-credit-card-appoval.onrender.com/
+ https://deploypredictcreditcardapproval.streamlit.app/

## Technologies Used:
+ Python
+ Machine learning
+ Scikit-learn
+ NumPy
+ Pandas
+ Streamlit Share
+ Render

