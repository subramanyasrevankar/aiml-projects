  import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import HistGradientBoostingClassifier
   from sklearn.preprocessing import StandardScaler
   from sklearn.experimental import enable_hist_gradient_boosting 
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 # load the dataset 
  df=pd.read_csv('Loan_Default.csv')
  
  #map yes/no binary categorial columns to 0/1
  binary_cols=['Gender','approv_in_adv', 'business_or_commercial', 
                  'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'submission_of_application']
   for col in binary_cols:
       df[col]=df[col].map({'No':0,'Yes':1,'True':1,'False':0,'Male':1,'Female':0})
  
  #drop columns that are not useful or duplicates for modeling 
   df.drop(columns=['ID', 'year', 'Region', 'Security_Type', 'co-applicant_credit_type', 
                    'submission_of_application', 'total_units', 'Secured_by', 'loan_limit', 
                    'approv_in_adv', 'loan_type', 'loan_purpose', 'business_or_commercial', 
                    'open_credit', 'Credit_Worthiness'], inplace=True)
  
 df['dtir1']=pd.to_numeric(df['dtir1'],errors='coerce')
  
  # #step 5:drop rows where target variable status is nan 
   df=df[df['Status'].notna()]
  
  # #separate x and y values 
   x=df.drop('Status',axis=1)
   y=df['Status']
  
  # #one hot encode categorial columns (if any)
   x=pd.get_dummies(x,drop_first=True) #it will solve the problem of multiclinearity first drop the column and after that go to next one
  
   #step 8:train test_split 
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  
   #initialize the histGradientboosting classifier(only label coding is needed ) #can use logistic regression but not because in this some of the input features are nan so to use logistic regression first we need to convert it into not nan using simple imputer 
  # 9. Handle missing values using SimpleImputer (mean strategy) #only impute x before we train and it will handle the nan values . 
   imputer = SimpleImputer(strategy='mean')
   X_imputed = imputer.fit_transform(x) with feature scaling 
   model=HistGradientBoostingClassifier(max_iter=100)
  
   model.fit(x_train,y_train)
  
  # #predict on the test data 
   pred=model.predict(x_test)
   print(accuracy_score(y_test,pred))
