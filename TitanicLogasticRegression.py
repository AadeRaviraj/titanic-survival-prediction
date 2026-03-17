import pandas as pd
import numpy as np 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


#-----------------------------------------------------------------------------------------------------
#   Function name :  PreserveModel
#   Description :    It is used to preserve model on secondary  
#   Parameter :      model , file name
#   Return :         None
#   Date :           14/03/2025
#   Author:          Raviraj Aade 
#----------------------------------------------------------------------------------------------------


def PreserveModel(model, filename):
    joblib.dump(model,filename)
    print("Model preserved successfully with name : ", filename)

#-----------------------------------------------------------------------------------------------------
#   Function name :  TrainTitanicModel
#   Description :    It Does split X,Y, training data , testing data 
#   Parameter :      df
#   Return :         None
#   Date :           14/03/2025
#   Author:          Raviraj Aade 
#----------------------------------------------------------------------------------------------------



def TrainTitanicModel(df):
    
    # Split features and labels 
    X = df.drop("Survived",axis = 1)
    Y = df["Survived"]
    
    print("\nFeatures :")
    print(X.head())
    print("\n Labels : ")
    print(Y.head())
    
    print("Shape of X :", X.shape)
    print("Shape of Y :", Y.shape)
    
    X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    
    print("Shape of X_train :", X_train.shape)
    print("Shape of X_test :", X_test.shape)
    print("Shape of Y_train :", Y_train.shape)
    print("Shape of Y_test :", Y_test.shape)
    
    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train,Y_train)
    
    print("MOdel Train successfully ")
    
    print("\n Intercept of model : ")
    print(model.intercept_)
    
    print("\n Coefficient of model ")
    for feature , coefficient in zip(X.columns, model.coef_[0]):
        print(feature,":",coefficient)
    
    PreserveModel(model,"marvellousTitanic.pkl")
    
        
    

#-----------------------------------------------------------------------------------------------------
#   Function name :  DisplayInfo
#   Description :    It Displays the formatted title
#   Parameter :      Title(Str)
#   Return :         None
#   Date :           14/03/2025
#   Author:          Raviraj Aade 
#----------------------------------------------------------------------------------------------------

def DisplayInfo(title):
    print("\n" + "=" * 70 )
    print(title)
    print("=" * 70 )


#-----------------------------------------------------------------------------------------------------
#   Function name :  ShowData
#   Description :    It shows basic information about dataset
#   Parameter :      df 
#                    df ->          Pandas dataframe object
#                    message
#                    message ->     Heading text to display
#   Return :         None
#   Date :           14/03/2025
#   Author:          Raviraj Aade 
#----------------------------------------------------------------------------------------------------

def ShowData(df, message):
    
    DisplayInfo(message)
    
    print("\nFirst 5 rows of dataset :")
    print(df.head())
    
    print("\nShape of dataset ")
    print(df.shape)
    
    print("\nColumn names : ")
    print(df.columns.tolist())
    
    print("\nMissing values in each column : ")
    print(df.isnull().sum())
    
    


#-----------------------------------------------------------------------------------------------------
#   Function name :  CleanTitanicData
#   Description :    It does Preprocessing
#                    It handles missing values
#                    It removed unnecessary columns 
#                    It Converts text data to numeric format 
#                    It does encoding to categorical columns 
#   Parameter :      df   ->    Pandas Dataframe 
#                    
#   Return :         df   ->    Clean Pandas dataframe 
#   Date :           14/03/2025
#   Author:          Raviraj Aade 
#----------------------------------------------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original Data")
    print(df.head())
    
    # Removed Unnecessary columns 
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]
    
    
    print("\n Columns to be dropped : ")
    print(existing_columns)
    
    # drop the unwanted columns 
    df = df.drop(columns = existing_columns)   
    DisplayInfo("Step 2 : Data After column removal")
    print(df.head())
    
    # handle age column 
    
    if "Age" in df.columns:
        print("Age  column before filling missing values")
        print(df["Age"].head(10))
        
        # coerce ->  Invalid value gets converted as NaN 
        df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
        
        age_median = df["Age"].median()
        
        # Replace missing values with median
        df["Age"] = df["Age"].fillna(age_median)
        
        print("Age column after preprocessing : ")
        print(df["Age"].head(10))
    
    # handle fair column 
    if "Fare" in df.columns:
        print("\n Fare Column before preprocessing ")
        print(df["Fare"].head(10))
        
        # coerce ->  Invalid value gets converted as NaN 
        df["Fare"]  = pd.to_numeric(df["Fare"], errors="coerce")
        
        fare_median = df["Fare"].median()
        
        print("\n Median of far is : ",fare_median)
        
        # Replace missing values with median
        df["Fare"] = df["Fare"].fillna(fare_median)
            
        print("\n Fare column after preprocessing : ")
        print(df["Fare"].head(10))
    
    # Handle Embarked column
    if "Embarked" in df.columns:
        print("\n Embarked Column before preprocessing ")
        print(df["Embarked"].head(10))
        
        # Convert the data into string 
        df["Embarked"]  = df["Embarked"].astype(str).str.strip()
        
        # Remove Missing values 
        df["Embarked"] = df["Embarked"].replace(["nan","None",""],np.nan)
        
        # Get most frequent value (mode)
        Embarked_mode =df["Embarked"].mode()[0]
            
        print("\nMode of Embarked column",Embarked_mode)
        
        df["Embarked"] = df["Embarked"].fillna(Embarked_mode)
        
        
        print("\n Embarked column after preprocessing : ")
        print(df["Embarked"].head(10))
    
    
    # handle Sex column 
    if "Sex" in df.columns:
        print("\n Sex Column before preprocessing ")
        print(df["Sex"].head(10))
        
        # coerce ->  Invalid value gets converted as NaN 
        df["Sex"]  = pd.to_numeric(df["Sex"], errors="coerce")
        
        
        print("\n Sex column after preprocessing : ")
        print(df["Sex"].head(10))
    
    
    DisplayInfo("Data After Preprocessing  ")
    print(df.head())
    
    print("\n Missing values after preprocessing ")
    print(df.isnull().sum())
    
    # Encode Embarked Column 
    df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)
    print("\n Data Before Encoding ")
    
    print(df.head())
    
    print("Shape of dataset : ",df.shape)
        
    # Convert Boolean Columns into integer 
    for col in df.columns:
        if df[col].dtype == bool :
            df[col] = df[col].astype(int)
    
    print("\n Data after Encoding ")
    
    print(df.head())
    
    
    
    
    
    return df


#-----------------------------------------------------------------------------------------------------
#   Function name :  MarvellousTitanicLogistic
#   Description :    This is main pipeline Controller
#                    It loads y=the dataset , show raw data 
#                    It Preprocess the dataset & train the model 
#   Parameter :      DataPath of dataset  file
#   Return :         None
#   Date :           14/03/2025
#   Author:          Raviraj Aade 
#----------------------------------------------------------------------------------------------------


def TitanicLogisticReg(DataPath):
    DisplayInfo("Step 1: Loading the dataset ")
    
    df = pd.read_csv(DataPath)
    ShowData(df,"Initial Dataset")
    
    df = CleanTitanicData(df)
    
    TrainTitanicModel(df)
    
    
    
    
    
    
#-----------------------------------------------------------------------------------------------------
#   Function name :  main
#   Description :    Starting point of application
#   Parameter :      None
#   Return :         None
#   Date :           14/ 03/2025
#   Author:          Raviraj Aade 
#----------------------------------------------------------------------------------------------------

def main():
    TitanicLogisticReg("TitanicDataset.csv")

if __name__ == "__main__":
    main()