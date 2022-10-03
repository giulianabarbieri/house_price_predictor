import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import xgboost as xgb
import time

labelencoder = LabelEncoder()
xgb_reg = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

class HouseScorerModel:
    eliminadas = []
    model_performance = {}
    features_importance = {}
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.get_data()

    def clean_data(self,df):
        # we will use that if the standard deviation is zero, this column has the same value for all the rows, therefore it does not carry information to the training
        cols = df.columns
        std = df[cols].std()
        cols_to_drop = std[std==0].index
        self.eliminadas = cols_to_drop
        print("Eliminamos las columnas:" , cols_to_drop)
        df = df.drop(cols_to_drop, axis=1)
        # TODO podriamos poner un warning del estilo : tal columna tiene datos 99% nan o 99% iguales
        return [df, cols_to_drop ]
    
    def encode_data(self,df):
        cols = df.columns
        num_cols = df._get_numeric_data().columns
        categorical_columns = list(set(cols) - set(num_cols))
        # Assigning numerical values and storing in the same column
        for categorical_column in categorical_columns:
            df[categorical_column] = labelencoder.fit_transform(df[categorical_column])
            print("The column:" , categorical_column, " was encoded.")
        return df

    def get_data(self):
        # Read from csv
        start_read = time.time()
        df = pd.read_csv(self.data_path)
        end_read = time.time()
        print('Reading CSV took', str(end_read - start_read),' seconds.')
        
        # Cleaning data
        start_clean = time.time()
        df = self.clean_data(df)[0]
        end_clean = time.time()
        print('Cleaning data took', str(end_clean - start_clean),' seconds.')
        start_encode = time.time()
        
        # Encoding data
        df = self.encode_data(df)
        end_encode = time.time()
        print('Encoding data took', str(end_encode - start_encode),' seconds.')
        return df 
    
    def get_model(self):
        df = self.data
        X = df.drop(columns = ['sale_price'])
        y = df.sale_price
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return xgb_reg, X_train, X_test, y_train, y_test
    
    def fit(self):
        xgb_reg, X_train, X_test, y_train, y_test = self.get_model()
        xgb_reg.fit(X_train,y_train)
        self.features_importance = xgb_reg.get_booster().get_score(importance_type='weight')
        pred = xgb_reg.predict(X_test)
        rmse = np.sqrt(MSE(y_test, pred))
        print("Root mean square error: ", rmse)
        mape = MAPE(y_test, pred)
        print("Mean absolute porcentage error: ", mape)
        self.model_performance = {'model_performance': {'root_mean_square_error': rmse, 'mean_absolute_porcentage_error': mape} }
        
        return 
    
    def get_model_performance(self):
        return self.model_performance

    def get_prediction(self, data_point):
        data_point_df = pd.DataFrame(data_point, index=[0])
        # data is clean and encoded
        data_point_df = data_point_df.drop(self.eliminadas, axis=1)
        data_point_df = self.encode_data(data_point_df)
        prediction = xgb_reg.predict(data_point_df)
        responce = {"Prediction": prediction[0],
                   "Top Features": self.features_importance}
        return responce 
        
    

    
    



