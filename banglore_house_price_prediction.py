import pandas as pd
import numpy as np

df = pd.read_csv('bengaluru_house_prices.csv')
df.describe()

df.info()

df.head()

df.shape

for column in df.columns:
  print(df[column].value_counts())
  print("*"*20)

df.isnull().sum()

df.drop(columns=['area_type','availability','society','balcony'],inplace=True)

df.describe()

df.info()

df['location'].value_counts()

df['location'] = df['location'].fillna('sarjapur Road')

df['size'].value_counts()

df.isna().sum()

df['size'] = df['size'].fillna('2 BHK')

df['bath'] = df['bath'].fillna(df['bath'].median())

df.info()

df['bhk'] = df['size'].str.split().str.get(0).astype(int)

df[df.bhk > 20]

df['total_sqft'].unique()

def convertRange(x):
  temp = x.split('-')
  if len(temp) == 2:
    return (float(temp[0]) + float(temp[1]))/2
  try:
    return float(x)
  except:
    return None

df['total_sqft']=df['total_sqft'].apply(convertRange)

df.head()

"""Price Per Square feet"""

df['price_per_sqrt']=df['price']*100000/df['total_sqft']

df['price_per_sqrt']

df['location'].value_counts()

df['location']= df['location'].apply(lambda x: x.strip())
location_count= df['location'].value_counts()

location_count

location_count_less_10 = location_count[location_count<=10]
location_count_less_10

df['location']=df['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)

"""Outlier Detection and removal"""

df.describe()

(df['total_sqft']/df['bhk']).describe()

df = df[((df['total_sqft']/df['bhk']) >= 300)]
df.describe()

df.shape

df.price_per_sqrt.describe()

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqrt)

        st = np.std(subdf.price_per_sqrt)

        gen_df = subdf[(subdf.price_per_sqrt > (m-st)) & (subdf.price_per_sqrt <= (m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index =True)
    return df_output
df = remove_outliers_sqft(df)
df.describe()

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqrt),
                'std': np.std(bhk_df.price_per_sqrt),
                'count': bhk_df.shape[0]
            }

        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)

            if stats and stats['count'] > 5:
                exclude_indices = np.append (exclude_indices,bhk_df[bhk_df.price_per_sqrt< stats['mean']].index.values)

    return df.drop(exclude_indices,axis='index')

df = bhk_outlier_remover(df)

df.shape

df

df.drop(columns=['size','price_per_sqrt'],inplace=True)

"""Clean Data"""

df.head()

df.to_csv("Cleaned_data.csv")

x=df.drop(columns=['price'])
y=df['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print(x_train.shape)
print(x_test.shape)

"""Applying Linear Regression"""

column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['location']),
                                       remainder='passthrough')

scaler = StandardScaler()

lr = LinearRegression()

pipe = make_pipeline(column_trans,scaler,lr)

pipe.fit(x_train,y_train)

y_pred_lr = pipe.predict(x_test)

r2_score(y_test,y_pred_lr)

"""Applying Lasso

"""

lasso = Lasso()

pipe = make_pipeline(column_trans,scaler,lasso)

pipe.fit(x_train,y_train)

y_pred_lasso = pipe.predict(x_test)
r2_score(y_test,y_pred_lasso)

"""Applying Ridge"""

ridge = Ridge()

pipe = make_pipeline(column_trans,scaler,ridge)

pipe.fit(x_train,y_train)

y_pred_ridge = pipe.predict(x_test)
r2_score(y_test,y_pred_ridge)

print("No Regularization: ",r2_score(y_test,y_pred_lr))
print("Lasso: ",r2_score(y_test,y_pred_lasso))
print("Ridge: ",r2_score(y_test,y_pred_ridge))

import pickle

pickle.dump(pipe,open('RidgeModel.pkl','wb'))
