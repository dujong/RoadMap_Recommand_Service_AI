import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

roadMap_df = pd.read_csv('./Data/roadMap.csv')
userInfo_df = pd.read_csv('./Data/userInfo.csv')

#DataFrame 정보 확인
def df_info(df=None):
    print(df.info())
    print(df.describe())

def df_value_counts(df=None, column=None):
    return df[column].value_counts()

#전처리
def df_preprocessing_null(df=None):
    print('Nan 값 :', df.isnull().sum())
    print('총 Nan 값:', df.isnull().sum().sum())
    if df.isnull().sum().sum() > 1000:
        df.dropna(axis=0, inplace=True)
    return df

def df_preprocessing_oneHot(df=None, column=None):
    encoder = LabelEncoder()
    encoder.fit(['남', '여'])
    encode_col = encoder.transform(df[column])
    df[column] = encode_col
    return df


abc = df_preprocessing_oneHot(userInfo_df, 'Sex')
abc.to_csv('./abc.csv', index=False, encoding='utf-8-sig')