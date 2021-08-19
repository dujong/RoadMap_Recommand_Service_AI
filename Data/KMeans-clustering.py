import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
# String형으로된 feature를 Category형으로 바꾸어 분석 가능하게
def df_preprocessing_oneHot(UserInfo=None, column=None):
    encoder = LabelEncoder()
    encoder.fit(UserInfo[column].unique())
    encode_col = encoder.transform(UserInfo[column])
    UserInfo.drop(column, axis=1, inplace=True)
    UserInfo[column] = encode_col
    
    return UserInfo

# 데이터 가져오기
Data = pd.read_csv('./userInfo.csv')
Data.head()

# 분석 feature만 가져오기
UserInfo = Data.loc[:, ['Age', 'Area', 'Job', 'Sex']]
UserInfo.head()

# Area, Job, Sex의 OneHotEncoding
for i in ['Area', 'Job', 'Sex']:
    UserInfo = df_preprocessing_oneHot(UserInfo,i)
    
UserInfo.head()

# Kmeans 클러스터링을 통해서 군집화 값 도출
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(UserInfo)
# 값을 보여주는 속성 labels_
kmeans.labels_

# 분석된 군집화 값 cluster feature로 추가
UserInfo['cluster'] = kmeans.labels_
UserInfo.head()

# 데이터 차원축소를 통한 데이터 시각화
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(UserInfo.iloc[:,:])

UserInfo['pca_x'] = pca_transformed[:,0]
UserInfo['pca_y'] = pca_transformed[:,1]
UserInfo.head(3)

# cluster 값이 0, 1, 2 인 경우마다 별도의 Index로 추출
marker0_ind = UserInfo[UserInfo['cluster']==0].index
marker1_ind = UserInfo[UserInfo['cluster']==1].index
marker2_ind = UserInfo[UserInfo['cluster']==2].index

# cluster값 0, 1, 2에 해당하는 Index로 각 cluster 레벨의 pca_x, pca_y 값 추출. o, s, ^ 로 marker 표시
plt.scatter(x=UserInfo.loc[marker0_ind,'pca_x'], y=UserInfo.loc[marker0_ind,'pca_y'], marker='o') 
plt.scatter(x=UserInfo.loc[marker1_ind,'pca_x'], y=UserInfo.loc[marker1_ind,'pca_y'], marker='s')
plt.scatter(x=UserInfo.loc[marker2_ind,'pca_x'], y=UserInfo.loc[marker2_ind,'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusters Visualization by 2 PCA Components')
plt.show()

Data['cluster'] = UserInfo['cluster']
Data.head()

A = pd.DataFrame(Data[Data['cluster'] == 0]['Interest'].value_counts())
A['Interest_name'] = A.index.values
A.set_index('Interest_name')
A = A['Interest'].sort_values(ascending=False)
A.head()

B = pd.DataFrame(Data[Data['cluster'] == 1]['Interest'].value_counts())
B['Interest_name'] = B.index.values
B.set_index('Interest_name')
B = B['Interest'].sort_values(ascending=False)
B.head()

B = pd.DataFrame(Data[Data['cluster'] == 2]['Interest'].value_counts())
B['Interest_name'] = B.index.values
B.set_index('Interest_name')
B = B['Interest'].sort_values(ascending=False)
B.head()