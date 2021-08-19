import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# 데이터 가져오기
Data = pd.read_csv('./userInfo.csv')
Data.head()

# 분석 feature만 가져오기
UserInfo = Data.loc[:, ['Age', 'Area', 'Job', 'Sex']]
UserInfo.head()

# 분석 feature만 가져오기
UserInfo = Data.loc[:, ['Age', 'Area', 'Job', 'Sex']]
UserInfo.head()

# String형으로된 feature를 Category형으로 바꾸어 분석 가능하게
def df_preprocessing_oneHot(UserInfo=None, column=None):
    encoder = LabelEncoder()
    encoder.fit(UserInfo[column].unique())
    encode_col = encoder.transform(UserInfo[column])
    UserInfo.drop(column, axis=1, inplace=True)
    UserInfo[column] = encode_col
    
    return UserInfo

# Area, Job, Sex의 OneHotEncoding
for i in ['Area', 'Job', 'Sex']:
    UserInfo = df_preprocessing_oneHot(UserInfo,i)
    
UserInfo.head()

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=14, random_state=0).fit(UserInfo)
gmm_cluster_labels = gmm.predict(UserInfo)

UserInfo['gmm_cluster'] = gmm_cluster_labels

User_cluster_result = UserInfo['gmm_cluster'].value_counts()
User_cluster_result.sort_values(ascending=False)

### 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 인자로 받아 클러스터링 결과를 시각화하는 함수  
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise=True
        else :
            cluster_legend = 'Cluster '+str(label)
        
        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70,\
                    edgecolor='k', marker=markers[label], label=cluster_legend)
        
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',\
                        edgecolor='k', marker='$%d$' % label)
    if isNoise:
        legend_loc='upper center'
    else: legend_loc='upper right'
    
    plt.legend(loc=legend_loc)
    plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(UserInfo.iloc[:,:])

UserInfo['pca_x'] = pca_transformed[:,0]
UserInfo['pca_y'] = pca_transformed[:,1]
UserInfo['pca_z'] = pca_transformed[:,2]
UserInfo.head(3)

marker0_ind = UserInfo[UserInfo['gmm_cluster']==0].index
marker1_ind = UserInfo[UserInfo['gmm_cluster']==1].index
marker2_ind = UserInfo[UserInfo['gmm_cluster']==2].index
marker3_ind = UserInfo[UserInfo['gmm_cluster']==3].index
marker4_ind = UserInfo[UserInfo['gmm_cluster']==4].index
marker5_ind = UserInfo[UserInfo['gmm_cluster']==5].index
marker6_ind = UserInfo[UserInfo['gmm_cluster']==6].index
marker7_ind = UserInfo[UserInfo['gmm_cluster']==7].index
marker8_ind = UserInfo[UserInfo['gmm_cluster']==8].index
marker9_ind = UserInfo[UserInfo['gmm_cluster']==9].index
marker10_ind = UserInfo[UserInfo['gmm_cluster']==10].index
marker11_ind = UserInfo[UserInfo['gmm_cluster']==11].index
marker12_ind = UserInfo[UserInfo['gmm_cluster']==12].index
marker13_ind = UserInfo[UserInfo['gmm_cluster']==13].index

%matplotlib notebook

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(UserInfo.loc[marker0_ind,'pca_x'], UserInfo.loc[marker0_ind,'pca_y'],UserInfo.loc[marker0_ind,'pca_z'],  marker='o')
ax.scatter(UserInfo.loc[marker1_ind,'pca_x'], UserInfo.loc[marker1_ind,'pca_y'],UserInfo.loc[marker1_ind,'pca_z'],  marker='s')
ax.scatter(UserInfo.loc[marker2_ind,'pca_x'], UserInfo.loc[marker2_ind,'pca_y'],UserInfo.loc[marker2_ind,'pca_z'],  marker='^')
ax.scatter(UserInfo.loc[marker3_ind,'pca_x'], UserInfo.loc[marker3_ind,'pca_y'],UserInfo.loc[marker3_ind,'pca_z'],  marker='o')
ax.scatter(UserInfo.loc[marker4_ind,'pca_x'], UserInfo.loc[marker4_ind,'pca_y'],UserInfo.loc[marker4_ind,'pca_z'],  marker='s')
ax.scatter(UserInfo.loc[marker5_ind,'pca_x'], UserInfo.loc[marker5_ind,'pca_y'],UserInfo.loc[marker5_ind,'pca_z'],  marker='^')
ax.scatter(UserInfo.loc[marker6_ind,'pca_x'], UserInfo.loc[marker6_ind,'pca_y'],UserInfo.loc[marker6_ind,'pca_z'],  marker='o')
ax.scatter(UserInfo.loc[marker7_ind,'pca_x'], UserInfo.loc[marker7_ind,'pca_y'],UserInfo.loc[marker7_ind,'pca_z'],  marker='s')
ax.scatter(UserInfo.loc[marker8_ind,'pca_x'], UserInfo.loc[marker8_ind,'pca_y'],UserInfo.loc[marker8_ind,'pca_z'],  marker='^')
ax.scatter(UserInfo.loc[marker9_ind,'pca_x'], UserInfo.loc[marker9_ind,'pca_y'],UserInfo.loc[marker9_ind,'pca_z'],  marker='o')
ax.scatter(UserInfo.loc[marker10_ind,'pca_x'], UserInfo.loc[marker10_ind,'pca_y'],UserInfo.loc[marker10_ind,'pca_z'],  marker='s')
ax.scatter(UserInfo.loc[marker11_ind,'pca_x'], UserInfo.loc[marker11_ind,'pca_y'],UserInfo.loc[marker11_ind,'pca_z'],  marker='^')
ax.scatter(UserInfo.loc[marker12_ind,'pca_x'], UserInfo.loc[marker12_ind,'pca_y'],UserInfo.loc[marker12_ind,'pca_z'],  marker='o')
ax.scatter(UserInfo.loc[marker13_ind,'pca_x'], UserInfo.loc[marker13_ind,'pca_y'],UserInfo.loc[marker13_ind,'pca_z'],  marker='s')

plt.show()