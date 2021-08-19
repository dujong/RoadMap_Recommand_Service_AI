import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# by종두 데이터 가져오기
Data = pd.read_csv('./userInfo.csv')
Data.head()

# by종두 분석 feature만 가져오기
UserInfo = Data.loc[:, ['Age', 'Job', 'Sex']]
UserInfo.head()

# by종두 String형으로된 feature를 Category형으로 바꾸어 분석 가능하게
def df_preprocessing_oneHot(UserInfo=None, column=None):
    encoder = LabelEncoder()
    encoder.fit(UserInfo[column].unique())
    encode_col = encoder.transform(UserInfo[column])
    UserInfo.drop(column, axis=1, inplace=True)
    UserInfo[column] = encode_col
    
    return UserInfo

# by종두 Area, Job, Sex의 LabelEncoding
for i in ['Job', 'Sex']:
    UserInfo = df_preprocessing_oneHot(UserInfo,i)
    
UserInfo.head(10)
UserInfo.info()

# visualize_silhouette을 하기위한 import ***데이터 분포도 3D를 보려면 import 하면 안된다!!***
import matplotlib.pyplot as plt

# by종두 Kmeans의 군집 갯수를 시각화로 확인하기!!
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

#Kmeans 군집 갯수를 입력해 시각화 확인
visualize_silhouette([ 2, 3, 4,5 ],  UserInfo)
visualize_silhouette([6, 7, 8, 9],  UserInfo)
visualize_silhouette([10, 11, 12, 13],  UserInfo)

# by종두 위에서 확인한 KMeans 
kmeans = KMeans(n_clusters=10, init='k-means++',max_iter=500, random_state=156)
kmeans.fit(UserInfo)
np.unique(kmeans.labels_)

UserInfo['cluster'] = kmeans.labels_
UserInfo.head()


# 모델 평가
UserInfo_eval = UserInfo.loc[:, 'Age':'cluster']
UserInfo_eval.head()

from sklearn.metrics import silhouette_samples, silhouette_score

score_samples = silhouette_samples(UserInfo_eval.loc[:,'Age':'Sex'], UserInfo_eval['cluster'])
UserInfo_eval['silhouette_coeff'] = score_samples
UserInfo_eval.head()

average_score = silhouette_score(UserInfo_eval.loc[:,'Age':'Sex'], UserInfo_eval['cluster'])
print('User 데이터 셋 Sihouette Analysis Score: {0:.3f}' .format(average_score))

UserInfo_eval['silhouette_coeff'].hist()
UserInfo_eval.groupby('cluster')['silhouette_coeff'].mean()