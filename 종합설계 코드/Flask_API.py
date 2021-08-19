# by 종두 Flask API 구현
# basic
import pandas as pd
import numpy as np
# sql DB
import pymysql
from sqlalchemy import create_engine
# Flask server
from flask import Flask, request
# pickle
import pickle

app = Flask (__name__)

def open_model(df_data):
    analysis_data = df_data.loc[:, ['UID','USERAGE', 'USERJOB'  , 'USERSEX']]
    kmeans_open = pickle.load(open("kmeans.pkl", "rb"))
    pred = kmeans_open.predict(analysis_data)
    df_data['cluster'] = pred
    return df_data

# json
from pandas import json_normalize
# New user 회원가입 시 클러스터링 작업 후 DB에 적재
@app.route('/newUserClustering', methods=['POST'])
def newUserClustering():
    #json 값 받아오기
    jsonData = request.get_json()
    newUser = json_normalize(jsonData)
    newUser.columns = ['UID', 'USERNAME', 'USERPW', 'USERAGE', 'USERAREA', 'USERJOB', 'USERINTEREST', 'USERMAIL', 'USERPHONE', 'USERSEX']
    #encoding
    for i in ['USERJOB', 'USERSEX']:
        newUser = df_preprocessing_oneHot(newUser,i)

    newUser = open_model(newUser)

    # DB에 회원가입 + cluster 데이터 input
    # conn_x userClustering에서 conn 사용하고 있어서 임시로 conn_x로 지정
    conn_x = pymysql.connect(host='localhost', user='root',
                                passwd='1234', db='roadmap', charset='utf8')

    engine = create_engine("mysql+pymysql://root:"+"1234"+"@localhost/roadmap", encoding='utf-8')
    conn_x = engine.connect()
    newUser.to_sql(name='test_input', con=engine, if_exists='append', index=False)

    return 0


# KMeans
from sklearn.cluster import KMeans
# userInfo data로 군집화 진행 및 interest 재정의
@app.route('/userClustering')
def clustering():

    sql_text = "select * from USERINFO"
    data = db_connect(sql_text)
    '''
    # DB에서 사용자 테이블 가져오기
    conn = pymysql.connect(host='localhost',user='root',
                            passwd='1234', db='roadmap', charset='utf8')
    
    curs = conn.cursor()
    sql = "select * from userInfo"
    curs.execute(sql)
    data = curs.fetchall()
    '''
    
    # 사용할(필요한) column만 수집
    data = pd.DataFrame(data, columns=['UID','USERNAME','USERID','USERPW','USERAGE','USERAREA','USERJOB','USERINTEREST','USERMAIL','USERPHONE', 'USERSEX', 'USERCHECK'])
    userInfo = data.loc[:, ['UID', 'USERAGE', 'USERJOB','USERSEX']]

    # LabelEncoding 함수 실행
    #
    for i in ['USERJOB' ,'USERSEX']:
        userInfo = df_preprocessing_oneHot(userInfo,i)

    # KMeans 사용해서 Clustering
    kmeans = KMeans(n_clusters=10, init='k-means++',max_iter=1000, random_state=156)
    kmeans.fit(userInfo)

    # KMeans 결과 값 column 추가
    userInfo['cluster'] = kmeans.labels_

    # 기존 data에 cluster column 추가
    data['cluster'] = userInfo['cluster']

    # Group의 interest 값을 담을 DF
    userMostInterest = pd.DataFrame()

    # 각 Group의 interest를 종합해서 DF로 재구성
    for i in range(0,10):
        A = pd.Series(data[data['cluster'] == i]['USERINTEREST'].value_counts())
        A = A.sort_values(ascending=False)
        A = A.index
        userMostInterest[i] = pd.Series(A)

    userMostInterest.columns = ['A','B','C','D','E','F','G','H','I','J']

    # DB에 적재
    # ################# DB 경로 확인하기 ##################
    engine = create_engine("mysql+pymysql://root:"+"fhemaoqservice!1234"+"@35.197.102.174/ROADMAP", encoding='utf-8')
    conn = engine.connect()
    userMostInterest.to_sql(name='USER_GROUP', con=engine, if_exists='replace', index=False)
    
    #  ######### conn.close() 이걸 쓰면 저장이 되지 않는다? 해결방법 찾기 ########

    # model save
    pickle.dump(kmeans, open("kmeans.pkl", "wb"))
    return 0

# encoding
from sklearn.preprocessing import LabelEncoder
# String -> LabelEncoding
def df_preprocessing_oneHot(userInfo=None, column=None):
    encoder = LabelEncoder()
    encoder.fit(userInfo[column].unique())
    encode_col = encoder.transform(userInfo[column])
    userInfo.drop(column, axis=1, inplace=True)
    userInfo[column] = encode_col
    
    return userInfo

# 모델 평가
from sklearn.metrics import silhouette_samples, silhouette_score
def model_eval(model, userInfo_eval):
    score = silhouette_samples(userInfo_eval.loc[:,'USERAGE':'USERSEX'], userInfo_eval['cluster'])
    userInfo_eval['silhouette_coeff'] = score
    average_score = silhouette_score(userInfo_eval.loc[:,'Age':'Sex'], userInfo_eval['cluster'])
    print('User 데이터 셋 Sihouette Analysis Score: {0:.3f}' .format(average_score))

    # 간단한 시각화
    # userInfo_eval['silhouette_coeff'].hist()

    # 그룹마다의 silhouette_coeff 값
    userInfo_eval.groupby('cluster')['silhouette_coeff'].mean()

# roadmap Data 전처리
def preprocessing_Recommend(rating, roadmaps):
    rating.drop('timestamp', axis=1, inplace=True)
    rating_roadmaps = pd.merge(rating, roadmaps, on='rid')
    ratings_matrix = rating_roadmaps.pivot_table(values='rating', index='userid', columns='title')
    ratings_matrix = ratings_matrix.fillna(0)
    
    return ratings_matrix

# DB connect
def db_connect(sql_text):
    conn = pymysql.connect(host='35.197.102.174',user='root',
                            passwd='fhemaoqservice!1234', db='ROADMAP', charset='utf8')

    curs = conn.cursor()
    sql = sql_text
    curs.execute(sql)
    data = curs.fetchall()
    conn.close()

    return data

# 사용자와 비슷한 user 추출
from sklearn.metrics.pairwise import cosine_similarity
def get_user_sim(ratings_matrix, userId):
    user_sim = cosine_similarity(ratings_matrix, ratings_matrix)
    user_sim = pd.DataFrame(data=user_sim, index=ratings_matrix.index, columns=ratings_matrix.index)
    similar_users = user_sim.loc[userId,:]
    similar_user = similar_users.sort_values(ascending=False)[1:].index[0:10]  
    return similar_user

# 사용자가 안본 roadmap 추출
def unseen_roadmap(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId, :]
    movies_list = user_rating[user_rating == 0].index.tolist()
    
    return movies_list
# roadmap 추천
def recomm_roadmap_by_userId(pred_df, userId, movies_list, top_n=10):
    recomm_movies = pred_df.loc[userId, movies_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

import random
# 사용자 기반 추천
@app.route('/userRecommend')
def userRecommend():
    userId = request.args.get('userId', '0')
    rating_sql = "select * from roadmap_ratings"
    ratings = db_connect(rating_sql)

    roadmap_sql = 'select * from roadmaps'
    roadmaps = db_connect(roadmap_sql)

    # 데이터 전처리
    ratings_matrix = preprocessing_Recommend(ratings, roadmaps)

    # 유사도 값을 가지는 사용자 row
    similar_user = get_user_sim(ratings_matrix, userId)

    # 사용자가 보지 않은 roadmap 추출
    unseen_roadmaps = unseen_roadmap(ratings_matrix, userId)

    # 사용자의 추천 roadmap 30개 추출
    recomm_roadmaps = recomm_roadmap_by_userId(ratings_matrix, userId, unseen_roadmaps, 30)
    recomm_roadmap_result = []

    # 사용자와 유사한 사용자들의 추천 roadmap을 비교하여 겹치는 roadmap을 출력(나와 비슷한 사용자들의 취향 고려)
    for i in similar_user:
        recomm_roadmaps_similar_user = recomm_roadmap_by_userId(ratings_matrix, i, unseen_roadmaps, 30)
        for j in range(len(recomm_roadmaps.index)):
            if recomm_roadmaps.index[j] in recomm_roadmaps_similar_user.index:
                recomm_roadmap_result.append(recomm_roadmaps.index[j])
    recomm_roadmap_result = set(recomm_roadmap_result)
    recomm_roadmap_result = random.sample(recomm_roadmap_result, 5)

    return recomm_roadmap_result

# item 유사도 측정
def get_item_sim(ratings_matrix_T):
    item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)
    item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix_T.index, columns=ratings_matrix_T.index)

    return item_sim_df

# rating 값 예측
def predict_rating_topsim(rating_arr, item_sim_arr, n=20):
    pred = np.zeros(rating_arr.shape)
    
    for col in range(rating_arr.shape[1]):
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1] ]
        
        for row in range(rating_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(rating_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))
    return pred

# Item 기반 추천
@app.route('/itemRecommed')
def itemRecommed():
    # 사용자 ID 입력받고 DB에 접속해 Data select
    userId = request.args.get('userId', '0')
    rating_sql = "select * from roadmap_ratings"
    ratings = db_connect(rating_sql)

    roadmap_sql = 'select * from roadmaps'
    roadmaps = db_connect(roadmap_sql)

    # Data 전처리
    ratings_matrix = preprocessing_Recommend(ratings, roadmaps)
    ratings_matrix_trans = ratings_matrix.transpose()

    # 유사 item 
    item_sim = get_item_sim(ratings_matrix_trans)

    # 유사도 높은 item 찾기
    rating_pred = predict_rating_topsim(ratings_matrix.values, item_sim.values, n=20)
    ratings_pred_matrix = pd.DataFrame(data=rating_pred, index=ratings_matrix.index, columns=ratings_matrix.columns)

    # 보지 않은 roadmap 추출
    unseen_list = unseen_roadmap(ratings_matrix, userId)

    # 유사도가 높고 평점이 좋은 item 5가지 선별
    recomm_roadmap_item = recomm_roadmap_by_userId(ratings_pred_matrix, userId, unseen_list, top_n=10)
    recomm_roadmap_item = pd.DataFrame(data=recomm_roadmap_item.values, index=recomm_roadmap_item.index, columns=['pred_columns'])
    recomm_roadmap_item_result = random.sample(recomm_roadmap_item, 5)

    return recomm_roadmap_item_result

'''
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
'''
'''      
#Kmeans 군집 갯수를 입력해 시각화 확인
visualize_silhouette([ 2, 3, 4,5 ],  UserInfo)
visualize_silhouette([6, 7, 8, 9],  UserInfo)
visualize_silhouette([10, 11, 12, 13],  UserInfo)
'''

@app.route('./association_Search')
def association_Search():
    
    return 0

if __name__ == '__main__':
    app.run()