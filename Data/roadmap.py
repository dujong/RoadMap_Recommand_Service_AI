import pandas as pd
import random

dataframe = pd.DataFrame()

ts_ms = pd.date_range(start='2020-01-01',end=None, periods=10000, freq='H')

# col = ['Root', 'trunk', 'branch', 'leaf', 'book']
#['웹', '인공지능', '앱', '정보보안', '서버개발자' 등등]
Root = []
root = ['웹', '인공지능']
 
# 웹: ['프론트엔드', '백엔드', '데브옵스'] 인공지능: ['Fundamentals', 'Data Scientist', 'Data Engineer']
Trunk = []
trunk = [['프론트엔드', '백엔드', '데브옵스'], ['Fundamentals', 'Data Scientist', 'Data Engineer']]

Branch = []
branch = [['HTML', 'CSS', 'JavaScript', 'Git'], ['os사용법', '언어', 'DB', 'API'], 
                                                        ['Server_manager', 'teminal', 'infrastructure as Code'],
            ['Database Basics', 'Data Frames&Series', 'Data Formats', 'Tabular Data'], ['Probability Theory', 'Discrete distributions', 'Summary statistics', 'Estimation'],
                                                        ['Summary of Data Formate', 'Data Integration', 'Data Survey', 'Data Fusion']]

Leaf = []

leaf = [['HTML+CSS+자바스크립트 웹 표준의 정석', '생활코딩! HTML+CSS+자바스크립트', 'Do it! HTML5+CSS3 웹 표준의 정석'],
             ['CSS 설계 실전 가이드', '자바스크립트 닌자 비급', '모두의 HTML5 & CSS3'], 
                ['모던 웹을 위한 Javascript + jQuery 입문', '자바스크립트 + jQuery 완전정복 스터디 3', '자바스크립트 완벽가이드'],
                 ['팀 개발을 위한 GIt GitHub 시작하기', '만들면서 배우는 Git + GitHub 입문', 'Do it! 지옥에서 온 문서관리자 깃&깃허브 입문'],
                 
                 ['os사용법111', 'os사용법222', 'os사용법333'],
                  ['언어111', '언어222', '언어333'],
                   ['DB구축111', 'DB구축222', 'DB구축333'],
                    ['API사용해보기111', 'API사용해보기222', 'API사용해보기333'],
                    
                     ['Server_manager111', 'Server_manager222', 'Server_manager333'],
                      ['teminal111', 'teminal222', 'teminal333'],
                       ['Code111', 'Code222', 'Code333'],
                       
                        ['Database Basics111', 'Database Basics222', 'Database Basics333'],
                         ['Data Frames&Series111', 'Data Frames&Series222', 'Data Frames&Series333'],
                          ['Data Formats111', 'Data Formats222', 'Data Formats333'],
                           ['Tabular Data111', 'Tabular Data222', 'Tabular Data333'],
                           
                            ['Probability Theory111', 'Probability Theory222', 'Probability Theory333'],
                             ['Discrete distributions111', 'Discrete distributions222', 'Discrete distributions333'],
                              ['Summary statistics111', 'Summary statistics222', 'Summary statistics333'],
                               ['Estimation111', 'Estimation222', 'Estimation333'],
                               
                                ['Summary of Data Formate111', 'Summary of Data Formate222', 'Summary of Data Formate333'],
                                 ['Data Integration111', 'Data Integration222', 'Data Integration333'],
                                  ['Data Survey111', 'Data Survey222', 'Data Survey333'],
                                   ['Data Fusion111', 'Data Fusion222', 'Data Fusion333']]

for i in range(0,10000):
    Root.append(root[random.randrange(0,2)])
    if Root[i] == '웹':
        Trunk.append(trunk[0][random.randrange(0,3)])
        #'HTML', 'CSS', 'JavaScript', 'Git'
        if Trunk[i] == '프론트엔드':
            Branch.append(branch[0][random.randrange(0,4)])
            if Branch[i] == 'HTML':
                Leaf.append(leaf[0][random.randrange(0,3)])
            elif Branch[i] == 'CSS':
                Leaf.append(leaf[1][random.randrange(0,3)])
            elif Branch[i] == 'JavaScript':
                Leaf.append(leaf[2][random.randrange(0,3)])
            elif Branch[i] == 'Git':
                Leaf.append(leaf[3][random.randrange(0,3)])

        #'os사용법', '언어', 'DB', 'API'
        elif Trunk[i] == '백엔드':
            Branch.append(branch[1][random.randrange(0,4)])
            if Branch[i] == 'os사용법':
                Leaf.append(leaf[4][random.randrange(0,3)])
            elif Branch[i] == '언어':
                Leaf.append(leaf[5][random.randrange(0,3)])
            elif Branch[i] == 'DB':
                Leaf.append(leaf[6][random.randrange(0,3)])
            elif Branch[i] == 'API':
                Leaf.append(leaf[7][random.randrange(0,3)])

        #'Server_manager', 'teminal', 'infrastructure as Code'
        elif Trunk[i] == '데브옵스':
            Branch.append(branch[2][random.randrange(0,3)])
            if Branch[i] == 'Server_manager':
                Leaf.append(leaf[8][random.randrange(0,3)])
            elif Branch[i] == 'teminal':
                Leaf.append(leaf[9][random.randrange(0,3)])
            elif Branch[i] == 'infrastructure as Code':
                Leaf.append(leaf[10][random.randrange(0,3)])

    else :
        Trunk.append(trunk[1][random.randrange(0,3)])

            #'Database Basics', 'Data Frames&Series', 'Data Formats', 'Tabular Data'
        if Trunk[i] == 'Fundamentals':
            Branch.append(branch[3][random.randrange(0,4)])
            if Branch[i] == 'Database Basics':
                Leaf.append(leaf[11][random.randrange(0,3)])
            elif Branch[i] == 'Data Frames&Series':
                Leaf.append(leaf[12][random.randrange(0,3)])
            elif Branch[i] == 'Data Formats':
                Leaf.append(leaf[13][random.randrange(0,3)])
            elif Branch[i] == 'infrastructure as Code':
                Leaf.append(leaf[14][random.randrange(0,3)])

            #'Probability Theory', 'Discrete distributions', 'Summary statistics', 'Estimation'
        elif Trunk[i] == 'Data Scientist':
            Branch.append(branch[4][random.randrange(0,4)])
            if Branch[i] == 'Database Basics':
                Leaf.append(leaf[15][random.randrange(0,3)])
            elif Branch[i] == 'Data Frames&Series':
                Leaf.append(leaf[16][random.randrange(0,3)])
            elif Branch[i] == 'Data Formats':
                Leaf.append(leaf[17][random.randrange(0,3)])
            elif Branch[i] == 'infrastructure as Code':
                Leaf.append(leaf[18][random.randrange(0,3)])
            #'Summary of Data Formate', 'Data Integration', 'Data Survey', 'Data Fusion'
        elif Trunk[i] == 'Data Engineer':
            Branch.append(branch[5][random.randrange(0,4)])
            if Branch[i] == 'Summary of Data Formate':
                Leaf.append(leaf[19][random.randrange(0,3)])
            elif Branch[i] == 'Data Integration':
                Leaf.append(leaf[20][random.randrange(0,3)])
            elif Branch[i] == 'Data Survey':
                Leaf.append(leaf[21][random.randrange(0,3)])
            elif Branch[i] == 'Data Fusion':
                Leaf.append(leaf[22][random.randrange(0,3)])


dataframe['Time'] = ts_ms
dataframe['Root'] = pd.Series(Root)
dataframe['Trunk'] = pd.Series(Trunk)
dataframe['Branch'] = pd.Series(Branch)
dataframe['Leaf'] = pd.Series(Leaf)
dataframe.to_csv('./testRoadMap.csv', index=False, encoding='utf-8-sig')

print(dataframe)
