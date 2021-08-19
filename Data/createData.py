import pandas as pd
import random

dataframe = pd.DataFrame()

# col = ['Name', 'Age', 'Area', 'Job', 'Interest', 'Email', 'Phone_number', 'sex']

Name = []
Age = []
Area = []
Job = []
Interest = []
Email = []
Phone_number = []
Sex = []

area = ['서울특별시', '경기도', '충청남도', '충청북도', '강원도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도']
job = ['대학생', '웹개발자', '앱개발자', '정보보안가', '머신러닝 엔지니어', '데이터 사이언티스트']
interest = ['웹개발', '앱개발', '머신러닝', '딥러닝', '정보보안', '서버개발']
email = ['naver.com', 'daum.net', 'google.com', 'nate.com']
sex = ['남', '여']

for i in range(1,10001):
    Name.append(('a'+str(i)))
    Age.append(random.randrange(15,36))
    Area.append(area[random.randrange(0,10)])
    Job.append(job[random.randrange(0,6)])
    Interest.append(interest[random.randrange(0,6)])
    Email.append('a'+str(i)+email[random.randrange(0,3)])
    Phone_number.append(random.randrange(11111111111, 99999999999))
    Sex.append(sex[random.randrange(0,2)])
    


dataframe['Name'] = pd.Series(Name)
dataframe['Age'] = pd.Series(Age)
dataframe['Area'] = pd.Series(Area)
dataframe['Job'] = pd.Series(Job)
dataframe['Interest'] = pd.Series(Interest)
dataframe['Email'] = pd.Series(Email)
dataframe['Phone_number'] = pd.Series(Phone_number)
dataframe['Sex'] = pd.Series(Sex)

dataframe.to_csv('./test.csv', index=False, encoding='utf-8-sig')
print(dataframe)