import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from sklearn.model_selection import train_test_split # 정제한 데이터를 훈련 / 테스트 데이터로 분할
from sklearn.linear_model import LinearRegression # 선형회귀 모델
import os
import s3fs

#def get_clean_data(dataset):
#    dataset = dataset.dropna(axis=0, subset=['부채비율', '신평사등급점수', '종합점수']) # (부채비율 컬럼의 데이터가 na인 row 삭제)
#    dataset = dataset[ (dataset['부도정보'] != 1.0) & (dataset['종합점수'] != '#VALUE!') ]
#    # (부도정보 컬럼의 데이터가 1인 row 삭제, 종합점수 컬럼의 데이터가 '#VALUE!'인 row 삭제)
#    dataset = dataset[(dataset != '-').all(axis=1)] # '-'와 동일한 데이터를 가진 row 삭제
#    dataset = dataset.fillna(0) # na값 0으로 대체
#    dataset = dataset.dropna() # #REF 삭제
#    dataset = dataset.reset_index(drop=True) # 인덱스 재정렬
#    dataset = dataset.replace(',', '', regex=True) # , 삭제

#    dataset = dataset.astype({
#        '부채비율': float, '매출액': float, '매출액 영업이익률': float, '총자산 영업이익률': float, '자산규모': float,
#        '이자보상비율': float, '자기자본 순이익률': float, '차입금의존도': float,
#        '매출액증가율(1~9단계)': float, '유동비율': float, '종합시공능력': float, '전문설비시공능력': float,
#        '의료재단등급(1~6)': float, '신평사등급/watch(1~7단계)': float, '주요주주지분(1~4단계)': float, '시장금리변동(1~5단계)': float,
#        '상장여부(1~4단계)': float, '배서인정보(1~4단계)': float, '과거이력(1~4단계)': float , '업력(년수)': float,
#        '종합점수': float
#    })

#    x = dataset[['부채비율', '매출액', '매출액 영업이익률', '총자산 영업이익률', '자산규모', '이자보상비율', '자기자본 순이익률', '차입금의존도',
#                '매출액증가율(1~9단계)', '유동비율', '종합시공능력', '전문설비시공능력', '의료재단등급(1~6)', '신평사등급/watch(1~7단계)',
#                '주요주주지분(1~4단계)', '시장금리변동(1~5단계)', '상장여부(1~4단계)', '배서인정보(1~4단계)', '과거이력(1~4단계)', '업력(년수)',
#                ]]
#    y = dataset[['종합점수']]

#    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

#    classifier = LinearRegression()
#    classifier.fit(x_train, y_train)

#    return classifier, x_train, x_test, y_train, y_test

#def plot_scatter(x, c, y): # 산점도 객체 생성
#    fig, ax = plt.subplots()
#    ax.scatter(x, y, alpha=0.4)
#    ax.set_xlabel(c)
#    ax.set_ylabel('종합 점수')
#    ax.set_title(f'{c}와 종합 점수의 상관관계')
#    return fig

#dataset = pd.read_csv('276CSS.csv', dtype={"사업자번호": str})

#classifier, x_train, x_test, y_train, y_test = get_clean_data(dataset)

#y_pred = classifier.predict(x_test)

st.set_page_config(layout='wide')
st.header('276 CSS ML?')

# 연결 개체 만들기
# 'anon=False'는 익명이 아님을 의미합니다. 즉, 액세스 키를 사용하여 데이터를 가져옵니다.
fs = s3fs.S3FileSystem(anon=False)

# 파일 내용을 검색합니다.
# st.cache_data를 사용하여 쿼리가 변경되거나 10분 후에만 다시 실행합니다.
@st.cache_data(ttl=600)
def read_file(filename):
    with fs.open(filename) as f:
        return pd.read_csv(f)

content = read_file("276ml/276CSS.csv")

st.dataframe(content)

#st.write('AWS_ACCESS_KEY_ID')
#print('os.getenv', '\n', os.getenv('AWS_ACCESS_KEY_ID'))
#st.write(os.getenv('AWS_SECRET_ACCESS_KEY'))

#st.write(f'훈련 세트 정확도: {round(classifier.score(x_train, y_train), 3)  * 100}%') # 훈련 세트 평가
#st.write(f'테스트 세트 정확도: {round(classifier.score(x_test, y_test), 3)  * 100}%') # 테스트 세트 평가

#plt.rc('font', family='AppleGothic') 
#plt.rcParams['axes.unicode_minus'] = False #한글 폰트 사용시 마이너스 폰트 깨짐 해결
##matplotlib 패키지 한글 깨짐 처리 끝

#st.write('아래 두 그래프는 같은 그래프리지만 크기를 지정할수 없어 작게 만들기 위해 아래와 같이 출력')
#fig, ax = plt.subplots()
#ax.scatter(y_test, y_pred, alpha=0.4)
#ax.set_xlabel('예측 종합 점수')
#ax.set_ylabel('실제 종합 점수')
#ax.set_title(f'276 CSS ML')

#for i in range(0, 1):
#    cols = st.columns(2)
#    for col in cols:
#        col.pyplot(fig)

#st.write('[독립변수]')
#st.write('- 종합점수')

#st.write('[종속변수]')
#st.write('- 부채비율, 매출액, 매출액, 영업이익률, 총자산 영업이익률, 자산규모, 이자보상비율, 자기자본 순이익률, 차입금의존도')
#st.write('- 매출액증가율(1-9단계), 유동비율, 종합시공능력, 전문설비시공능력, 의료재단등급(1-6단계), 신평사등급/watch(1-7단계),')
#st.write('- 주요주주지분(1-4단계), 시장금리변동(1-5단계), 상장여부(1-4단계), 배서인정보(1-4단계), 과거이력(1-4단계), 업력(년수)')

#x_list = []
#x_list_columns = []

#for i in range(len(x_test.columns)):
#    x_list_columns.append(x_test.columns[i])
#    x_list.append( x_test[ x_test.columns[i] ] ) # 종속 변수 Columns array

#fig_list = []
#for i in range(len(x_list)):
#    fig = plot_scatter(x_list[i], x_list_columns[i], y_pred)
#    fig_list.append(fig)

#idx = 0
#for i in range(int(len(fig_list) / 2)):
#    cols = st.columns(2)
#    for col in cols:
#        col.pyplot(fig_list[idx])
#        idx += 1