########## RNN을 이용한 텍스트 생성(Text Generation using RNN) ##########
#  다 대 일(many-to-one) 구조의 RNN을 사용하여 문맥을 반영해서 텍스트를 생성하는 모델을 만들어봅니다.

##### 1. LSTM을 이용하여 텍스트 생성하기
# 사용할 데이터는 뉴욕 타임즈 기사의 제목입니다. 아래의 링크에서 ArticlesApril2018.csv 데이터를 다운로드 합니다.
# 파일 다운로드 링크 : https://www.kaggle.com/aashita/nyt-comments


# 필수 함수
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

### 1.1 데이터에 대한 이해와 전처리
# 다운로드한 훈련 데이터를 데이터프레임에 저장
df = pd.read_csv('text_generation/dataset/nyt-comments/ArticlesApril2018.csv')
df.head()

# 열의 개수가 굉장히 많기 때문에 한 눈에 보기 어렵습니다.
# 어떤 열이 있고, 열이 총 몇 개가 있는지 출력
print('열의 개수: ', len(df.columns))
print(df.columns)

# 총 15개의 열이 존재합니다. 우리가 사용할 열은 제목에 해당되는 headline 열입니다.
# null 값이 있는지 확인
df['headline'].isnull().values.any()


# null값이 없는 것으로 확인되었습니다.
# headline 열에서 모든 신문 기사의 제목을 뽑아서 하나의 리스트로 저장
headline = []                              # 리스트 선언
headline.extend(list(df.headline.values))  # 헤드라인의 값들을 리스트로 저장
headline[:5]                               # 상위 5개만 출력

# 4번째, 5번째 샘플에 Unknown 값이 들어가있습니다.
# headline 전체에 걸쳐서 Unknown 값을 가진 샘플이 있을 것으로 추정됩니다.
# 비록 Null 값은 아니지만 지금 하고자 하는 실습에 도움이 되지 않는 노이즈 데이터이므로 제거해줄 필요가 있습니다.
# 제거하기 전에 현재 샘플의 개수를 확인해보고 제거 전, 후의 샘플의 개수를 비교해보겠습니다.
print('총 샘플의 개수 : {}'.format(len(headline)))             # 현재 샘플의 개수

headline = [n for n in headline if n != 'Unknown']           # Unknown 값을 가진 샘플 제거
print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline))) # 제거 후 샘플의 개수
headline[:5]

# 기존에 4번째, 5번째 샘플에서는 Unknown 값이 있었는데 현재는 제거가 된 것을 확인하였습니다.
# 이제 데이터 전처리를 수행합니다.
# 여기서 선택한 전처리는 구두점 제거와 단어의 소문자화입니다.
# 전처리를 수행하고, 다시 샘플 5개를 출력합니다.
def repreprocessing(s):
    s = s.encode('utf8').decode('ascii', 'ignore')
    return ''.join(c for c in s if c not in punctuation).lower() # 구두점 제거와 동시에 소문자화

text = [repreprocessing(x) for x in headline]
text[:5]

# 기존의 출력과 비교하면 모든 단어들이 소문자화되었으며 N.F.L.이나 Cheerleaders’ 등과 같이 기존에 구두점이 붙어있던 단어들에서 구두점이 제거되었습니다.
#
# 단어 집합(vocabulary)을 만들고 크기를 확인
t = Tokenizer()
t.fit_on_texts(text)
vocab_size = len(t.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)

# 총 3,494개의 단어가 존재합니다.
#
# 정수 인코딩과 동시에 하나의 문장을 여러 줄로 분해하여 훈련 데이터를 구성
sequences = list()

for line in text:                              # 1,214 개의 샘플에 대해서 샘플을 1개씩 가져온다.
    encoded = t.texts_to_sequences([line])[0]  # 각 샘플에 대한 정수 인코딩
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

sequences[:11]                                 # 11개의 샘플 출력

# 어떤 정수가 어떤 단어를 의미하는지 알아보기 위해 인덱스로부터 단어를 찾는 index_to_word를 만듭니다.
index_to_word={}
for key, value in t.word_index.items(): # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key

print('빈도수 상위 582번 단어 : {}'.format(index_to_word[582]))

# y 데이터를 분리하기 전에 전체 샘플의 길이를 동일하게 만드는 패딩 작업을 수행합니다.
# 패딩 작업을 수행하기 전에 가장 긴 샘플의 길이를 확인합니다.
max_len = max(len(l) for l in sequences)
print('샘플의 최대 길이 : {}'.format(max_len))

# 가장 긴 샘플의 길이인 24로 모든 샘플의 길이를 패딩
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences[:3])

# 맨 우측 단어만 레이블로 분리
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

print(X[:3])
print(y[:5]) # 레이블

# 레이블 데이터 y에 대해서 원-핫 인코딩을 수행
y = to_categorical(y, num_classes=vocab_size)
print(y[:5])

### 1.2 모델 설계하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, SimpleRNN


# 각 단어의 임베딩 벡터는 10차원을 가지고, 128의 은닉 상태 크기를 가지는 LSTM을 사용
# model = Sequential()
# model.add(Embedding(vocab_size, 10, input_length=max_len-1))   # y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
# model.add(LSTM(128))
# model.add(Dense(vocab_size, activation='softmax'))
# model.summary()
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, y, epochs=200, verbose=2)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1)) # 레이블을 분리하였으므로 이제 X의 길이는 5
model.add(SimpleRNN(128))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)


print("\n Test Accuracy: %.4f" % (model.evaluate(X, y)[1]))   # Test Accuracy: 0.9246

# sentence_generation을 만들어서 문장을 생성
def sentence_generation(model, t, current_word, n):                  # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word                                         # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n):                                               # n번 반복
        encoded = t.texts_to_sequences([current_word])[0]            # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)

    # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items():
            if index == result:                                      # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break                                                # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word                    # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word                             # 예측 단어를 문장에 저장

    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence

# 임의의 단어 'i'에 대해서 10개의 단어를 추가 생성
print(sentence_generation(model, t, 'i', 10))

# 임의의 단어 'how'에 대해서 10개의 단어를 추가 생성
print(sentence_generation(model, t, 'how', 10))
