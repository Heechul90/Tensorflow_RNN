##### LSTM을 이용한 로이터 뉴스 카테고리 분류하기

### 필수 함수 및 라이브러리
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 불러온 데이터를 학습셋과 테스트셋으로 나누기
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000,   # 빈도수가 1~1000까지의 단어만 불러옴
                                                         test_split=0.2)   # 20%를 테스트 셋으로

# 데이터 확인하기
category = numpy.max(Y_train) + 1       # y_train의 종류를 구하니 46개의 카테고리(0부터 세기 때문에 1을 더해서 출력)
print(category, '카테고리')               # 46개 카테고리
print(len(X_train), '학습용 뉴스 기사')     # 학습용으로 8982개
print(len(X_test), '테스트용 뉴스 기사')    # 테스트용으로 2246개

print(X_train[0])                        # 해당 단어가 몇 번이나 나타나는지 세어 빈도에 따라 번호를 붙힘
print(Y_train[0])                        # 예를 들어 숫자가 5면 5번째로 빈도가 많은 단어
                                         # 이 작업을 위해서 tokenizer() 함수를 사용

# 데이터 전처리
# 각 기사의 단어 수가 제각각 다르므로 단어의 숫자를 맞춰야 함
# 이 때 데이터 전처리 함수 sequence를 이용
x_train = sequence.pad_sequences(X_train, maxlen=100)   # maxlen=100은, 단어수 100개로 맞추라는 의미
x_test = sequence.pad_sequences(X_test, maxlen=100)     # 단어수가 100보다 크면 그 이후는 버림, 모자랄 때는 0으로 채움
y_train = np_utils.to_categorical(Y_train)              # y 값은 원-핫 인코딩 처리를 하여 0~1사이 값으로 변환
y_test = np_utils.to_categorical(Y_test)



# 모델의 설정
model = Sequential()
model.add(Embedding(1000, 100))              # embedding 층은 데이터 전처리 과정을 통해 입력된 값을 받아 다음 층이 알아들을 수 있는 형태로 변환
                                             # Embedding('불러온 단어의 총 개수', '기사당 단어 수')
model.add(LSTM(100, activation='tanh'))      # LSTM은 RNN에서 기억 값에 대한 가중치를 제어
                                             # LSTM('기사당 단어수, 기타 옵션)
model.add(Dense(46, activation='softmax'))   # 출력층



# 모델의 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델의 실행
history = model.fit(x_train, y_train,
                    batch_size=100,
                    epochs=20,
                    validation_data=(x_test, y_test))

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))   # Test Accuracy: 0.7012

# 테스트셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()