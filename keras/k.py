import os
import cv2
import numpy as np
from tensorflow.keras import layers, models, activations

model = models.Sequential()

model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Convolution2D(64, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(64, (3, 3), activation=activations.relu))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 모델에 가중치를 불러옴
model.load_weights('model_weights.h5')

# 외부 이미지("7.png")를 불러오고 전처리
test_img = cv2.imread("7.png", cv2.IMREAD_GRAYSCALE)
test_img = test_img / 255.0  # 0~1로 정규화
row, col, channel = test_img.shape[0], test_img.shape[1], 1
test_img = test_img.reshape((1, row, col, channel))


# 예측 결과 출력
confidence = model.predict(test_img.reshape((1, row, col, channel)))
predicted_digit = np.argmax(confidence, axis=1)[0]
print(f"모델이 예측한 숫자: {predicted_digit}")
