import numpy as np

# npz 파일을 읽어옴
data = np.load('mnist.npz')

# 파일 안에 어떤 데이터가 있는지 확인
print(list(data))

# 각 데이터에 접근
train_images = data['x_test']
train_labels = data['x_train']
test_images = data['y_train']
test_labels = data['y_test']

# 데이터 확인
print('Train Images Shape:', train_images.shape)
print('Train Labels Shape:', train_labels.shape)
print('Test Images Shape:', test_images.shape)
print('Test Labels Shape:', test_labels.shape)

# 파일 닫기
data.close()