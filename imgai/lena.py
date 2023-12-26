import cv2

# //칼라
img_color = cv2.imread('./lena.jpg', cv2.IMREAD_COLOR)
# //흑백
img_gray = cv2.imread('./lena.jpg', cv2.IMREAD_GRAYSCALE)

# //이미지 행렬차원 확인
print(type(img_color),img_color.shape)

img_resized = cv2.resize(img_color, (100,100))
cv2.imwrite('./d.jpg',img_resized)