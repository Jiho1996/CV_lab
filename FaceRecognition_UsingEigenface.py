import cv2
import numpy as np

# 학습 데이터 및 라벨 데이터 배열 생성
faces = []
labels = []

# 1.txt~50.jpg and 1.txt~16.txt 로드 후 배열에 삽입
for i in range(1, 16):
    image = cv2.imread("image/" + str(i) + ".jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # GrayScale로 변환
    faces.append(image) # faces라는 배열에 image 삽입

    f = open("image/" + str(i) + ".txt", "r")
    line = int(f.readline()) # txt파일의 첫번째 줄을 읽어와 line에 삽입
    labels.append(line) # labels이라는 배열에 label 데이터 삽입

# FaceRecognizer를 생성 후 변수에 대입
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# train_data와 label_data 사이의 관계를 학습
face_recognizer.train(faces, np.array(labels))

# Test
#test_image = cv2.imread("test_image_Ryo.jpg")
#test_image = cv2.imread("test_image_Lee.jpg")
#test_image = cv2.imread("test_image_Kim.jpg")
#test_image = cv2.imread("test_image_Park.jpg")
test_image = cv2.imread("test_image_lee.jpg")

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) # GrayScale로 변환

# 학습된 FaceRecognizer 모델이 추측한 값을 Return
predict_label, confidence = face_recognizer.predict(test_image)

if(predict_label == 1):
    print("군복무 후임입니다.")
elif(predict_label == 2):
    print("본인 입니다.")
elif(predict_label == 3):
    print("고등학교 친구입니다.")

print(confidence)
cv2.imshow("test_image", test_image)
cv2.waitKey()
