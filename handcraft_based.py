# import library ต่าง ๆ 
import cv2
import numpy as np
import tqdm as t
import sklearn.neighbors as sn
import skimage.feature as skf
import matplotlib.pyplot as plt


import os
from matplotlib import pyplot as plt

Xtrain = []
Ytrain = []

# เรียกและเข้าถึงรูปภาพจาก Folder ทั้งหมด 10 Folder ของรูปภาพที่จะใช้ในการ Training
for i in range(10):
	# เปลี่ยน directory ของ path_train
	path_train = 'D:/Image/Tr/' + str(i+1) + '_left index' # ตัวอย่างชื่อไฟล์ Folder '1_left index' ถ้าชื่อเปลี่ยนต้องเปลี่ยนด้วย ตามชื่อ Folder ที่กำหนดไว้
	for fn in os.listdir(path_train):
		if fn.endswith('jpg'):
			img = plt.imread(os.path.join(path_train, fn), cv2.COLOR_BGR2GRAY) # ให้แปลงรูปภาพเป็นภาพสีเทา

			Xtrain.append(img)
			Ytrain.append(i)

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)

print(Xtrain)
print(Ytrain)


# Training Image Loader and Feature Extraction โดยใช้ วิธี เมทริกซ์ระดับสีเทาร่วม (Grey Level Co-occurrence Matrix : GLCM)
# ซึ่ง GLCM เป็น tool ที่ใช้สำหรับ สกัด feature Texture
featureTr = [];
labelTr = [];
paraQuantize = 32  # กำหนด parameter Quantization
paraAngle = [0, 45, 90, 135]   # กำหนดทิศทาง ทั้งหมด 4 ทิศทาง
paraDistance = [1, 2, 3]  # กำหนด offset

for i in range(70):
	img = (Xtrain[i] / (256/paraQuantize)).astype(int); # Image Quantization

	glcm = skf.greycomatrix(img, distances=paraDistance, angles=paraAngle,
	levels=paraQuantize, symmetric=True, normed=True)  # การนำรูปภาพเข้ามาแล้วคิด glcm --> return มาเป็น Co-occurrence Matrix ในทุก ๆ ทิศทางมาให้
	# ซึ่งจะ return ออกมาให้ทั้งหมด 12 อัน ได้แก่ 4 ทิศทาง * 3 offset ในกรณีนี้

	#คำนวณหาค่าทางสถิติต่าง ๆ
	featureCon = skf.greycoprops(glcm, 'contrast')[0]
	featureEne = skf.greycoprops(glcm, 'energy')[0]
	featureHom = skf.greycoprops(glcm, 'homogeneity')[0]
	featureCor = skf.greycoprops(glcm, 'correlation')[0]
	featureTmp = np.hstack((featureCon, featureEne, featureHom, featureCor))  # นำค่าที่ได้จากการคำนวณค่าสถิติต่าง ๆ มาต่อกันเป็นแนวนอน
	featureTr.append(featureTmp)

size = len(featureTr)
print(featureTr)  # แสดง featureTr ที่ได้
print(size)  # แสดงจำนวน featureTr ทั้งหมดที่ได้


# Testing Image Loader and Feature Extraction
Xtest = []
Ytest = []

# เรียกและเข้าถึงรูปภาพจาก Folder ทั้งหมด 10 Folder ของรูปภาพที่จะใช้ในการ Testing
for i in range(10):
	# เปลี่ยน directory ของ path_test
	path_test = 'D:/Image/Test/' + str(i+1) + ' test';  # ตัวอย่างชื่อไฟล์ Folder '1 test' ถ้าชื่อเปลี่ยนต้องเปลี่ยนด้วย ตามชื่อ Folder ที่กำหนดไว้
	for fn in os.listdir(path_test):
		if fn.endswith('jpg'):
			img = plt.imread(os.path.join(path_test, fn), cv2.COLOR_BGR2GRAY)  # ให้แปลงรูปภาพเป็นภาพสีเทา

			Xtest.append(img)
			Ytest.append(i)

Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

print(Xtest)
print(Ytest)  # แสดงการแบ่งกลุ่มข้อมูลที่ถูกต้อง : เฉลย


check = []  # กำหนดไว้ เพื่อคำนวณ Percent ความถูกต้องของการทำนายข้อมูล
for i in range(30):
	img = cv2.imread(path_test,cv2.COLOR_BGR2GRAY)  # แปลงรูปภาพเป็นภาพสีเทา
	img = (Xtest[i] / (256/paraQuantize)).astype(int);  # Image Quantization

	glcm = skf.greycomatrix(img, distances=paraDistance, angles=paraAngle, levels=paraQuantize,
	symmetric=True, normed=True)

	featureCon = skf.greycoprops(glcm, 'contrast')[0]
	featureEne = skf.greycoprops(glcm, 'energy')[0]
	featureHom = skf.greycoprops(glcm, 'homogeneity')[0]
	featureCor = skf.greycoprops(glcm, 'correlation')[0]
	featureTs = [np.hstack((featureCon, featureEne, featureHom, featureCor))]
	#labelTs = 2

	#print(featureTs)

	# ใช้ การค้นหาเพื่อนบ้านที่ใกล้ที่สุด K ตัว (K-Neaerest Neighbors) และ การวัดความคล้ายโดยอาศัยระยะทาง (Distance) โดยใช้ Euclidean ในการแบ่งกลุ่มข้อมูล
	classifier = sn.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
	classifier.fit(featureTr, Ytrain)
	out = classifier.predict(featureTs)
	#print(out[0])
	check.append(out[0])
	#print(check)
	print('Answer is ' + str(out))  # แสดงคำตอบที่ได้จากการแบ่งกลุ่มข้อมูล

# คำนวณหาว่าทำนายถูกต้องกี่ตัว และ คำนวณหา Percent ความถูกต้องที่ทำนายได้
win = 0
for i in range(len(Ytest)):
	if Ytest[i] == check[i]:
		win += 1

print('Correct ' + str(win) + ' from 30 tests')           # แสดงจำนวนข้อมูลที่ทำนายถูก จากจำนวนข้อมูลที่ใช้ test 30 ตัว
print('Accuracy is ' + str((win*100)/len(Ytest)) + ' %')  # แสดง Percent ความถูกต้องของการทำนาย
