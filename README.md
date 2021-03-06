# Project-Image-Recognition &#9757;
# ชื่อกลุ่ม แอนฟิล์ดนรกทีมเยือน &#128038;
## รายชื่อสมาชิก<br>
  <br>1.นาย ภากร ศุภนิมิตวาสนา 60070071<br />
  <br>2.นางสาว กชวรรณ  มาสทอง 62070229<br />
  <br>3.นาย ณัฐภาส คำผุด 62070243<br />
  <br>4.นางสาว วิชญา  ฉิมสา 62070270<br />
  
## Project 2 : Handcraft_base
### วิธี Run Code &#128187;	
```
python handcraft_based.py
```

ต้องเปลี่ยน directory ที่อยู่ของไฟล์ handcraft_based.py ก่อน ตามที่อยู่ที่เก็บไฟล์ไว้ <br />
หรือ ใช้วิธี “ลากและวาง” ไฟล์ handcraft_based.py ลงใน Anaconda Prompt <br />

โดยจะแบ่งเป็น <br />
  1. Train - โดยใช้ วิธี เมทริกซ์ระดับสีเทาร่วม (Grey Level Co-occurrence Matrix : GLCM) <br />
  2. Test – โดยใช้การค้นหาเพื่อนบ้านที่ใกล้ที่สุด K ตัว (K-Neaerest Neighbors) และ การวัดความคล้ายโดยอาศัยระยะทาง (Distance) โดยใช้ Euclidean ในการแบ่งกลุ่มข้อมูล <br />
<br>
เมื่อ Run เสร็จ จะแสดงผลลัพธ์จากการทำนายของแบบจำลอง โดย Answer is [ผลลัพธ์], <br />
แสดงจำนวนข้อมูลที่ทำนายถูก จากจำนวนข้อมูลที่ใช้ test 30 ตัว (เปลี่ยนจำนวนที่จะใช้ทดสอบได้) <br />
และแสดง Percent ความถูกต้องของการทำนาย <br />

### วิธีการเปลี่ยนรูป Dataset &#128190;
สำหรับ Train <br />
![path_train](https://github.com/Witchaya270/Project-Image-Recognition/blob/main/Image%20for%20README/%E0%B9%80%E0%B8%9B%E0%B8%A5%E0%B8%B5%E0%B9%88%E0%B8%A2%E0%B8%99%20path_train.png?raw=true) <br />
ให้เปลี่ยนค่าที่ path_train ในบรรทัดที่ 19 และสามารถเปลี่ยนชื่อไฟล์ได้ ดังตัวอย่างนี้ <br />
ตัวอย่างชื่อไฟล์ Folder '1_left index' ถ้าชื่อเปลี่ยนต้องเปลี่ยนด้วย ตามชื่อ Folder ที่กำหนดไว้ <br />
ในบรรทัดที่ 21 คือ นามสกุลของไฟล์ เช่น .jpg, .gif, .png เป็นต้น ซึ่งเปลี่ยนได้ตามนามสกุลไฟล์ที่ต้องการ <br />

สำหรับ Test <br />
![path_test](https://github.com/Witchaya270/Project-Image-Recognition/blob/main/Image%20for%20README/%E0%B9%80%E0%B8%9B%E0%B8%A5%E0%B8%B5%E0%B9%88%E0%B8%A2%E0%B8%99%20path_test.png?raw=true) <br />
ให้เปลี่ยนค่าที่ path_test ในบรรทัดที่ 69 และสามารถเปลี่ยนชื่อไฟล์ได้ ดังตัวอย่างนี้ <br />
ตัวอย่างชื่อไฟล์ Folder '1 test' ถ้าชื่อเปลี่ยนต้องเปลี่ยนด้วย ตามชื่อ Folder ที่กำหนดไว้ <br />
ในบรรทัดที่ 71 คือ นามสกุลของไฟล์ เช่น .jpg, .gif, .png เป็นต้น ซึ่งเปลี่ยนได้ตามนามสกุลไฟล์ที่ต้องการ <br />

## Project 3 : Learning_base 
### วิธี Run Code Train
```
python learning_based_train_run.py
```
เมื่อรันเสร็จจะได้ train_learning_based.pt ออกมา ซึ่งก็คือแบบจำลอง


วิธีเปลี่ยน Dataset<br />
![path_train](https://github.com/Witchaya270/Project-Image-Recognition/blob/main/Image%20for%20README/03.jpg?raw=true) <br />

### วิธี Run Code Test
```
python learning_based_test_run.py
```
เมื่อรันเสร็จจะแสดงผลลัพธ์ของการทำนายว่าตรงกับข้อมูลชุดทดสอบหรือไม่ แล้วแสดงผลของการทำนายออกมา

วิธีเปลี่ยน Dataset<br />
![path_train](https://github.com/Witchaya270/Project-Image-Recognition/blob/main/Image%20for%20README/04.jpg?raw=true) <br />

เนื่องจากไฟล์ model มีขนาดใหญ่เกินไป รบกวนดาวน์โหลด <a href="https://drive.google.com/file/d/13z8WBP3UlfZV282edHnlWi0gY8UKauBo/view?usp=sharing">จากที่นี่</a>
ไปใส่ไว้ในโฟลเดอร์ model
