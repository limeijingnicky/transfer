import cv2 as cv
import mediapipe as mp
import time



##使用摄像头
cap = cv.VideoCapture(0)

mphands = mp.solutions.hands
hands=mphands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime=0
pTime=0

while True:
    success,img = cap.read()

    #将图片转换为rgb格式
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #当检测到手的坐标时
    if results.multi_hand_landmarks:
        #当有多个手对象时,分别提取每一个对象的值,并连接起来
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mphands.HAND_CONNECTIONS)

    #计算帧
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #将帧速标记在视频图片上
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv.imshow("image",img)
    cv.waitKey(1)

