import cv2 as cv
import mediapipe as mp
import time



##使用摄像头
cap = cv.VideoCapture(0)

mpPose= mp.solutions.pose
pose=mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cTime=0
pTime=0

while True:
    success,img = cap.read()

    #将图片转换为rgb格式
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    #当检测到坐标时
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x * w),int(lm.y * h)

            if id==11:
                cv.circle(img,(cx,cy),20,(255,0,255),cv.FILLED)

    #计算帧
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #将帧速标记在视频图片上
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv.imshow("image",img)
    cv.waitKey(1)

