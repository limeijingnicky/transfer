import cv2 as cv
import mediapipe as mp
import time



##使用摄像头
cap = cv.VideoCapture(0)

mpFace= mp.solutions.face_detection
face=mpFace.FaceDetection()
mpDraw = mp.solutions.drawing_utils
cTime=0
pTime=0

while True:
    success,img = cap.read()

    #将图片转换为rgb格式
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results =face.process(imgRGB)

    #当检测到坐标时
    if results.detections:
        for id, detecion in enumerate(results.detections):
            # mpDraw.draw_detection(img,detecion)
            # print(id,detecion) #数据包含了脸部的bbox和6个点的坐标（眼睛，鼻子和嘴）
            # print(detecion.score)
            # print(detecion.location_data.relative_bounding_box)
            bboxC=detecion.location_data.relative_bounding_box
            ih,iw,ic=img.shape

            bbox=int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                 int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img, bbox, (255, 0, 255), 5)

            mouth=detecion.location_data.relative_keypoints[3]
            cx, cy = int(mouth.x * iw), int(mouth.y * ih)
            # print(cx, cy)

            cv.circle(img,(cx,cy),2,(255,0,255),2,cv.FILLED)

            # 将得分写在图片上
            cv.putText(img, f'{int(detecion.score[0]* 100)}%', (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    #计算帧
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #将帧速标记在视频图片上
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


    cv.imshow("image",img)
    cv.waitKey(1)

