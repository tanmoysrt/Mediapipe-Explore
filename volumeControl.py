import cv2
import mediapipe as mp
import time
from subprocess import call

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


def distanceBetweenTwoXYPoints(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            landmark4 = handLms.landmark[4]
            landmark8 = handLms.landmark[8]
            cv2.line(img, (int(landmark4.x*w), int(landmark4.y*h)), (int(landmark8.x*w),int(landmark8.y*h)), (255, 255, 0), 5)
            # cv2.line(img,(0,0),(150,150),(255,255,255),15)


            distance_between = distanceBetweenTwoXYPoints(landmark4.x*w,landmark4.y*h, landmark8.x*w,landmark8.y*h)

            volume = distance_between/250*100

            if volume > 100 :
                volume = 100

            
            # This line of code is specific to ubuntu
            call(["amixer", "set", "Master", str(volume)+"%"])

            cv2.putText(img, str(f"{int(volume)} %"), (int(landmark4.x*w), int(landmark4.y*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)