import dlib
import cv2
# 載入並初始化檢測器
detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("cannot open camear")
    exit(0)
j=0
while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 檢測臉部
    dets = detector(frame_new, 1)
   
    # 查詢臉部位置
    for i, face in enumerate(dets):
       
        # 繪製臉部位置
        
        #儲存臉部圖片
        img1=frame[face.top():face.bottom(),face.left():face.right()]
        
    img_name = '%s/%d.jpg'%('C:/Users/vince/face/train/vincent',j)
    cv2.imwrite(img_name,frame)   
    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),(0,255,0), 3)    
    cv2.imshow("Camera", frame) 
   
    j+=1
        
    if (j>100):
          break
        
    key = cv2.waitKey(1)
    if key == 27:
        break
camera.release()    
cv2.destroyAllWindows()