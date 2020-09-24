import dlib                     
import cv2                      
import numpy as np
import os
from keras.preprocessing import image
from keras.models import load_model
import math
import serial
from time import sleep
# model
model = load_model("C:/Users/as100/Downloads/others.h5") 

class face_emotion():

    def __init__(self):
        
        self.detector = dlib.get_frontal_face_detector()
       
        self.predictor = dlib.shape_predictor("C:/Users/as100/Downloads/shape_predictor_68_face_landmarks.dat")

        
        self.cap = cv2.VideoCapture(1)
       
        self.cap.set(3, 480)

    def learning_face(self):
       
        face_width_max = 0
        face_width_min = 0
        count=0
        success=0
        while(self.cap.isOpened()):
            
            flag, im_rd = self.cap.read()
            
            k = cv2.waitKey(100)
            
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            
            faces = self.detector(img_gray, 0)
         
            font = cv2.FONT_HERSHEY_SIMPLEX
          
            if(len(faces)!=0):
                
                for i in range(len(faces)):
                    
                    for k, d in enumerate(faces):
                       
                        self.face_width = d.right() - d.left()
                        
                        shape = self.predictor(im_rd, d)

                        face_width_new=(shape.part(15).x-shape.part(34).x)/(shape.part(28).y-shape.part(9).y)                                                
                        face_width_right=(shape.part(15).x-shape.part(34).x)/(shape.part(28).y-shape.part(9).y)
                        face_width_left=(shape.part(34).x-shape.part(3).x)/(shape.part(28).y-shape.part(9).y) 
                        
                    if(face_width_max==0):
                        face_width_max = face_width_new
                        face_width_min = face_width_new
                    else:
                        if(face_width_new > face_width_max):
                            face_width_max = face_width_new
                                
                        if(face_width_new < face_width_min):
                            face_width_min = face_width_new
                                
                    if((face_width_max-face_width_min) > 0.3 and face_width_right-face_width_left < 0.07 and face_width_right-face_width_left > -0.07):
                        success = 1
                        break
                    count+=1
                    if(face_width_right-face_width_left < 0.07 and face_width_right-face_width_left > -0.07):
                        cv2.imwrite("C:/Users/as100/Downloads/20200528/prediction/0.png", im_rd)
                        

                if(count==20 or success == 1):
                    print("")
                    break
            else:
             
                cv2.putText(im_rd, "No Face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
          
            im_rd = cv2.putText(im_rd, "Esc: quit", (20, 450), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
           
            if(cv2.waitKey(10) == 27):
                break
            
            cv2.imshow("camera", im_rd)
       
        self.cap.release()
        
        cv2.destroyAllWindows()
        return success


def load_data(path):
    files = os.listdir(path)
    images = []
    for f in files:
        img_path = path + f
        img = image.load_img(img_path , grayscale = False , target_size = (100,100))
        img_array = image.img_to_array(img)
        images.append(img_array)
        
    data = np.array(images)
    print("-------Loading data-------")
    return data
    

def face_prediction(success):
    if(success==1):
        path= "C:/Users/as100/Downloads/20200528/prediction/"
        face_cut(path,path)
        images  = load_data(path)
        images /=255
        predictions = model.predict_on_batch(images)
       

        print('%.3f' %(predictions[0,0]))
        print('%.3f' %(predictions[0,1]))
        print('%.3f' %(predictions[0,2]))
        print('%.3f' %(predictions[0,3]))
        print('%.3f' %(predictions[0,4]))
        if( predictions[0,0] >= 0.7):
            print('Borong')
        elif( predictions[0,1] >= 0.7):
            print('Hao')
        elif( predictions[0,2] >= 0.7):
            print('Vincent')
        elif( predictions[0,3] >= 0.7):
            print('Teng')
        else:
            print('other')
        if(predictions[0,0] >= 0.7 or predictions[0,1] >= 0.7 or predictions[0,2] >= 0.7 or predictions[0,3] >= 0.7):
            return 1
        else:
            return 0
            
def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
   
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

   
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

 
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
  
    return hull_mask


def merge_add_alpha(img_1, mask):
  
    r_channel, g_channel, b_channel = cv2.split(img_1)
    if mask is not None:
        alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
        alpha_channel *= mask*255
    else:
        alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def merge_add_mask(img_1, mask):
    if mask is not None:
        height = mask.shape[0]
        width = mask.shape[1]
        channel_num = mask.shape[2]
        for row in range(height):
            for col in range(width):
                for c in range(channel_num):
                    if mask[row, col, c] == 0:
                        mask[row, col, c] = 0
                    else:
                        mask[row, col, c] = 255

        r_channel, g_channel, b_channel = cv2.split(img_1)
        r_channel = cv2.bitwise_and(r_channel, mask)
        g_channel = cv2.bitwise_and(g_channel, mask)
        b_channel = cv2.bitwise_and(b_channel, mask)
        res_img = cv2.merge((b_channel, g_channel, r_channel))
    else:
        res_img = img_1
    return res_img

def get_landmarks(image):
    predictor_model = 'C:/Users/as100/Downloads/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    if(len(rects)==0):
        landmarks="-1"
        return "-1"
    else:
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])
            
    return landmarks

def single_face_alignment(face, landmarks):
    eye_center = ((landmarks[36, 0] + landmarks[45, 0]) * 1. / 2,  
                  (landmarks[36, 1] + landmarks[45, 1]) * 1. / 2)
    dx = (landmarks[45, 0] - landmarks[36, 0])  
    dy = (landmarks[45, 1] - landmarks[36, 1])

    angle = math.atan2(dy, dx) * 180. / math.pi  
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  
    align_face = cv2.warpAffine(face, RotateMatrix, (face.shape[1], face.shape[0])) 
    return align_face

def face_cut(path,out_path):
    count=0
    for item in os.listdir(path):
        img_path = os.path.join(path,item)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        landmarks = get_landmarks(image)
        if(landmarks=="-1"):
            print("Fail")
        else:
            mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)
            image_bgra = merge_add_mask(image , mask)
            image_ = single_face_alignment(image_bgra,landmarks)
            landmarks = get_landmarks(image_)
            if(landmarks=="-1"):
                print("Fail")
            else:
                min_x = landmarks[0,0]
                max_x = landmarks[0,0]
                min_y = landmarks[0,1]
                max_y = landmarks[0,1]
                i = 1
                for i in range(67):
                    if(max_x < landmarks[i+1,0]):
                        max_x = landmarks[i+1,0]
                    if(min_x > landmarks[i+1,0]):
                         min_x = landmarks[i+1,0]
                    if(max_y < landmarks[i+1,1]):
                        max_y = landmarks[i+1,1]
                    if(min_y > landmarks[i+1,1]):
                        min_y = landmarks[i+1,1]
                img = image_[min_y : min_y+(max_y - min_y)  , min_x : min_x + (max_x - min_x)]
                img_100 = cv2.resize(img, (100, 100))
                cv2.imwrite("C:/Users/as100/Downloads/20200528/prediction/%d.png"%(count), img_100)
                count+=1
    print("OK")
    
if __name__ == "__main__":
    my_face = face_emotion()
    success = my_face.learning_face()
    if(success==1):
        print("success")
    else:
        print("Error")
    unlock = face_prediction(success)
    COM_PORT = 'COM3'
    BAUD_RATES = 9600
    com = serial.Serial(COM_PORT,BAUD_RATES)
    if(unlock==1):
        
        print("unlock")
        com.write(b'ON\n')
        sleep(2)
        com.write(b'ON\n')
        sleep(2)
        print('End')
    else:
        print("Fail")
        
    com.close()
