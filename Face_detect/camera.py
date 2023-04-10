import cv2



def gen_frames():  
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def face_detect():  
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
            for x, y, width, height in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
                roi_gray = image_gray[y:y+height, x:x+width]
                roi_color = frame[y:y+height, x:x+width]
  
                # Detects eyes of different sizes in the input image
                eyes = eye_cascade.detectMultiScale(roi_gray) 
  
                #To draw a rectangle in eyes
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')