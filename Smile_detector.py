import cv2

#loading the cascade
face_cascade = cv2.CascadeClassifier('E:\ML\\Computer_vision\\codes\\Module_1_Face_Recognition\\Module_1_Face_Recognition\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('E:\\ML\\Computer_vision\\codes\\Module_1_Face_Recognition\\Module_1_Face_Recognition\\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('E:\\ML\\Computer_vision\\codes\\Module_1_Face_Recognition\\Module_1_Face_Recognition\\haarcascade_smile.xml')

#function for detecting face and eye
def detect(gray,frame):
    #harr classifier to detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # face will be a 2 dimensioal  which has the value of cordinate where the faces has detected
    # faces will contain 4 tuples x , y , h(height), w(width)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #detect eyes inside the detected face 
        rif_gray = gray[y:y+h, x:x+w]
        # same for the color image
        rif_frame = frame[y:y+h, x:x+w]
        #detect eyes in the rif_gray region
        eyes = eye_cascade.detectMultiScale(rif_gray, 1.3, 5) 
        #for smile detection
        smile = smile_cascade.detectMultiScale(rif_gray, 1.7, 22)
        #draw eyes rectangle on color image
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(rif_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
        #draw when smile detected
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(rif_frame, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
    return frame
#open webcam for internal web cam use parameter 0
# for external use 1
video_capture = cv2.VideoCapture(0)

#will take input till we dont press 'q'
while True:
    #detect color images in video
    _,frame=video_capture.read()
    #convert it to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #pass to the function
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    #to break the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#close the webcam
video_capture.release()
#destroy the window
cv2.destroyAllWindows()

