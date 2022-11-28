from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(1)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
background=None

def gen_frames():  # generate frame by frame from camera
    global background
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray=cv2.GaussianBlur(gray,(21,21),0)
            if background is None:
                background=gray
                continue
            delta_frame=cv2.absdiff(background,gray)
            treshold_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
            treshold_frame=cv2.dilate(treshold_frame,None,iterations=0)

            (cntr,_)=cv2.findContours(treshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in cntr:
                if cv2.contourArea(contour)<1000:
                    continue
                (x,y,w,h)=cv2.boundingRect(contour)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)