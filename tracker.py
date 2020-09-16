#object tracker to improve thermal camera
#author: abhinandan.vellanki@gmail.com

#import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

class track():
    
    def __init__(self, video, tracker): 
        # #construct argument parser and parse arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-v", "--video", type=str, help="path to video") # '--video ' and specify path; default: opens webcam
        # ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV tracker type")# '--tracker ' and specify tracker to use; default: kcf 
        # args=vars(ap.parse_args())
        
        self.video = video
        self.tracker = tracker
        
        #extract OpenCV version info
        (major, minor) = cv2.__version__.split(".")[:2]

        #if version <=3.2, use special factory function to create object tracker
        if int(major) == 3 and int(minor) <3:
            tracker=cv2.Tracker_create(self.tracker.upper())

        #if version >=3.3, need to explicitly call tracker constructor
        else:
            OPENCV_TRACKERS={ #mapper, does not include GOTURN
                "csrt": cv2.TrackerCSRT_create,
                "kcf": cv2.TrackerKCF_create,
                "boosting": cv2.TrackerBoosting_create,
                "mil": cv2.TrackerMIL_create,
                "tld": cv2.TrackerTLD_create,
                "medianflow": cv2.TrackerMedianFlow_create,
                "mosse": cv2.TrackerMOSSE_create
            }
            tracker=OPENCV_TRACKERS[self.tracker]() #call constructor at runtime

        #initialize bounding box coordinates to track
        BBtrack = None

        #if no video path, choose webcam
        if self.video == None:
            print("Starting video capture...")
            try:
                vs=VideoStream(src=0).start()
                time.sleep(1.0)#time to start stream
            except:
                print("Unable to capture video from webcam!!")
                vs=None
        else:
            vs=cv2.VideoCapture(self.video)

        #initialize fps throughput estimator
        fps=None
    
    def start():
        #iterate over video frames
        while True:
            frame=vs.read()
            frame=frame[1] if not self.video == None else frame #handle separate cases for VideoStream or VideoCapture
            
            if frame is None:#reached end of stream
                break;
            
            #resize for faster processing and get dimensions
            frame=imutils.resize(frame, width=500)
            (H,W) = frame.shape[:2]
            
            #if an object is being tracked
            if BBtrack is not None:
                (success, box) = tracker.update(frame)#get new BB coordinates of selected box
                
                if success:
                    (x,y,w,h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) #draw green rectangle around new coordinates of tracking box
                    
                #update fps counter
                fps.update()
                fps.stop()
                
                #initialize info to be diplayed on frame
                info=[
                    ("Tracker", self.tracker),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(fps.fps()))                    
                ]
                
                #loop over info and add to frame
                for (i, (k,v)) in enumerate(info):
                    text="{}: {}".format(k,v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            #show output frame
            cv2.imshow("Frame", frame)
            key=cv2.waitKey(1) & 0xFF
            
            #if the 's' key is selected, a bounding box can be drawn
            if key == ord("s"):
                #draw bounding box and press ENTER or SPACE after selecting the region of interest (ROI) or press ESCAPE to reselect
                BBtrack=cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                
                #start object tracker on the supplied bounding box, start FPS throughput estimator
                tracker.init(frame, BBtrack)
                fps=FPS().start()
            
            #if 'q' key is pressed, break
            elif key==ord("q"):
                break
            
        if self.video == None:
            vs.stop()#release webcam pointer
        else:
            vs.release()#release file pointer
            
        cv2.destroyAllWindows()
    
    if __name__ == "__main__":
        self.video = "/home/robot/abhi/Object-Tracker/cv2_test.mp4"
        self.tracker = "csrt"
        start()
        
    
    
    
            
        
        
    
