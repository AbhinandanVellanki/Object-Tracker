#object tracker to improve thermal camera
#author: abhinandan.vellanki@gmail.com

#import the necessary packages
import time
import cv2
import numpy as np

class track():
    
    def __init__(self, video, tracker):        
        self.video = video
        self.tracker_name = tracker
        OPENCV_TRACKERS={ #mapper, does not include GOTURN
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.tracker=OPENCV_TRACKERS[self.tracker_name]() #call constructor at runtime

        #initialize bounding box coordinates to track
        self.BBtrack = None
        self.frames=[]
        #initialize fps throughput estimator
        #self.fps=None
    
    def start(self):
        #start video stream
        self.vs=cv2.VideoCapture(self.video)
                
        #iterate over video frames
        while self.vs.isOpened():
            ret,frame=self.vs.read()
            if ret :

                if frame is None: #reached end of stream
                    break

                #frame = cv2.resize(frame,(500,500))
                (H,W) = frame.shape[:2]
                
                #if an object is being tracked
                if self.BBtrack is not None:
                    (success, box) = self.tracker.update(frame) #get new BB coordinates of selected box
                    
                    if success:
                        print("update success")
                        (x,y,w,h) = [int(v) for v in box]
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) #draw green rectangle around new coordinates of tracking box
                        
                    #update fps counter
                    #self.fps.update()
                    #self.fps.stop()

                    #initialize info to be diplayed on frame
                    info=[
                        ("Tracker", self.tracker),
                        ("Success", "Yes" if success else "No")
                        #("FPS", "{:.2f}".format(fps.fps()))                    
                    ]
                    
                    #loop over info and add to frame
                    for (i, (k,v)) in enumerate(info):
                        text="{}: {}".format(k,v)
                        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                #uncomment the following block to test without screen and with input coordinates
                
                if self.BBtrack is None:
                    (x,y,w,h) = input("Enter initial bounding box coordinates as \"[topleftX, topleftY, width, height]")
                    self.BBtrack = (x,y,w,h)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) #draw initial ROI
                    print(self.BBtrack)
                    self.tracker.init(frame,self.BBtrack)
                    #self.fps=FPS().start()
                    print("Starting ROI drawn, tracker initialized")


                #add output frame to frames list
                self.frames.append(frame) #add frame to list of frames
                
                #end of test without screen block

                #uncomment the following block to test with screen
                """
                #show output frame
                cv2.imshow("Frame", frame)
                key=cv2.waitKey(1) & 0xFF
                
                #if the 's' key is selected, a bounding box can be drawn
                if key == ord("s"):
                    #draw bounding box and press ENTER or SPACE after selecting the region of interest (ROI) or press ESCAPE to reselect
                    self.BBtrack=cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                    
                    #start object tracker on the supplied bounding box, start FPS throughput estimator
                    self.tracker.init(frame, self.BBtrack)
                    #self.fps=FPS().start()
                
                #if 'q' key is pressed, break
                elif key==ord("q"):
                    break
                """
                #end of test with screen block
            else:
                print("ERROR!! Stream could not be opened!!")
                break
            
        self.vs.release() #release file pointer
        #combine frames and save video
        saved_videoname=self.video+"_tracked"
        video=cv2.VideoWriter(saved_videoname, 0, 1, (500,500))

        for f in self.frames:
            video.write(f)
        video.release() #release video pointer

        #end cv2 processing
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    track_type=str(input("Enter the type of tracker to use :"))
    target_video = "car_short_test.mp4"
    obj_tracker=track(target_video, track_type)
    obj_tracker.start()
