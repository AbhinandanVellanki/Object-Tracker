#object tracker to improve thermal camera
#author: abhinandan.vellanki@gmail.com

#import the necessary packages
import time
import cv2
#import FPS
import sys

class Track():
    def __init__(self, tracker_type):
        self.tracker_type=tracker_type
        OPENCV_TRACKERS={ #name to function mapper, does not include GOTURN
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.tracker=OPENCV_TRACKERS[self.tracker_type]() #call constructor at runtime

    def track_next(self, old_bb, new_frame, old_frame):

        if old_frame is None or new_frame is None:
            print("No frame given")
            return None
        
        if old_bb is None:
            print("No Bounding Box given")
            return None
        try:
            self.tracker.init(old_frame,old_bb)
        except Exception as e:
            print("Caught : ",e)
            return None

        (success,box)=self.tracker.update(new_frame)

        if success:
            (x,y,w,h) = [int(v) for v in box]
            return (x,y,w,h)
        else:
            print("Error in Tracking")
            return None
 
        #cv2.destroyAllWindows

if __name__ == "__main__":

    #the following block is for testing purposes without a screen
    tracker_type="csrt"
    target_video = "face_test.mp4"
    tracker=Track(tracker_type=tracker_type)
    frames=[] #list to store video frames
    latest_box=None #stores latest bounding box coordinates
    vs=cv2.VideoCapture(target_video)
    W=0 #initial frame width
    H=0 #initial frame height
    while vs.isOpened():
        ret,new_frame = vs.read() #read next frame <- to draw latest ROI 
        if new_frame is None:
            print("Reached end of video, stopping tracker...")
            break
        if ret: #if successfully able to read next frame
            (H,W) = new_frame.shape[:2] #to set size of saved video
            new_frame = cv2.resize(new_frame,(1535,863)) #resize all frames to Dell Inspiron 15 screen size for accurate input
            
            if not latest_box: #nothing being tracked
                (x,y,w,h) = input("Enter initial bounding box coordinates as \"(topleftX, topleftY, width, height)\" :")
                rect = cv2.rectangle(new_frame, (x,y), (x+w, y+h), (0,255,0), 2) #draw initial ROI
                latest_box = (x,y,w,h) #storing bb coordinates
                frames.append(new_frame) #adding first frame to list
                print("Created ROI, starting tracking...")
                continue

            old_frame=frames[-1] #fetching previous frame
            (x,y,w,h) = latest_box #fetching bb coordinates
            (a,b,c,d) = tracker.track_next(old_bb = (x,y,w,h), new_frame=new_frame, old_frame=old_frame) #calling tracker
            rect2 = cv2.rectangle(new_frame, (a,b), (a+c, b+d), (0,255,0)) #drawing updated ROI on new frame
            frames.append(new_frame) #adding new frame to list
            latest_box = (a,b,c,d) #setting updated bb coordinates
        else:
            print("UNABLE TO READ STREAM!!")
            sys.exit(0)
    vs.release()

    #combine frames and save video
    saved_videoname=target_video[:-4]+"_tracked_"+tracker_type+".avi"
    print("Saving video as: ",saved_videoname," ...")
    out = cv2.VideoWriter(saved_videoname,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1535,863))
    for i in range(len(frames)): #iterate through frames array, write frames to video
        out.write(frames[i])
    out.release()
    print("Video saved successfully")

    #end cv2 processing 
    cv2.destroyAllWindows

        
