#!/usr/bin/env python3
#multi - object tracker to improve thermal camera
#author: abhinandan.vellanki@gmail.com

#import the necessary packages

import time
import cv2
#import FPS
import sys

class Track():
    def __init__(self, tracker_type):
        self.tracker_type=tracker_type
    
    def create(self,tracker_type):
        OPENCV_TRACKERS={ #name to function mapper, does not include GOTURN
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        tracker=OPENCV_TRACKERS[tracker_type]() #call constructor at runtime
        return tracker

    def track(self, old_bbs, old_frame, new_frame):
        num_trackers = len(old_bbs)

        if old_frame is None or new_frame is None:
            print("Tracker did not get two frames")
            return None
        
        if num_trackers == 0:
            print("No Bounding Box given")
            return None

        try:
            t1=time.time()
            trackers= cv2.MultiTracker_create() #intialize multi-object tracker
            for i in range(num_trackers):
                tracker = self.create(self.tracker_type)
                trackers.add(tracker, old_frame, tuple(old_bbs[i]))
            t2=time.time() - t1
            print("Init time: ", t2)
        except Exception as e:
            print("Caught: ", e)
            return None

        t1 = time.time()
        (success, boxes) = trackers.update(new_frame)
        t2 = time.time() - t1
    	print("Track time: ", t2)

        if success:
            return boxes
        else:
            print("Tracker Failed!!")
            return None

        #cv2.destroyAllWindows

if __name__ == "__main__":

    #the following block is for testing purposes without a screen

    tracker_type=input("Enter type of tracker to use")
    target_video = "face_test.mp4"
    tracker=Track(tracker_type=tracker_type) #initialise multi-tracker object

    frames=[] #list to store video frames
    latest_boxes = [] #stores coordinates of latest bounding boxes
    vs=cv2.VideoCapture(target_video)
    W=0 #initial frame width
    H=0 #initial frame height
    track_times=[]
    while vs.isOpened(): #while videostream is open
        ret,new_frame = vs.read() #read next frame <- draw latest ROIs 
        
        if new_frame is None:
            print("Reached end of video, stopping tracker...")
            print("Average time per track: ", float(sum(track_times)/len(track_times)))
            break
        
        if ret: #if successfully able to read next frame
            (H,W) = new_frame.shape[:2] #to set size of saved video
            new_frame = cv2.resize(new_frame,(600,400)) #resize all frames to Dell Inspiron 15 screen size for accurate input
            
            if  len(latest_boxes) == 0: #nothing being tracked
                boxes = input("Enter coordinates of initial bounding boxes  as \"((topleftX1, topleftY1, width1, height1),(topleftX2, topleftY2, width2, height2)...)\" :")
                
                if len(boxes) == 0: 
                    print("No bounding box coordinates entered")
                    sys.exit(0)

                for box in boxes: #draw initial ROIs    
                    (x,y,w,h) = [int(v) for v in box] 
                    rect = cv2.rectangle(new_frame, (x,y), (x+w, y+h), (0,255,0), 2) 

                latest_boxes = boxes #storing bb coordinates
                frames.append(new_frame) #adding first frame to list
                print("Created ROI, started tracking...")
                continue

            old_frame = frames[-1] #fetching previous frame
            old_boxes = tuple(latest_boxes) #fetching old bb coordinates
            before_track = time.time()
            new_boxes = tracker.track(old_bbs=old_boxes, new_frame=new_frame, old_frame=old_frame) #calling multi-tracker
            after_track = time.time()
            track_times.append(after_track-before_track)

            for nbox in new_boxes: #draw updated ROIs    
                    (x,y,w,h) = [int(v) for v in nbox] 
                    rect = cv2.rectangle(new_frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            frames.append(new_frame) #adding new frame to list
            latest_boxes = new_boxes #setting updated bb coordinates
            
        else:
            print("!!UNABLE TO READ STREAM!!")
            sys.exit(0)
    vs.release()

    #combine frames and save video
    saved_videoname=target_video[:-4]+"_tracked_"+tracker_type+".avi"
    print("Saving video as: ",saved_videoname," ...")
    out = cv2.VideoWriter(saved_videoname,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (600,400))
    for i in range(len(frames)): #iterate through frames array, write frames to video
        out.write(frames[i])
    out.release()
    print("Video saved successfully!")

    #end cv2 processing 
    cv2.destroyAllWindows

        
