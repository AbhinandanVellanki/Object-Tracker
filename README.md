# Object-Tracker

# Installation
 
 Required: OpenCV 4.1 or above, installed with Opencv_contrib dependency

# Usage

from tracker import track

tracker = track(tracker_type)

tracker_type =  "csrt": Discriminative Correlation Filter
                "kcf": Kernelized Correlation Filters
                "boosting": BOOSTING Tracker
                "mil": MIL Tracker,
                "tld": TLD Tracker,
                "medianflow": MedianFlow Tracker
                "mosse": MOSSE Tracker

new bounding box coordinates = tracker.track(<old bounding box coordinates>, <old frame>, <new frame>)


# Testing

unit test defined in main, specify target video and type of tracker to use. 


