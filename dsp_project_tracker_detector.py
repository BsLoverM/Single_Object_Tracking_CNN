

import pandas as pd
import numpy as np
import time
import cv2

video_choice_index = 0 #Please change this value between 0 to 9 for running code on given dataset of 10 videos

#List of videos in our dataset
dataset_video_list = [
    'Dancer2',
    'Gym',
    'person-1',
    'person-10',
    'person-13',
    'person-16',
    'Human9',
    'Skater',
    'Human2',
    'Skater2'
]

data_dir = "dataset/" + dataset_video_list[video_choice_index] + '/'

#Function to calculate IoU for 2 set of bounding boxes
def iou_calc(box1, box2):
    
    #Calculating respective bounding box areas
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    #Calculating top-left and bottom-right coordinates for intersection of bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    area_of_intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1) #Calculating the area of intersection
    
    return area_of_intersection / float(box1_area + box2_area - area_of_intersection) #IoU computation (Area of Intersection / Area of Union)

#File structure: [x, y, width, height]
gt_boxes = pd.read_csv(data_dir + 'groundtruth.txt', sep = ',', header = None, names = ['x', 'y', 'w', 'h'])   #Reading groud truth box coordinates of the object to be tracked and saving in a dataframe
df = gt_boxes

print("Loading Caffe model")
net = cv2.dnn.readNetFromCaffe('caffe_j/MobileNetSSD_deploy.prototxt', 'caffe/MobileNetSSD_deploy.caffemodel') #Loading graph and weights of caffe model which is pre-trained as a pedestrian detection CNN

cap = cv2.VideoCapture(data_dir + dataset_video_list[video_choice_index] + '.mp4') #Read video file using cv2.VideoCapture

i = 0 #Initial frame count

tracking_length = 30 #Tracking length parameter: defines after how many frames the tracker will reset and capture base-frame from the detector

#Initializations to store results
iou_total_detector = 0
iou_detector_list = []
misses_total_detector = 0
average_time_detector = 0

iou_total_tracker = 0
iou_tracker_list = []
misses_total_tracker = 0
average_time_tracker = 0

#While loop until the video has frames left to be read
while cap.isOpened():

    ret, frame = cap.read() #Reading frame as an image
    
    if ret == True and i != len(df):

        #Plotting Ground Truth
        xmin_gt = df['x'][i] #xmin of ground truth rectangle (top-left corner)
        ymin_gt = df['y'][i] #ymin of ground truth rectangle (top-left corner)
        
        xmax_gt = df['x'][i] + df['w'][i] #xmax of ground truth rectangle (bottom-right corner)
        ymax_gt = df['y'][i] + df['h'][i] #ymax of ground truth rectangle (bottom-right corner)
        
        gt_bb = [xmin_gt, ymin_gt, xmax_gt, ymax_gt]
        
        cv2.rectangle(frame, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]), (125, 255, 51), 1) #Drawing ground truth rectangle on the frame
        cv2.putText(frame, 'gt' , (gt_bb[0], gt_bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 51), 1) #Drawing text 'gt' on ground truth rectangle
        
        #Detector
        start_time_detector = time.time() #Starting timer for Detector computation 
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5) #Generating blobs from Caffe model
            
        net.setInput(blob)
        detections = net.forward() #Extracting detections in the form of dictionary from Caffe model
        
        output_classes = detections[0, 0, :, 1]
        
        person_indexes = []
        
        #Computing indexes of boxes where person was found
        for j in range(len(output_classes)):
            if output_classes[j] == 15:
                person_indexes.append(j)
        
        confidence = detections[0, 0, person_indexes, 2]
        
        iou_temp_detector = 0
        if len(confidence) != 0:
            
            confidence = np.max(confidence) #Keeping box with maximum confidence and discarding others
            
            if confidence > 0.3: #Detection threshold
                
                idx = np.where(detections[0, 0, :, 2] == confidence)
            
                idx = idx[0][0]
                
                box = detections[0, 0, idx, 3:7] * np.array([w, h, w, h])
                
                (xmin_detector, ymin_detector, xmax_detector, ymax_detector) = box.astype('int') #Extracting box coordinates for detector's output
                
                detector_bb = [xmin_detector, ymin_detector, xmax_detector, ymax_detector]
                
                cv2.rectangle(frame, (detector_bb[0], detector_bb[1]), (detector_bb[2], detector_bb[3]), (0, 0, 255), 1) #Drawing detector rectangle on the frame
                cv2.putText(frame, 'd' , (detector_bb[0], detector_bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) #Drawing text 'd' on detector rectangle
                
                iou_temp_detector = iou_calc(gt_bb, detector_bb) #Calculating IoU between detector and ground truth
        
        if iou_temp_detector == 0:
            misses_total_detector += 1
        
        end_time_detector = time.time()
        average_time_detector += (end_time_detector - start_time_detector) #Calculating total computation time for detection
        print("IoU (Detector) for Frame " + str(i+1) + ": " + str(iou_temp_detector))
        iou_total_detector += iou_temp_detector
        iou_detector_list.append(iou_temp_detector)        
        
        start_time_tracker = time.time() #Starting timer for Tracking computation

        if i % tracking_length == 0 and confidence != 0: #Initializing RoI for tracking after every tracking_length frames
            print("Resetting Camshift Tracker")
            
            tracker_window = (xmin_detector, ymin_detector, xmax_detector - xmin_detector, ymax_detector - ymin_detector) #Initializing the rectangular window to be tracked which is equal to the detector rectangle
            
            roi = [xmin_detector, ymin_detector, xmax_detector, ymax_detector] #Saving the rectangle as RoI
        
            roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]] #Extracting RoI from the actual full frame
            
            hsv_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV) #Converting BGR to HSV
            
            mask = cv2.inRange(hsv_roi_frame, np.array((0., 62., 32.)), np.array((180., 255., 255.))) #Generating a mask from HSV converted frame
            
            roi_hist = cv2.calcHist([hsv_roi_frame], [0], mask, [180], [0, 180]) #Calculating histogram of roi in HSV
            
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX) #Normalizing the histogram
            
        #Camshift Tracking
        _, frame2 = cv2.threshold(frame, 180, 155, cv2.THRESH_TOZERO_INV) #Thresholding
        
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV) #Converting BGR to HSV
        
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1) #Calculating back projection to see how many pixels fit the histogram
        
        ret2, track_window = cv2.CamShift(dst, tracker_window, (cv2.TERM_CRITERIA_COUNT, tracking_length, 1)) #Applying Camshift tracking
        
        pts = cv2.boxPoints(ret2) #Generating points for updated RoI
        
        pts = np.int0(pts)
        
        #cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        
        #Converting points in space to rectangle coordinates
        xmin_tracker = pts[list(pts[:, 0] * pts[:, 1]).index(np.min(pts[:, 0] * pts[:, 1]))][0]
        ymin_tracker = pts[list(pts[:, 0] * pts[:, 1]).index(np.min(pts[:, 0] * pts[:, 1]))][1]
        
        xmax_tracker = pts[list(pts[:, 0] * pts[:, 1]).index(np.max(pts[:, 0] * pts[:, 1]))][0]
        ymax_tracker = pts[list(pts[:, 0] * pts[:, 1]).index(np.max(pts[:, 0] * pts[:, 1]))][1]
        
        tracker_bb = [xmin_tracker, ymin_tracker, xmax_tracker, ymax_tracker]
        
        cv2.rectangle(frame, (tracker_bb[0], tracker_bb[1]), (tracker_bb[2], tracker_bb[3]), (255, 0, 0), 1) #Drawing tracker rectangle on the frame
        cv2.putText(frame, 'cs' , (tracker_bb[0], tracker_bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) #Drawing text 'cs' on detector rectangle
        
        iou_temp_tracker = iou_calc(gt_bb, tracker_bb) #Calculating IoU between detector and ground truth
        if iou_temp_tracker == 0:
            misses_total_tracker += 1
        
        end_time_tracker = time.time()
        average_time_tracker += (end_time_tracker - start_time_tracker) #Calculating total computation time for tracking
        print("IoU (Tracker) for Frame " + str(i+1) + ": " + str(iou_temp_tracker))
        print("")
        iou_total_tracker += iou_temp_tracker
        iou_tracker_list.append(iou_temp_tracker)
        
        cv2.imshow('output', frame)
        
        i = i + 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #time.sleep(0.01)
        
    else:
        break
    
cap.release()
cv2.destroyAllWindows()

print("")
print("Results for video titled " + dataset_video_list[video_choice_index] + '.mp4')
print("")
print("Average IoU (Detector): " + str(iou_total_detector/(i+1)))
print("Standard Deviation in IoU (Detector):" + str(np.std(np.array(iou_detector_list))))
print("Total Misses (Detector): " + str(misses_total_detector))
print("Average computation time per frame (Detector): " + "{:.2f}".format(1000 * average_time_detector/(i+1)) + " ms")

print("")
print("Average IoU (Tracker): " + str(iou_total_tracker/(i+1)))
print("Standard Deviation in IoU (Tracker):" + str(np.std(np.array(iou_tracker_list))))
print("Total Misses (Tracker): " + str(misses_total_tracker))
print("Average computation time per frame (Tracker): " + "{:.2f}".format(1000 * average_time_tracker/(i+1)) + " ms")