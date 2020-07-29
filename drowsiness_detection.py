# Import relevant packages
import cv2
import dlib
import time
import imutils
import argparse
import playsound
import numpy as np
from threading import Thread
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist


# contruct the argument parser
arg = argparse.ArgumentParser()
arg.add_argument("-p", "--shape_predictor", required=True, help="path to shape predictor")
arg.add_argument("-a", "--alarm", type=str, default="", help="path to alarm .WAV/.MP3 file")
args = vars(arg.parse_args())


# load the frontal face detector and shape predictor
print("[INFO] Loading the shape predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

	
# grab the indexes of facial landmarks of both left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	
# define few IMPORTANT constants here
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
FRAME_COUNTER = 0
ALARM_ON = False


def sound_alarm(path):
	playsound.playsound(path)
	

# method to compute eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
	# vertical distance
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	
	# horizontal distance
	C = dist.euclidean(eye[0], eye[3])
	
	ear = (A + B) / (2.0 * C)
	return ear


# Start the webcam feed...
print("[INFO] Starting the webcam and warming up the sensors...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
	# get the webcam stream, resize and and convert to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# detect faces in feed using detector
	rects = detector(gray, 0)
	
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		# extract the left and right eye (x,y)-coordinates and compute their EAR
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		
		#combined EAR of both eyes
		ear = (leftEAR + rightEAR) / 2.0
		
		# compute the convex hull for the left and right eye, then visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		# Finally we are ready to check if person is showing symptoms of drowsiness.
		if ear < EYE_AR_THRESH:
			FRAME_COUNTER += 1
		
			if FRAME_COUNTER >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True
					if args["alarm"] != "":
						t = Thread(target=sound_alarm, args=(args["alarm"],))
						t.deamon = True
						t.start()
				
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		else:
			FRAME_COUNTER = 0
			ALARM_ON = False
			
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
	# show the frame
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('x'):
		break

vs.stop()
cv2.destroyAllWindows()
