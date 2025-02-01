import numpy as np
import cv2
import sys # for command line arguments
import os  # for file validation

# This program uses median for background estimation and subtraction.
# Calculates sparse optical flow of objects by Lucas-Kanade method for motion estimation.
# Uses Frame differencing for object separation, 
# Contour detection and analysis for object counting and classification.
def main() :
  # Message to display if invalid arguments are passed
  message = """\n Please provide correct arguments as below
     \n -b \"video_file_name.avi\" : for background modelling of video"
     \n -s \"video_file_name.avi\" : for segmentation of video"""
  n = len(sys.argv) # Number of command line arguments passed
  if n != 3 : # Check total number of arguments except sys.argv[0] as it refers to the same file
    print(message) # print message
    exit() # exit the program
  if sys.argv[1] == "-b" : # if argument 1 is correct for background modelling
    background_modelling(sys.argv[2]) # call function for backgroud modelling
  elif sys.argv[1] == "-s" : # if argument 1 is correct for segmentation
    segmentation(sys.argv[2]) # call function for segmentation
  else :
     print(message) # print message
     exit() # exit program

# Displays multiple frames in single window.
def concat_frames(f1,f2,f3,f4): # all frames are passed as an argument
   row1 = np.hstack((f1, f2)) # arrange first 2 frames/arrays horizontally
   row2 = np.hstack((f3, f4)) # arrange next 2 frames/arrays horizontally
   return np.vstack((row1,row2)) # arrange both the above horizontal frames/arrays vertically 

# Extraction of estimated background frame
def backgroud_frame(path) : # takes in video path to read random frames
  cap = cv2.VideoCapture(path) # video object 'cap' is created
  # Randomly select 25 frames/frameIds from all the video frames
  frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
  # Store selected frames in an array
  frames = []
  for fid in frameIds: # Iterate through all 25 random frames
      cap.set(cv2.CAP_PROP_POS_FRAMES, fid) # set the frame in video object
      ret, frame = cap.read() # read the frame
      frames.append(frame) # append the frame in frames array
  # Calculate the median on frames array along the time axis
  medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
  # Reset frame number in video object to 0
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  return medianFrame # return median frame for processing

# Background Subtraction
def pre_processing(frame, medianFrame) :
  # Convert background frame to grayscale frame
  grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
  # Convert current frame to grayscale
  gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Calculate absolute difference of current frame and 
  # the median frame
  diff = cv2.absdiff(gframe, grayMedianFrame)
  # Apply binary threshold on the obtained frame
  th, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
  return diff

# Noise removal and countour extraction
def getcontours(frame) :
  # Noise Removal
  frame = cv2.medianBlur(frame, 7)
  # Calculate threshold after noise removal
  _, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_OTSU)
  # Find contours for the frame
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

# Object counting and classification
def object_count(frame, contour, fcount) :
  # Variables for object count
  car = 0
  person = 0
  others = 0
  all_objs = 0
  # Iterate over all contours and fiter invalid ones
  # based on contour areas
  for(i,c) in enumerate(contour):
   (x,y,w,h) = cv2.boundingRect(c)
   area = cv2.contourArea(c)
   valid_contour = area > float(400)
   if not valid_contour:
         continue
   # For valid contours, classify objects 
   # person and cars based on width and height ratio
   # if width/height ratio is less than 1
   # the rectangle is vertical signifying a person
   valid_person = (w/float(h) <= 0.8)
   # if width/height ratio is greater than 1
   # the rectangle is horizontal signifying a car
   valid_car = (w/float(h) >= 1.1)
   if valid_person :
     person += 1
   elif valid_car :
     car += 1
   else :
     others += 1
   all_objs = person + car + others
  # Print object count to console
  print("Frame "+str(fcount)+": "+str(all_objs)+" objects ("+str(person)+" person, "+str(car)+" cars and "+str(others)+" others)")
  
  return frame
  

def background_modelling(path) :
  message = """\n Invalid video file.
    \n Place the video file in current folder or provide full video file path")
    \n along with extension such as .avi or .mp4"""
  # Check if file is available or not
  if not os.path.isfile(path) :
    print(message)
    exit()
  # Path to video file to create object
  cap = cv2.VideoCapture(path)
  # Medianframe object
  bgFrame = backgroud_frame(path)
  # Setting window to Full screen
  cv2.namedWindow("Background Modelling", cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty("Background Modelling",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
  # Frame count
  fcount = 0
  # Loop over all frames
  ret = True
  while(ret):
    # Read frame
    ret, frame = cap.read()
    if ret == 0:
      break
    # Incrementing frame count
    fcount += 1
	# Original frame
    org_frame = frame.copy()
	# Frame after backgroud subtraction
    bin_frame = pre_processing(frame.copy(), bgFrame)
	# Contours retrieval
    contours = getcontours(bin_frame)
    # Counting, Classifying and printing objects
    newframe = object_count(frame.copy(), contours, fcount)
	# Object detection in original color by applying binary mask on frame
    detected_objects = cv2.bitwise_and(frame, frame, mask=bin_frame)
    # Convert current frame to BGR for same channel
    bin_frame = cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)
	# Increase intensity of frame
    detected_objects = cv2.add(detected_objects,detected_objects)
	# Concatenate all 4 frames
    all_frames = concat_frames(org_frame, bgFrame, bin_frame, detected_objects)
    # Display the frame on window
	cv2.imshow("Background Modelling", all_frames)
    # Wait for 1 milliseconds for any key input, if input is 'q' break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  # Release video object
  cap.release()
  # Destroy all windows
  cv2.destroyAllWindows()

def segmentation(path) :
  message = """\n Invalid video file.
    \n Place the video file in current folder or provide full video file path")
    \n along with extension such as .avi or .mp4"""
  # Check if file is available or not
  if not os.path.isfile(path) :
    print(message)
    exit()
  # Medianframe object
  bgFrame = backgroud_frame(path)
  # Parameters for Shi-Tomasi corner detection
  feature_params = dict(maxCorners = 400, qualityLevel = 0.03, minDistance = 4, blockSize = 7)
  # Parameters for Lucas-Kanade optical flow
  lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  # The video feed is read in as a VideoCapture object
  cap = cv2.VideoCapture(path)
  # Variable for color to draw optical flow track
  color = (255, 255, 0)
  # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
  ret, first_frame = cap.read()
  # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
  prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
  prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
  # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
  mask = np.zeros_like(first_frame)
  
  # Setting window to Full screen
  cv2.namedWindow("Segmentation", cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty("Segmentation",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
  # Frame count
  fcount = 0
  while(True):
      # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
      ret, frame = cap.read()
      if ret == 0 :
        break
	  # Incrementing frame count
      fcount += 1
      curr_frame = frame.copy()
      seg_frame = segemented_objects(frame.copy(), bgFrame, fcount)
      # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Calculates sparse optical flow by Lucas-Kanade method
      next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
      # Selects good feature points for previous position
      good_old = prev[status == 1]
      # Selects good feature points for next position
      good_new = next[status == 1]
  	  # Draws the optical flow tracks
      for i, (new, old) in enumerate(zip(good_new, good_old)):
          # Returns a contiguous flattened array as (x, y) coordinates for new point
          a, b = new.ravel()
          # Returns a contiguous flattened array as (x, y) coordinates for old point
          c, d = old.ravel()
          # Draws line between new and old position with green color and 2 thickness
          mask = cv2.arrowedLine(mask,(int(c), int(d)),(int(a), int(b)), color, 2,tipLength = 0.3)
          # Draws filled circle (thickness of -1) at new position with green color and radius of 3
          frame = cv2.circle(frame, (int(a), int(b)), 3, color, -1)
      # Overlays the optical flow tracks on the original frame
      output = cv2.add(frame, mask)
      # Copy current gray frame to previous frame for next iteration
      prev_gray = gray.copy()	
      prev = good_new.reshape(-1, 1, 2)
      # Concatenate all frames
	  all_frames = concat_frames(curr_frame,first_frame,output,seg_frame)
      # Display the frame on window
      cv2.imshow("Segmentation", all_frames)
      first_frame = curr_frame.copy()
      # Wait for 1 milliseconds for any key input, if input is 'q' break the loop
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  # Release video object
  cap.release()
  # Destroy all windows
  cv2.destroyAllWindows()

def segemented_objects(frame, bgFrame, fcount) :
  # Frame after backgroud subtraction
  bin_frame = pre_processing(frame.copy(), bgFrame)
  # Segmented object frame
  seg_objects = cv2.bitwise_and(frame, frame, mask=bin_frame)
  # Increasing intensity of frame
  seg_objects = cv2.add(seg_objects,seg_objects)
  # Contours retrieval
  contours = getcontours(bin_frame)
  # Counting, Classifying and printing objects
  object_count(frame.copy(), contours, fcount)
  return seg_objects

if __name__ == '__main__':
    main()