from gooey import Gooey, GooeyParser
import cv2
import os
import logging
import sys
import argparse
import numpy as np
from datetime import datetime
import imutils
from imutils.video import FPS
import pandas as pd

# Commandline Argument 1: the video file
video = None
# Commandline Argument 2: the path
output_file_location = None
# Commandline Argument 3: the datafile type
output_file_type = None

framerate = None

# Uable file types for writing data
FILE_TYPES = [".csv", ".xlsx", ".json", ".xml"]

def init_dir():
  """
  Make folder structure if it does not exist, 
    - default output folder
    - default model folder
  """
  if not os.path.exists("\\model") or not os.path.exists("\\Model"):
    os.system("mkdir model")
  elif not os.path.exists("\\output"):
    os.system("mkdir output")

def init_logger():
  """
  Generate logger and suppress the pillow library warnings from polluting log files
  """
  # pillow library pollutes log, changing it to now show it's debug messages
  pil_logger = logging.getLogger('PIL')
  pil_logger.setLevel(logging.INFO)
  
  # Create and configure logger with current date time
  log_name = datetime.now().strftime('output_%b-%d-%H-%M')
  logging.basicConfig(filename = log_name + ".log", level = logging.DEBUG)
  logger = logging.getLogger()
  
  return logger

@Gooey(program_name="Object Detection", 
        required_cols=1, 
        default_size=(600,650),
        progress_regex=r"^progress: (\d+)%$",
        hide_progress_msg=True,
        timing_options = {
          'show_time_remaining':True,
          'hide_time_remaining_on_complete': False
        }
      )
def get_gooey_args():
  '''
    Use GooeyParser to build up the arguments we will use in our script
    Save the arguments in a default json file so that we can retrieve them
    every time we run the script.
  '''
  # If there are no Commandline arguments, the Gooey GUI loads
  if len(sys.argv) == 1:
  
    parser = GooeyParser(description="Archipelago's very own Video Processing Software")
    # To find the video.
    parser.add_argument('Video', 
                        help="Browse to select video for processing", 
                        widget="FileChooser"
                      )
    # Output directory for the resulting dataset.
    parser.add_argument('Directory', 
                        help="Select folder for saving outputted data", 
                        widget="DirChooser")
    # The output type for the dataset when created.
    parser.add_argument('Type', 
                        help="Select file type for outputted Data", 
                        widget="Dropdown", choices=FILE_TYPES, default=".csv")
    # Determin if we display the video while processing
    # parser.add_argument('--show_video', 
    #                     help="Tick box to have video shown while processing", 
    #                     widget="BlockCheckbox")
    return parser.parse_args()
  
def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('Video', type=str)
  parser.add_argument('Directory', type=str)
  parser.add_argument('Type', type=str)
  # parser.add_argument('--show_video', type=str)
  return parser.parse_args()

# Logging and printing the same text
def output_log(text:str, level:str="info"):
  """
  Sends text for output to console and to a specified level logger
   - level="info" (Default)
   - level="exception"
  """
  if level == "info":
    logger.info(text)
  if level == "exception":
    logger.exception(text)
  else:
    logger.info(text)
  print(text)

class FisheriesData:
  def __init__(self):
    self.species = "species"
    self.timeStampStart = "time_stamp_start"
    self.timeStampEnd = "time_stamp_end"
    self.direction = "direction"
    self.dataFrame = []
    self.speciesList = []
    self.timeStampStartList = []
    self.timeStampEndList = []
    self.directionList = []
  
  def makeDF(self):
    self.dataFrame = pd.DataFrame({self.species:self.speciesList,self.timeStampStart:self.timeStampStartList,self.timeStampEnd:self.timeStampEndList,self.direction:self.directionList})
  
  def addData(self, species: str, timeStampStart: str, timeStampEnd: str, direction: str):
    self.speciesList.append(species)
    self.timeStampStartList.append(timeStampStart)
    self.timeStampEndList.append(timeStampEnd)
    self.directionList.append(direction)

  def addDictData(self, dictionary: dict):
    for k, v in dictionary.items():
      for kk, vv in v.items():
        if type(kk) is tuple:
          x, y = kk
          self.timeStampStartList.append(x)
          self.timeStampEndList.append(y)
          self.speciesList.append(vv) 
        else:
          self.directionList.append(vv)

  def writeCSV(self, fileName: str):
    """
    Write dataframe of stored data to .csv
    """
    file_name = fileName + '.csv'
    self.dataFrame.to_csv(file_name, sep=',', encoding='utf-8', index=False)

  def writeExcel(self, fileName: str):
    """
    Write dataframe of stored data to .xlsx
    """
    file_name = fileName + '.xlsx'
    self.dataFrame.to_excel(file_name, sheet_name='sheet1', index=False)

  def writeXML(self, fileName: str):
    """
    Write dataframe of stored data to .xml
    """
    file_name = fileName + '.xml'
    self.dataFrame.to_xml(file_name)

  def writeJSON(self, fileName: str):
    """
    Write dataframe of stored data to .json
    """
    file_name = fileName + '.json'
    self.dataFrame.to_json(file_name, orient='records', indent=2)
  
  def export(self, output_file_type: str, export_path: str):
    """
    Compile data added to the fisheries_data class instance to passed in output file type
    """
    self.makeDF()
    if output_file_type == '.csv':
      self.writeCSV(export_path)
    elif output_file_type == '.json':
      self.writeJSON(export_path)
    elif output_file_type == '.xlsx':
      self.writeExcel(export_path)
    elif output_file_type == '.xml':
      self.writeXML(export_path)
    else:
      output_log("[ERROR] FisheriesData.export(str,str): No valid export file type given use [{}]".format(", ".join(FILE_TYPES)),
                  level='exception')

class data_point:
  """
  Data point is a data class simpely to store the centroid and related data
  """
  def __init__(self, x:int, y:int, label:str, frame_num:int, direction:str):
    self.x = int(x)
    self.y = int(y)
    self.label = label
    self.frame_num = frame_num
    self.direction = direction


def get_time_from_frame(frame_num: int, framerate: int) -> str:
  """
  Converts the frame number parameter to a timestamp string from the input video.
  """
  hours, minutes, seconds = (0,0,0)
  
  seconds = int(frame_num / framerate)
  if seconds / 60 > 0:
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    if minutes / 60 > 0:
      hours = int(minutes / 60)
      minutes = int(minutes % 60)
  
  if minutes == 0 and hours == 0:
    return "{:2.0f} second(s)".format(seconds)
  elif hours == 0:
    return "{:2.0f}:{:2.0f} minutes(s)".format(minutes, seconds)
  else:
    return "{:2.0f}:{:2.0f}:{:2.0f}  hour(s) ".format(hours, minutes, seconds)

def validate_arguments(video: str, output_file_location: str, output_file_type: str) -> bool:
  """
  Evaluate passed in arguments to validate filepaths
  """
  eval = True
  # test arguments for valid file paths and None types
  if os.path.isfile(video) and os.path.exists(video):
    output_log("Processing file: {}".format(video))
  else:
    logger.exception("invalid file/directory")
    eval = False
  
  if os.path.isdir(output_file_location) and os.path.exists(output_file_location):
    output_log("Outputting data file and log file to: {}".format(output_file_location))
  else:
    logger.exception("invalid file/directory")
    eval = False
  
  if output_file_type in FILE_TYPES:
    output_log("Selected output data file type is: {}".format(output_file_type))
  else:
    logger.exception("invalid file type or none selected")
    eval = False
  
  return eval

def run(args):
  """
  Main execution funtion for the program.
  """
  (clip, framecount, framerate, duration) = (0,0,0,0)
  
  # If the only Commandline argument is the .py file, the Gooey GUI loads as normal, otherwise get args from args parser
  
  video = args.Video
  output_file_location = args.Directory
  output_file_type = args.Type
  
  validate_arguments(video,
                     output_file_location,
                     output_file_type)

  clip = cv2.VideoCapture(video)
  framecount = clip.get(cv2.CAP_PROP_FRAME_COUNT)
  framerate = clip.get(cv2.CAP_PROP_FPS)
  duration = float(framecount) / float(framerate)
  

  output_log("Frame rate:{:.2f}".format(framerate))
  output_log("Video Duration:{:.0f} seconds".format(duration))
  
  # Detection code starts here!
  OUTPUT_FILE=output_file_location + '\output.mp4'
  LABELS_FILE='model/obj.names'
  CONFIG_FILE='model/yolov4-obj2.cfg'
  WEIGHTS_FILE='model/yolov4-obj2_best.weights'
  CONFIDENCE_THRESHOLD=0.5

  H=None
  W=None

  # capture input video
  video_capture = cv2.VideoCapture(video)

  # get input video's frame size
  frame_width = int(video_capture.get(3))
  frame_height = int(video_capture.get(4))
  frame_size = (frame_width,frame_height)

  # get input video's fps
  input_fps = video_capture.get(cv2.CAP_PROP_FPS)

  fps = FPS().start()

  fourcc = cv2.VideoWriter_fourcc(*"mp4v") # for mp4
  writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, input_fps, frame_size, True)

  # make Labels with labels_file
  LABELS = open(LABELS_FILE).read().strip().split("\n")

  # set random color for labels and bounding boxes
  np.random.seed(4)
  COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

  # load the YOLO network model with config and weights file
  net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

  # determine only the *output* layer names that we need from YOLO
  ln = net.getLayerNames()
  ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
  cnt = 0

  # points array of centroid points
  points = []

  # temp var for direction text so it persists between frames
  points_size = 0
  frames_since_last_detection = 0

  # initialize FisheriesData to use for aggregating export data
  fishery_export = FisheriesData()

  # iterate through video frames
  while True:
    cnt+=1
    ok, image = video_capture.read()
    if ok == False:
      break

    #image transforms
    image = imutils.resize(image, width=500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
      (H, W) = image.shape[:2]

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    if cnt % (input_fps / 2) == 0:

      # transform image into a blob
      blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), crop=False)
      net.setInput(blob)
      # Get detections
      layerOutputs = net.forward(ln)

      # loop over each of the layer outputs
      for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
          # extract the class ID and confidence (i.e., probability) of
          # the current object detection
          scores = detection[5:]
          classID = np.argmax(scores)
          confidence = scores[classID]

          # filter out weak predictions by ensuring the detected
          # probability is greater than the minimum probability
          if confidence > CONFIDENCE_THRESHOLD:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
      # apply non-maxima suppression to suppress weak, overlapping bounding
      # boxes
      idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)

      # ensure at least one detection exists
      if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
          # extract the bounding box coordinates
          (x, y) = (boxes[i][0], boxes[i][1])
          (w, h) = (boxes[i][2], boxes[i][3])
          
          # if label is halibut, add a new centroid to array of points
          if LABELS[classIDs[i]] == 'Halibut':
            points.append(data_point(x + (w/2),y + (h/2), "halibut", cnt, "None"))
          
          color = [int(c) for c in COLORS[classIDs[i]]]
          output_log('\tDetected, {} at {}'.format(LABELS[classIDs[i]], get_time_from_frame(cnt, framerate)))

          cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
          text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
          cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)
    
    if cnt  % (input_fps * 2) == 0 and len(points) >= 2:

      # the difference between the y-coordinate of the *current*
      # centroid and the mean of *previous* centroids will tell
      # us in which direction the object is moving (negative for
      # 'up' and positive for 'down')
      y = points[-2].y
      direction = points[-1].y - np.mean(y)

      # if the direction is negative (indicating the object
      # is moving up)
      if direction < 0:
        points[-1].direction = "Onboard"
      # if the direction is positive (indicating the object
      # is moving down)
      elif direction > 0:
        points[-1].direction = "Offboard"
    
    # instantiate last pos as tuple
    last_pos = (0,0)
    
    # iterate over the stored points to draw points and lines between them
    for point in points:
      x = point.x
      y = point.y
      cv2.circle(image, (x, y), 10, (255, 255, 255), -1)
      if last_pos != (0,0):
        cv2.line(image, (x, y), last_pos, (255, 255, 255), 2)
        cv2.putText(image, point.direction, (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      last_pos = (x, y)
    # Clean up points array if there are no new points and it has been 5 seconds
    if points != []:
      if len(points) == points_size and frames_since_last_detection > input_fps * 5:
        # Send centroid data from points to fisheries data class to store for export
        last_point = points[-1]
        first_point = points[0]
        if last_point.direction != "None":
          fishery_export.addData(last_point.label, first_point.frame_num, 
            last_point.frame_num, last_point.direction)
        # empty the list
        points.clear()
        frames_since_last_detection = 0
      else:
        # Update points_size with the length of points if it has not been 5 seconds
        points_size = len(points)
        frames_since_last_detection += 1

    # print message needed to animate the progress bar
    print("progress: {}%".format(int((cnt/framecount)*100)))
    sys.stdout.flush()
    
    # if args.show_video:
    # show the output image
    cv2.imshow("output", image)
    # Convert image colors back to bgr from rgb for presentation
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    writer.write(cv2.resize(image,frame_size))
    fps.update()
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break

  fps.stop()

  output_log("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
  output_log("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

  # Export FisheriesData
  data_output_location = output_file_location + "\\" + video.split('\\')[-1].split('.')[0]
  fishery_export.export(output_file_type ,data_output_location)

  # do a bit of cleanup
  cv2.destroyAllWindows()

  # release the file pointers
  writer.release()
  video_capture.release()

#-------------Main execution-------------

logger = init_logger()

if __name__ == '__main__':
  init_dir()
  if len(sys.argv) == 1:
    args = get_gooey_args()
  else:
    args = get_args()
    run(args)