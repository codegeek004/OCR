import os
from pathlib import Path
import sys
from datetime import datetime
import datetime
import time
import threading
from threading import Thread
import cv2
import numpy
import pytesseract
#from PIL import Image
import Linguist
os.environ['TESSDATA_PREFIX'] = '/usr/local/Cellar/tesseract/5.3.4_1/share/tessdata'

def tesseract_location(tesseract_cmd_path):
   try:
        tesseract_cmd_path = '/usr/local/bin/tesseract'
   except FileNotFoundError:
        print("Please double check the Tesseract file directory or ensure it's installed.")
        sys.exit(1)


class RateCounter:
    def __init__(self):
        self.start_time = None
        self.iterations = 0

    def start(self):
       self.start_time = time.perf_counter()
       return self

    def increment(self):
       self.iterations += 1

    def rate(self):
       elapsed_time = (time.perf_counter() - self.start_time)
       return self.iterations / elapsed_time


class VideoStream:
   def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

   def start(self):
       Thread(target=self.get, args=()).start()
       return self

   def get(self):
       while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

   def get_video_dimensions(self):
       width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
       height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
       return int(width), int(height)

   def stop_process(self):
       self.stopped = True


class OCR:


    # def __init__(self, exchange: VideoStream, language=None):
    def __init__(self):
        self.boxes = None
        self.stopped = False
        self.exchange = None
        self.language = None
        self.width = None
        self.height = None
        self.crop_width = None
        self.crop_height = None

    def start(self):
        """
        Creates a thread targeted at the ocr process
        :return: self
        """
        Thread(target=self.ocr, args=()).start()
        return self

    def set_exchange(self, video_stream):
        """
        Sets the self.exchange attribute with a reference to VideoStream class
        :param video_stream: VideoStream class
        """
        self.exchange = video_stream

    def set_language(self, language):
        """
        Sets the self.language parameter
        :param language: language code(s) for detecting custom languages in pytesseract
        """
        self.language = language

    def ocr(self):
        """
        Creates a process where frames are continuously grabbed from the exchange and processed by pytesseract OCR.
        Output data from pytesseract is stored in the self.boxes attribute.
        """
        while not self.stopped:
            if self.exchange is not None:  # Defends against an undefined VideoStream reference
                frame = self.exchange.frame

                # # # CUSTOM FRAME PRE-PROCESSING GOES HERE # # #
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                # # # # # # # # # # # # # # # # # # # #

                frame = frame[self.crop_height:(self.height - self.crop_height),
                              self.crop_width:(self.width - self.crop_width)]
                self.boxes = pytesseract.image_to_data(frame, lang=self.language)
        
    def set_dimensions(self, width, height, crop_width, crop_height):
        """
        Sets the dimensions attributes

        :param width: Horizontal dimension of the VideoStream frame
        :param height: Vertical dimension of the VideoSteam frame
        :param crop_width: Horizontal crop amount if OCR is to be performed on a smaller area
        :param crop_height: Vertical crop amount if OCR is to be performed on a smaller area
        """
        self.width = width
        self.height = height
        self.crop_width = crop_width
        self.crop_height = crop_height

    def stop_process(self):
        """
        Sets the self.stopped attribute to True and kills the ocr() process
        """
        self.stopped = True


def capture_image(frame, captures=0):
    """
    Capture a .jpg during CV2 video stream. Saves to a folder /images in working directory.

    :param frame: CV2 frame to save
    :param captures: (optional) Number of existing captures to append to filename

    :return: Updated number of captures. If capture param not used, returns 1 by default
    """
"""    cwd_path = os.getcwd()
    Path(cwd_path + '/images').mkdir(parents=False, exist_ok=True)

#    now = datetime.now()
    # Example: "OCR 2021-04-8 at 12:26:21-1.jpg"  ...Handles multiple captures taken in the same second
    name = "OCR " + now.strftime("%Y-%m-%d") + " at " + now.strftime("%H:%M:%S") + '-' + str(captures + 1) + '.jpg'
    path = 'images/' + name
    cv2.imwrite(path, frame)
    captures += 1
    print(name)
    return captures
"""

def views(mode: int, confidence: int):
    """
    View modes changes the style of text-boxing in OCR.

    View mode 1: Draws boxes on text with >75 confidence level

    View mode 2: Draws red boxes on low-confidence text and green on high-confidence text

    View mode 3: Color changes according to each word's confidence; brighter indicates higher confidence

    View mode 4: Draws a box around detected text regardless of confidence

    :param mode: view mode
    :param confidence: The confidence of OCR text detection

    :returns: confidence threshold and (B, G, R) color tuple for specified view mode
    """
    conf_thresh = None
    color = None

    if mode == 1:
        conf_thresh = 75  # Only shows boxes with confidence greater than 75
        color = (0, 255, 0)  # Green

    if mode == 2:
        conf_thresh = 0  # Will show every box
        if confidence >= 50:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Red

    if mode == 3:
        conf_thresh = 0  # Will show every box
        color = (int(float(confidence)) * 2.55, int(float(confidence)) * 2.55, 0)

    if mode == 4:
        conf_thresh = 0  # Will show every box
        color = (0, 0, 255)  # Red

    return conf_thresh, color


def put_ocr_boxes(boxes, frame, height, crop_width=0, crop_height=0, view_mode=1):
    """
    Draws text bounding boxes at tesseract-specified text location. Also displays compatible (ascii) detected text
    Note: ONLY works with the output from tesseract image_to_data(); image_to_boxes() uses a different output format

    :param boxes: output tuple from tesseract image_to_data() containing text location and text string
    :param numpy.ndarray frame: CV2 display frame destination
    :param height: Frame height
    :param crop_width: (Default 0) Horizontal frame crop amount if OCR was performed on a cropped frame
    :param crop_height: (Default 0) Vertical frame crop amount if OCR was performed on a cropped frame
    :param view_mode: View mode to specify style of bounding box

    :return: CV2 frame with bounding boxes, and output text string for detected text
    """

    if view_mode not in [1, 2, 3, 4]:
        raise Exception("A nonexistent view mode was selected. Only modes 1-4 are available")

    text = ''  # Initializing a string which will later be appended with the detected text
    if boxes is not None:  # Defends against empty data from tesseract image_to_data
        for i, box in enumerate(boxes.splitlines()):  # Next three lines turn data into a list

            box = box.split()
            if i != 0:
                if len(box) == 12:
                    x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
                    conf = box[10]
                    word = box[11]
                    x += crop_width  # If tesseract was performed on a cropped image we need to 'convert' to full frame
                    y += crop_height

                    conf_thresh, color = views(view_mode, int(float(conf)))

                    if int(float(conf)) > conf_thresh:
                        cv2.rectangle(frame, (x, y), (w + x, h + y), color, thickness=1)
                        text = text  + ' ' +  word
        if text.isascii():  # CV2 is only able to display ascii chars at the moment
            cv2.putText(frame, text, (5, height - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200))

    return frame, text


def put_crop_box(frame: numpy.ndarray, width: int, height: int, crop_width: int, crop_height: int):
    cv2.rectangle(frame, (crop_width, crop_height), (width - crop_width, height - crop_height),
                  (255, 0, 0), thickness=1)
    return frame


def put_rate(frame: numpy.ndarray, rate: float) -> numpy.ndarray:

    cv2.putText(frame, "{} Iterations/Second".format(int(rate)),
                (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    return frame


def put_language(frame: numpy.ndarray, language_string: str) -> numpy.ndarray:
    cv2.putText(frame, language_string,
                (10, 65), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    return frame


def ocr_stream(crop: list[int, int], source: int = 0, view_mode: int = 1, language=None):
    captures = 0  # Number of still image captures during view session

    video_stream = VideoStream(source).start()  # Starts reading the video stream in dedicated thread
    img_wi, img_hi = video_stream.get_video_dimensions()

    if crop is None:  # Setting crop area and confirming valid parameters
        cropx, cropy = (100, 100)  # Default crop if none is specified
    else:
        cropx, cropy = crop[0], crop[1]
        if cropx > img_wi or cropy > img_hi or cropx < 0 or cropy < 0:
            cropx, cropy = 0, 0
            print("Impossible crop dimensions supplied. Dimensions reverted to 0 0")

    ocr = OCR().start()  # Starts optical character recognition in dedicated thread
    print("OCR stream started")
    print("Active threads: {}".format(threading.activeCount()))
    ocr.set_exchange(video_stream)
    ocr.set_language(language)
    ocr.set_dimensions(img_wi, img_hi, cropx, cropy)  # Tells the OCR class where to perform OCR (if img is cropped)

    cps1 = RateCounter().start()
    lang_name = Linguist.language_string(language)  # Creates readable language names from tesseract langauge code
    # Main display loop
    print("\nPUSH c TO CAPTURE AN IMAGE. PUSH q TO VIEW VIDEO STREAM\n")
    while True:

        # Quit condition:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            video_stream.stop_process()
            ocr.stop_process()
            print("OCR stream stopped\n")
            print("{} image(s) captured and saved to current directory".format(captures))
            break
        

        frame = video_stream.frame  # Grabs the most recent frame read by the VideoStream class

        # # # All display frame additions go here # # # CUSTOMIZABLE
        frame = put_rate(frame, cps1.rate())
        frame = put_language(frame, lang_name)
        frame = put_crop_box(frame, img_wi, img_hi, cropx, cropy)
        frame, text = put_ocr_boxes(ocr.boxes, frame, img_hi,
                                    crop_width=cropx, crop_height=cropy, view_mode=view_mode)


        # Photo capture:
        if pressed_key == ord('c'):
            print(text)
            captures = capture_image(frame, captures)
        
        cv2.imshow("realtime OCR", frame)
        cv2.imread("/Users/yashvaishnav/College-project/ocr")

        
        cps1.increment()  # Incrementation for rate counter

"""        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_entry = f"[{timestamp}] Your text: {text}\n"

        with open("yash.txt", "a") as log_file:
            log_file.write(log_entry)
        print(log_entry)"""
