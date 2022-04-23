import sys
import os.path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from openvino.inference_engine import IENetwork, IECore
import matplotlib.pyplot as plt
from time import time
import argparse

# ============================================================================================================================================
# region INTIAL PARAMS

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
fontSize = 1.25

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

# Load names of classes
MODEL_PATH = 'models/'
classesFile = MODEL_PATH + "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# endregion INTIAL PARAMS



# ============================================================================================================================================
# region DETECTION

# FOR YOLOV5 --------------------------------------------------------------------------------------
def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv.copyMakeBorder(img, top2, bottom2, left2, right2, cv.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv.copyMakeBorder(img, top2, bottom2, left2, right2, cv.BORDER_CONSTANT, value=color)  # add border
    return img/255


def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv.FILLED)
    # Display text inside the rectangle.
    cv.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv.LINE_AA)


def post_process_yolov5(input_image, outputs, class_name=None):

    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    
    outputs = [outputs]

    # Rows.
    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    box = None
    for i in indices:
        if class_name is not None:
            if classes[class_ids[i]] == class_name:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                cv.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
                label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
                draw_label(input_image, label, left, top)
                return input_image, box
        else:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            draw_label(input_image, label, left, top)
    
    boxes = np.array(boxes)
    ibox = list(boxes[indices])
    return input_image, ibox

    
# -------------------------------------------------------------------------------------------------

# endregion DETECTION


# ============================================================================================================================================
# region TRACKING
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE']
tracker_type = tracker_types[3]

if tracker_type == 'BOOSTING':
    tracker = cv.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv.legacy.TrackerMedianFlow_create()
elif tracker_type == 'GOTURN':
    tracker = cv.legacy.TrackerGOTURN_create()
elif tracker_type == "CSRT":
    tracker = cv.legacy.TrackerCSRT_create()
elif tracker_type == "MOSSE":
    tracker = cv.legacy.TrackerMOSSE_create()
else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
        print(t)
    
# endregion TRACKING


# ============================================================================================================================================
# region PROCESS
def detect_and_track(exec_net, input_blob, dims, fileName, saveName, class_name):
        # ===============================  INITIALIZE VIDEO  ===================================
        vcap = cv.VideoCapture(fileName)

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
     
        if int(major_ver)  < 3 :
            fps = vcap.get(cv.cv.CV_CAP_PROP_FPS)
            # print("Frames per second using video.get(cv.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = vcap.get(cv.CAP_PROP_FPS)
            # print("Frames per second using video.get(cv.CAP_PROP_FPS) : {0}".format(fps))


        frame_width = int(vcap.get(3))
        frame_height = int(vcap.get(4))

        half_h = int(frame_height/2)
        half_w = int(frame_width/2)

        # frame_width = int(vcap.get(cv.CAP_PROP_FRAME_WIDTH))
        # frame_height =int(vcap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object.
        # outavi = cv.VideoWriter(saveName, cv.VideoWriter_fourcc('M','J','P','G'), round(fps), (frame_width,frame_height))
        outmp4 = cv.VideoWriter(saveName, cv.VideoWriter_fourcc(*'XVID'), round(fps), (frame_width, frame_height))

        tmpFrame = np.zeros((frame_height, frame_width, 3)).astype(np.uint8)

        count = 0
        success = True
        success, imcv = vcap.read() # BGR color

        init_tracker = True
        update = False
        
        # Start timer
        timer = cv.getTickCount()

        fnum = 0
        
        # -------------------------------------- YOLO 5------------------------------------------
        n, c, h, w = dims
        wait_key_code = 1
        is_async_mode = True
        number_input_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))
        number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames
        if number_input_frames != 1:
            # ret, frame = vcap.read()
            pass
        else:
            is_async_mode = False
            wait_key_code = 0
        cur_request_id = 0
        next_request_id = 1
        # --------------------------------------------------------------------------------------
            

        while success:

            # ===============================  READ VIDEO FRAME IN ITERATION ===================================
            fnum += 1
            print('Frame =', fnum , '   ', end='')
            # if fnum == 130:
            #     print('Stand here')
            
            # imcv = cv.imread('data/sample.jpg')

            imcv = cv.cvtColor(imcv, cv.COLOR_BGR2RGB)
            # improc = processImage(imcv, self.edit_params, self.colors)
            # improc = cv.cvtColor(improc, cv.COLOR_RGB2BGR)




            # ===============================  DETECTION ==========================================
                    
            # -------------------------------------- YOLO 5------------------------------------------
            if is_async_mode:
                request_id = next_request_id
                in_frame = letterbox(imcv.copy(), (w, h))
            else:
                request_id = cur_request_id
                in_frame = letterbox(imcv.copy(), (w, h))
            # in_frame = in_frame[:, :, [2, 1, 0]]
            # in_frame = in_frame[:, :, ::-1] # CONVERT BACK TO BGR
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            
            start_time = time()
            exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame}) # CHANGE cur_request_id to re_quest_id
            det_time = time() - start_time
            if exec_net.requests[request_id].wait(-1) == 0: # CHANGE cur_request_id to re_quest_id
                output = exec_net.requests[request_id].output_blobs # CHANGE cur_request_id to re_quest_id
                for layer_name, out_blob in output.items():
                    imcv1, ibox = post_process_yolov5(imcv.copy(), out_blob.buffer, class_name)
                parsing_time = time() - start_time
                          
            if ibox is not None:
                # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                # t, _ = net.getPerfProfile()
                # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
                label = "Frame = %d, Inference time = %.2f ms" % (fnum, parsing_time)
                cv.putText(imcv1, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), 2)
            else:
                # label = 'No detection'
                label = "Frame = %d, NO DETECTION" % fnum
                print('NO DETECTION!', end='')
                cv.putText(imcv1, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 2)
                init_tracker = True
                # continue

            # plt.imshow(imcv1)
            # plt.show()
            # --------------------------------------------------------------------------------------   



           # ===============================  TRACKING ===========================================

            imcv1_rz = cv.resize(imcv1, (half_w, half_h))
            tmpFrame[0:half_h, 0:half_w, :] = imcv1_rz
            
            imcv2 = imcv.copy()

            if init_tracker and (ibox is not None):
                # Initialize tracker with first frame and bounding box
                print("Detected bounding box : {}".format(ibox), '   ', end='')
                ok = tracker.init(imcv2, tuple(ibox))
                bbox = ibox
                init_tracker = False
                update = False
                # out = cv.VideoWriter('{}_{}_{}.mp4'.format(video_name,tracker_type,bbox),cv.VideoWriter_fourcc(*'MP4V'), 30, (640,360))
            else:
                if bbox is not None:
                    # Update tracker
                    ok, bbox = tracker.update(imcv2)
                    update = True
                    # Calculate processing time and display results.
                    # Calculate Frames per second (FPS)
                    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
                else:
                    continue
                

            # Draw bounding box
            if ok:
            # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv.rectangle(imcv2, p1, p2, (255,0,0), 2, 1)
                print('\n', end='')
            else:
                # Tracking failure
                cv.putText(imcv2, "TRACKING FAILURE", (20,80), cv.FONT_HERSHEY_SIMPLEX, fontSize, (255,0,0), 2)
                print('TRACKING FAILURE!') # auto add new line
                if (update) and (ibox is not None):
                    ok = tracker.init(imcv2, tuple(ibox))
                    bbox = ibox
                # else:
                #     success, imcv = vcap.read()
                #     continue

            # Display tracker type on frame
            label = "Tracker, Frame = %d" % fnum
            cv.putText(imcv2, tracker_type + label, (20,40), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0,0,255), 2)
            
            # Display FPS on frame
            # cv.putText(imcv2, "FPS : " + str(int(fps)), (20,80), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0,0,255), 2)

            # plt.imshow(imcv2)
            # plt.show()

            imcv2_rz = cv.resize(imcv2, (half_w, half_h))
            tmpFrame[half_h:, half_w:, :] = imcv2_rz

            # plt.imshow(imcv1_rz)
            # plt.show()

            # plt.imshow(tmpFrame)
            # plt.show()

            if fnum == 97:
                print('Stand here')



            # ===============================  WRITE VIDEO ===================================
            tmpFrame = cv.cvtColor(tmpFrame, cv.COLOR_BGR2RGB)
            # outavi.write(tmpFrame)
            outmp4.write(tmpFrame)

            success, imcv = vcap.read()

        

        # Release everything if job is finished
        vcap.release()
        # outavi.release()
        outmp4.release()

        # Stop the program if reached end of video
        if not success:
            print("Done processing !!!")
            print("Output file is stored as ", saveName)
            # cv.waitKey(3000)

# endregion PROCESS


# ============================================================================================================================================
# MAIN

def get_argparser():
    parser = argparse.ArgumentParser(add_help=False)
    # args = parser.add_argument_group('Params')
    parser.add_argument('--model', action='store', type=str, default='models/yolov5s_openvino_model/yolov5s.xml', nargs=1)
    # parser.add_argument('--input', action='store', type=str, default='data/sample.jpg', nargs=1)
    parser.add_argument('--input', action='store', type=str, default='data/Project  Detection + Tracking_crop.mp4', nargs=1)
    parser.add_argument('--output', action='store', type=str, default='output/Project  Detection + Tracking_tracked_VINO.mp4', nargs=1)
    parser.add_argument('--class_name', action='store', type=str, default='sports ball', nargs=1)
    return parser

def main():
    args = get_argparser().parse_args()
    
    ie = IECore()
    net_2 = ie.read_network(model=args.model)
    net_2.batch_size = 1
    input_blob = next(iter(net_2.input_info))
    dims = net_2.input_info[input_blob].input_data.shape
    device = "CPU"
    exec_net = ie.load_network(network=net_2, num_requests=2, device_name=device)

    detect_and_track(exec_net, input_blob, dims, args.input, args.output, args.class_name)
    
    
if __name__ == '__main__':
    sys.exit(main() or 0)