import math
import cv2
import numpy as np

import pickle
from mapping import mapping_function, make_dict

from sample_with_better import play_haptic


#video_path = 'data/traffic/traffic.mp4'
video_path = '../../ONNX-RAFT-Optical-Flow-Estimation/doc/data/ball/ball.mp4'

cap = cv2.VideoCapture(video_path)
w = cap.get(3)
h = cap.get(4)


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (200, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()


fast = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
p0 = fast.detect(old_frame, None)
p0, p0_des = brief.compute(old_frame, p0)
p0 = np.float32([ m.pt for m in p0 ]).reshape(-1,1,2)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    vestIdx = make_dict()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        x = a - b
        y = b - d

        z = int(math.sqrt((x * x) + (y * y)))
        vestIdx = mapping_function(z, c, d, w, h, vestIdx)


        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        np.set_printoptions(linewidth=np.inf)

    vestIdx = np.trunc(vestIdx)
    vestIdx = np.where(vestIdx > 100, 100, vestIdx)
    indexlist = np.where(np.isin(vestIdx, 0) == False)[0]
    print(vestIdx)

    play_haptic(indexlist, vestIdx)

    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_frame = frame.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv2.destroyAllWindows()
