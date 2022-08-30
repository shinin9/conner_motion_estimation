import math
import cv2
import numpy as np

import pickle
from mapping import mapping_function, make_dict

from sample_with_better import play_haptic
from ui_vibration import ui_vib

#video_path = 'data/traffic.mp4'
#video_path = 'data/WaterDrop_Trim_Trim.mp4'
video_path = 'data/Dance.mp4'
# video_path = 'data/Fall.mp4'
#video_path = 'data/Baseball.mp4'
#video_path = 'data/Cheetah.mp4'
#video_path = 'data/ball.mp4'
#video_path = 'data/new_b_Trim.mp4'
# video_path = 'data/Eagle.mp4'
video_name = video_path[5:-4]
mag_arr_list = []
cap = cv2.VideoCapture(video_path)
w = cap.get(3)
h = cap.get(4)
print(w, h)
fps = cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(f'data_output/{video_name}.avi', fourcc, fps, (int(w)*2, int(h)))

zarr = []

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (1000, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()



size = 21
# fgbg = cv2.createBackgroundSubtractorMOG2()
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# old_fgmask = fgbg.apply(old_frame)
# old_fgmask = cv2.morphologyEx(old_fgmask, cv2.MORPH_OPEN, kernel)
#old_frame = cv2.medianBlur(old_frame,size)
#old_frame = cv2.fastNlMeansDenoisingColored(old_frame,None,10,10,7,21)
#old_frame = cv2.bilateralFilter(old_frame,27,75,75)
# cv2.imwrite('fgmask.jpg', old_fgmask)

fast = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
p0 = fast.detect(old_frame, None)
p0, p0_des = brief.compute(old_frame, p0)
p0 = np.float32([m.pt for m in p0]).reshape(-1, 1, 2)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    #frame = cv2.medianBlur(frame,size)
    #frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
    #frame = cv2.bilateralFilter(frame, 27, 75, 75)
    # fgmask = fgbg.apply(frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)


    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    # draw the tracks
    vestIdx = make_dict()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        x = a - b
        y = b - d

        z = int(math.sqrt((x * x) + (y * y)))
        zarr.append(z)
        vestIdx = mapping_function(z, c, d, w, h, vestIdx)

        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        np.set_printoptions(linewidth=np.inf)

    vestIdx = np.trunc(vestIdx)
    vestIdx = np.where(vestIdx > 100, 100, vestIdx)
    indexlist = np.where(np.isin(vestIdx, 0) == False)[0]



    play_haptic(indexlist, vestIdx)
    ui = ui_vib(vestIdx)

    img = cv2.add(frame, mask)
    ui = cv2.resize(ui,(int(w), int(h)))
    concat = np.concatenate((img, ui), axis=1)
    out.write(concat)
    cv2.imshow('output', concat)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_frame = frame.copy()
    p0 = good_new.reshape(-1, 1, 2)


zarr.sort()
print(zarr)
print(len(zarr))
print(zarr[-1])
print(zarr[int(len(zarr)*80/100)], int(len(zarr)*80/100))

cap.release()
cv2.destroyAllWindows()
# with open(f'mag/corner_{video_name}', 'wb') as fw:
#     pickle.dump(mag_arr_list, fw)