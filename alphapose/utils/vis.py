# -----------------------------------------------------
# Modified by Enrique Arroyo, February 2021
# -----------------------------------------------------

import math
import time
from json import loads

import cv2
import numpy as np
import torch

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color


def vis_frame_dense(frame, im_res, add_bbox=False, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 19), (19, 21), (21, 17), (17, 22), (22, 20), (20, 6), # Shoulder
        (6, 28), (28, 30), (30, 8), (8, 32), (32, 34), (34, 10),  # right arm
        (5, 27), (27, 29), (29, 7), (7, 31), (31, 33), (33, 9),  #left arm
        (17, 47), (47, 48), (48, 18), #spine
        (20, 24), (24, 26), (26, 12), #right spine
        (19, 23), (23, 25), (25, 11), #left spine
        (12, 36), (36, 38), (38, 18), (18, 37), (37, 35), (35, 11),#row
        (12, 40), (40, 42), (42, 14), (14, 44), (44, 46), (46, 16),#right leg
        (11, 39), (39, 41), (41, 13), (13, 43), (43, 45), (45, 15)#left leg
    ]
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
               (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # unkonw
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # unknown
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255), #unknown
               (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # unkonw
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # unknown
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)] #unknown

    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (77, 255, 222), (77, 255, 222), (77, 255, 222), (77, 196, 255), (77, 196, 255), (77, 196, 255),  # Shoulder
                  (77, 135, 255), (77, 135, 255), (77, 135, 255), (191, 255, 77), (191, 255, 77), (191, 255, 77), 
                  (77, 255, 77), (77, 255, 77), (77, 255, 77), (77, 222, 255), (77, 222, 255), (77, 222, 255),
                  (255, 156, 127), (255, 156, 127), (255, 156, 127),
                  (255, 156, 127), (255, 156, 127), (255, 156, 127),
                  (255, 156, 127), (255, 156, 127), (255, 156, 127),
                  (255, 156, 127), (255, 156, 127), (255, 156, 127), (255, 156, 127), (255, 156, 127), (255, 156, 127),
                  (0, 127, 255), (0, 127, 255), (0, 127, 255), (255, 127, 77), (255, 127, 77), (255, 127, 77), 
                  (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 77, 36), (255, 77, 36),  (255, 77, 36)]


    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        # Draw bboxes
        if add_bbox:
            from PoseFlow.poseflow_infer import get_box
            keypoints = []
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            bbox = get_box(keypoints, height, width)
            # color = get_color_fast(int(abs(human['idx'][0])))
            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), BLUE, 2)
            # Draw indexes of humans
            if 'idx' in human.keys():
                cv2.putText(img, ''.join(str(e) for e in human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.35:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], (kp_scores[start_p] + kp_scores[end_p]) + 1)
    return img


def vis_frame_fast(frame, im_res, people, add_bbox=False, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [ (0, 2), (1, 3), (2, 4), (3, 5)]

        p_color = [(204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle

        line_color = [(0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        NotImplementedError

    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]

    num_humans = 0

    if(people > len(im_res['result'])):
            people = len(im_res['result'])

    initialX = int(width / (people*6.66))
    initialY = int(height / (height*0.0083))
    endingY = int(height / (height*0.015))

    human_order = orderHumans(im_res)

    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        # Draw bboxes
        if add_bbox:
            from PoseFlow.poseflow_infer import get_box
            keypoints = []
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            bbox = get_box(keypoints, height, width)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), BLUE, 2)
            # Draw indexes of humans
            if 'idx' in human.keys():
                cv2.putText(img, ''.join(str(e) for e in human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.35:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], int(2 * (kp_scores[start_p] + kp_scores[end_p]) + 1))

        # PRINTING LEG ANGLES IN THE IMAGE
        if num_humans < people:

            index = checkHuman(kp_preds, human_order)
            if index < 0 or index > (people - 1):
                continue
            
            if (kp_preds[0, 0] == human_order[index]) or (kp_preds[1, 0] == human_order[index]):

                initialX = initialX + (int(width / (people)) * index)
                endingX = initialX + 165


                cv2.rectangle(img, (initialX, initialY), (endingX, endingY), (255, 255, 255), -1)
                cv2.putText(img, str(index+1), (initialX-20, initialY-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


                pointX = int((int(kp_preds[0, 0]) + int(kp_preds[1, 0])) / 2)
                pointY = int((int(kp_preds[0, 1]) + int(kp_preds[1, 1])) / 2)

                cv2.putText(img, str(index+1), (pointX, pointY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                # Get angle of left leg
                if kp_scores[0] > 0.35 and kp_scores[2] > 0.35 and kp_scores[4] > 0.35: 
                    p04 = segment_length(int(kp_preds[0, 0]), int(kp_preds[0, 1]), int(kp_preds[4, 0]), int(kp_preds[4, 1]))
                    p02 = segment_length(int(kp_preds[0, 0]), int(kp_preds[0, 1]), int(kp_preds[2, 0]), int(kp_preds[2, 1]))
                    p24 = segment_length(int(kp_preds[2, 0]), int(kp_preds[2, 1]), int(kp_preds[4, 0]), int(kp_preds[4, 1]))

                    angle_left = math.degrees(math.acos((math.pow(p02, 2) + math.pow(p24, 2) - math.pow(p04, 2)) / (2 * p02 * p24)))
                    angle_left = int(angle_left)

                    cv2.putText(img, 'Left leg: ' + str(angle_left), (initialX, initialY-5), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 77, 255), 2)

                # Get angle of right leg
                if kp_scores[1] > 0.35 and kp_scores[3] > 0.35 and kp_scores[5] > 0.35: 
                    p15 = segment_length(int(kp_preds[1, 0]), int(kp_preds[1, 1]), int(kp_preds[5, 0]), int(kp_preds[5, 1]))
                    p13 = segment_length(int(kp_preds[1, 0]), int(kp_preds[1, 1]), int(kp_preds[3, 0]), int(kp_preds[3, 1]))
                    p35 = segment_length(int(kp_preds[3, 0]), int(kp_preds[3, 1]), int(kp_preds[5, 0]), int(kp_preds[5, 1]))

                    angle_right = math.degrees(math.acos((math.pow(p13, 2) + math.pow(p35, 2) - math.pow(p15, 2)) / (2 * p13 * p35)))
                    angle_right = int(angle_right)

                    cv2.putText(img, 'Right leg: ' + str(angle_right), (initialX, initialY - 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 127, 77), 2)

                num_humans = num_humans + 1
                initialX = int(width / (people*6.66))


    return img


def leer_json(frame, json_file, people, add_bbox=False, format='coco'):
    f = open(json_file, "r")
    content = f.read()
    jsondecoded = loads(content)

    if format == 'coco':
        #l_pair = [ (2, 8), (5, 11), (8, 14), (11, 17)]
        l_pair = [ (0, 2), (1, 3), (2, 4), (3, 5)]

        p_color = [(204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle

        line_color = [(0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError


    img = frame.copy()
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width / 2), int(height / 2)))
    num_humans = 0


    if(people > len(jsondecoded)):
            people = len(jsondecoded)

    initialX = int(width / 2 / (people*6.66))
    initialY = int(height / 2 / (height*0.0083))
    endingY = int(height / 2 / (height*0.015))

    human_order = orderHumansJson(jsondecoded)

    for human in jsondecoded:
        part_line = {}
        kp_res = human['keypoints']
        

        # Draw bboxes
        if add_bbox:
            from PoseFlow.poseflow_infer import get_box
            keypoints = []
            for n in range(0, len(kp_res), 3):
                keypoints.append(float(kp_res[n]))
                keypoints.append(float(kp_res[n+1]))
                keypoints.append(float(kp_res[n+2]))
            bbox = get_box(keypoints, height, width)
            bg = img.copy()
            # color = get_color_fast(int(abs(human['idx'][0][0])))
            cv2.rectangle(bg, (int(bbox[0]/2), int(bbox[2]/2)), (int(bbox[1]/2),int(bbox[3]/2)), BLUE, 1)
            transparency = 0.8
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
            # Draw indexes of humans
            if 'idx' in human.keys():
                bg = img.copy()
                cv2.putText(bg, ''.join(str(e) for e in human['idx']), (int(bbox[0] / 2), int((bbox[2] + 26) / 2)), DEFAULT_FONT, 0.5, BLACK, 1)
                transparency = 0.8
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
        # Draw keypoints
        for n in range(0, len(kp_res), 3):
            if kp_res[n+2] <= 0.35:
                continue
            cor_x, cor_y = int(kp_res[n]), int(kp_res[n+1])
            part_line[int(n/3)] = (int(cor_x / 2), int(cor_y / 2))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x / 2), int(cor_y / 2)), 2, p_color[int(n/3)], -1)

            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_res[n]))
            img = cv2.addWeighted(bg, float(transparency), img, float(1 - transparency), 0)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_res[start_p*3+2] + kp_res[end_p*3+2]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                # cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                transparency = max(0, min(1, 0.5 * (kp_res[start_p*3+2] + kp_res[end_p*3+2])))
                img = cv2.addWeighted(bg, float(transparency), img, float(1 - transparency), 0)

        
        # PRINTING LEG ANGLES IN THE IMAGE
        if num_humans < people:

            index = checkHumanJson(kp_res, human_order)
            if index < 0 or index > (people - 1):
                continue
            
            if (kp_res[0] == human_order[index]) or (kp_res[3] == human_order[index]):

                initialX = initialX + (int(width / (people * 2)) * index)
                endingX = initialX + 85


                cv2.rectangle(img, (initialX, initialY), (endingX, endingY), (255, 255, 255), -1)
                cv2.putText(img, str(index+1), (initialX-15, initialY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                pointX = int((int(kp_res[0]/2) + int(kp_res[3]/2)) / 2)
                pointY = int((int(kp_res[1]/2) + int(kp_res[4]/2)) / 2)

                cv2.putText(img, str(index+1), (pointX, pointY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Get angle of left leg
                if kp_res[2] > 0.35 and kp_res[8] > 0.35 and kp_res[14] > 0.35: 
                    p04 = segment_length(int(kp_res[0]), int(kp_res[1]), int(kp_res[12]), int(kp_res[13]))
                    p02 = segment_length(int(kp_res[0]), int(kp_res[1]), int(kp_res[6]), int(kp_res[7]))
                    p24 = segment_length(int(kp_res[6]), int(kp_res[7]), int(kp_res[12]), int(kp_res[13]))

                    angle_left = math.degrees(math.acos((math.pow(p02, 2) + math.pow(p24, 2) - math.pow(p04, 2)) / (2 * p02 * p24)))
                    angle_left = int(angle_left)

                    cv2.putText(img, 'Left leg: ' + str(angle_left), (initialX, initialY), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 77, 255), 1)

                # Get angle of right leg
                if kp_res[5] > 0.35 and kp_res[11] > 0.35 and kp_res[17] > 0.35: 
                    p15 = segment_length(int(kp_res[3]), int(kp_res[4]), int(kp_res[15]), int(kp_res[16]))
                    p13 = segment_length(int(kp_res[3]), int(kp_res[4]), int(kp_res[9]), int(kp_res[10]))
                    p35 = segment_length(int(kp_res[9]), int(kp_res[10]), int(kp_res[15]), int(kp_res[16]))

                    angle_right = math.degrees(math.acos((math.pow(p13, 2) + math.pow(p35, 2) - math.pow(p15, 2)) / (2 * p13 * p35)))
                    angle_right = int(angle_right)

                    cv2.putText(img, 'Right leg: ' + str(angle_right), (initialX, initialY - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 127, 77), 1)

                num_humans = num_humans + 1

                initialX = int(width / 2 / (people*6.66))
 
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img


def vis_frame(frame, im_res, people, add_bbox=False, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [ (0, 2), (1, 3), (2, 4), (3, 5)]

        p_color = [(204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle

        line_color = [(0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width / 2), int(height / 2)))
    num_humans = 0

    if(people > len(im_res['result'])):
            people = len(im_res['result'])

    initialX = int(width / 2 / (people*6.66))
    initialY = int(height / 2 / (height*0.0083))
    endingY = int(height / 2 / (height*0.015))

    human_order = orderHumans(im_res)

    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        

        # Draw bboxes
        if add_bbox:
            from PoseFlow.poseflow_infer import get_box
            keypoints = []
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            bbox = get_box(keypoints, height, width)
            bg = img.copy()
            # color = get_color_fast(int(abs(human['idx'][0][0])))
            cv2.rectangle(bg, (int(bbox[0]/2), int(bbox[2]/2)), (int(bbox[1]/2),int(bbox[3]/2)), BLUE, 1)
            transparency = 0.8
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
            # Draw indexes of humans
            if 'idx' in human.keys():
                bg = img.copy()
                cv2.putText(bg, ''.join(str(e) for e in human['idx']), (int(bbox[0] / 2), int((bbox[2] + 26) / 2)), DEFAULT_FONT, 0.5, BLACK, 1)
                transparency = 0.8
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.35:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x / 2), int(cor_y / 2))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x / 2), int(cor_y / 2)), 2, p_color[n], -1)

            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_scores[n]))
            img = cv2.addWeighted(bg, float(transparency), img, float(1 - transparency), 0)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                # cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                transparency = max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])))
                img = cv2.addWeighted(bg, float(transparency), img, float(1 - transparency), 0)

        
        # PRINTING LEG ANGLES IN THE IMAGE
        if num_humans < people:

            index = checkHuman(kp_preds, human_order)
            if index < 0 or index > (people - 1):
                continue
            
            if (kp_preds[0, 0] == human_order[index]) or (kp_preds[1, 0] == human_order[index]):

                initialX = initialX + (int(width / (people * 2)) * index)
                endingX = initialX + 85


                cv2.rectangle(img, (initialX, initialY), (endingX, endingY), (255, 255, 255), -1)
                cv2.putText(img, str(index+1), (initialX-15, initialY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                pointX = int((int(kp_preds[0, 0]/2) + int(kp_preds[1, 0]/2)) / 2)
                pointY = int((int(kp_preds[0, 1]/2) + int(kp_preds[1, 1]/2)) / 2)

                cv2.putText(img, str(index+1), (pointX, pointY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Get angle of left leg
                if kp_scores[0] > 0.35 and kp_scores[2] > 0.35 and kp_scores[4] > 0.35: 
                    p04 = segment_length(int(kp_preds[0, 0]), int(kp_preds[0, 1]), int(kp_preds[4, 0]), int(kp_preds[4, 1]))
                    p02 = segment_length(int(kp_preds[0, 0]), int(kp_preds[0, 1]), int(kp_preds[2, 0]), int(kp_preds[2, 1]))
                    p24 = segment_length(int(kp_preds[2, 0]), int(kp_preds[2, 1]), int(kp_preds[4, 0]), int(kp_preds[4, 1]))

                    angle_left = math.degrees(math.acos((math.pow(p02, 2) + math.pow(p24, 2) - math.pow(p04, 2)) / (2 * p02 * p24)))
                    angle_left = int(angle_left)

                    cv2.putText(img, 'Left leg: ' + str(angle_left), (initialX, initialY), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 77, 255), 1)

                # Get angle of right leg
                if kp_scores[1] > 0.35 and kp_scores[3] > 0.35 and kp_scores[5] > 0.35: 
                    p15 = segment_length(int(kp_preds[1, 0]), int(kp_preds[1, 1]), int(kp_preds[5, 0]), int(kp_preds[5, 1]))
                    p13 = segment_length(int(kp_preds[1, 0]), int(kp_preds[1, 1]), int(kp_preds[3, 0]), int(kp_preds[3, 1]))
                    p35 = segment_length(int(kp_preds[3, 0]), int(kp_preds[3, 1]), int(kp_preds[5, 0]), int(kp_preds[5, 1]))

                    angle_right = math.degrees(math.acos((math.pow(p13, 2) + math.pow(p35, 2) - math.pow(p15, 2)) / (2 * p13 * p35)))
                    angle_right = int(angle_right)

                    #cv2.putText(image, text, org, font, fontScale, color, thickness)
                    cv2.putText(img, 'Right leg: ' + str(angle_right), (initialX, initialY - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 127, 77), 1)

                num_humans = num_humans + 1
                initialX = int(width / 2 / (people*6.66))
 
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img

# length of the segment from P1 to P2
def segment_length(p1_x, p1_y, p2_x, p2_y):
    length = math.sqrt(math.pow((p1_x - p2_x), 2) + math.pow((p1_y - p2_y), 2))
    return length

def orderHumans(im_res):
    order_human = []
    for human in im_res['result']:
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']

        if (kp_scores[0] > 0.35 and kp_scores[2] > 0.35 and kp_scores[4] > 0.35) or (kp_scores[1] > 0.35 and kp_scores[3] > 0.35 and kp_scores[5] > 0.35):
            izq = float('inf')
            der = float('inf')
            if (kp_scores[0] > 0.35 and kp_scores[2] > 0.35 and kp_scores[4] > 0.35):
                izq = kp_preds[0, 0]
            if (kp_scores[1] > 0.35 and kp_scores[3] > 0.35 and kp_scores[5] > 0.35):
                der = kp_preds[1, 0]
            if izq <= der:
                order_human.append(izq)
            else:
                order_human.append(der)
            order_human.sort()

    return order_human

def orderHumansJson(jsondecoded):
    order_human = []
    for human in jsondecoded:
        kp_res = human['keypoints']

        if (kp_res[2] > 0.35 and kp_res[8] > 0.35 and kp_res[14] > 0.35) or (kp_res[5] > 0.35 and kp_res[11] > 0.35 and kp_res[17] > 0.35):
            izq = float('inf')
            der = float('inf')
            if (kp_res[2] > 0.35 and kp_res[8] > 0.35 and kp_res[14] > 0.35):
                izq = kp_res[0]
            if (kp_res[5] > 0.35 and kp_res[11] > 0.35 and kp_res[17] > 0.35):
                der = kp_res[3]
            if izq <= der:
                order_human.append(izq)
            else:
                order_human.append(der)
            order_human.sort()

    return order_human

def checkHuman(kp_preds, human_order):
    for x in range(len(human_order)):
        if (human_order[x] == kp_preds[0, 0]) or (human_order[x] == kp_preds[1, 0]):
            return x
    return -1

def checkHumanJson(kp_res, human_order):
    for x in range(len(human_order)):
        if (human_order[x] == kp_res[0]) or (human_order[x] == kp_res[3]):
            return x
    return -1

def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval
