# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
# encoding=utf8
import sys

import cv2
import torchvision.transforms as transforms
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import transforms as T
from sklearn.preprocessing import normalize
import models
import sex_age
import torch
from util.FeatureExtractor import FeatureExtractor
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
age_dict = {"0": "0-15", "1": "15-30", "2": "30-45", "3": "45-60",
                 "4" : "60+"}
sex_dict = {"0": "M", "1": "F"}


def pool2d(tensor, type= 'max'):
    sz = tensor.size()
    if type == 'max':
        x = torch.nn.functional.max_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    if type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    x = x[0].cpu().data.numpy()
    x = np.transpose(x,(2,1,0))[0]
    return x


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    print("init model....")
    use_gpu = True
    exact_list = ['7']
    model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=use_gpu,
                              aligned=True)
    checkpoint = torch.load("./checkpoint_ep300.pth.tar")
    
    #global id
    global_next_id = []
    global_next_id.append(1)
    # tmp2 = {}
    # for item in checkpoint.items():
    #     key = item[0]
    #     values = item[1]
    #     newkey = key[7:]
    #     tmp2[newkey] = values
    # model.load_state_dict(tmp2)
    model.load_state_dict(checkpoint['state_dict'])

##sex and age
    transform_test = transforms.Compose([
        # transforms.Scale(224,224),
        transforms.ToTensor(),
        transforms.Normalize((0.429, 0.410, 0.382), (0.287, 0.285, 0.435))
    ])
    model_sex_age = sex_age.ResNet50(sex_classes=2, age_classes=5)
    tmp = torch.load('./sex_age/epoch5.pkl')
    tmp2 = {}
    for item in tmp.items():
        key = item[0]
        values = item[1]
        newkey = key[7:]
        tmp2[newkey] = values

    model_sex_age.load_state_dict(tmp2)
    model_sex_age.cuda()
    model_sex_age.eval()


    myexactor = FeatureExtractor(model, exact_list)

    model.eval()
    model.cuda()

    seq_info_list = []
    metric_list = []
    tracker = []
    angle_length = len(sequence_dir)
    for index in range(0, angle_length):
        seq_info_list.append(gather_sequence_info(sequence_dir[index], detection_file[index]))
        metric_list.append(nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget))
        tracker.append(Tracker(metric_list[index]))
    results1,results2,results3 = [],[],[]

    #define feature gallery
    dic_feature = {}
    def frame_callback(vis, frame_idx, seq_info, viewer,angle):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        # Update tracker.
        tracker[angle-1].predict()
        tracker[angle-1].update(detections, image,  myexactor, dic_feature, frame_idx, global_next_id, angle)
        id_dic = {}
        for index, track in enumerate(tracker[angle-1].tracks):
            str_id = str(track.track_id)
            if str_id in id_dic.keys():
                track.track_id = global_next_id[0]
                global_next_id[0] += 1
            else:
                id_dic[str_id] = (index, track.state)

        # Update visualization.
        if display:
            vis.set_image(image.copy(),viewer,angle)
            #print("deep_sort angle: "+str(angle))
            vis.draw_detections(detections,viewer,angle)
            vis.draw_trackers(tracker[angle-1].tracks,viewer,angle)

        # Store results.
        for track in tracker[angle-1].tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            print(angle)
            if angle == 1:
                print("angle1")
                results1.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.sex, track.person_age])
            if angle == 2:
                print("angle2")
                results2.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.sex, track.person_age])
            if angle == 3:
                print("angle3")
                results3.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.sex, track.person_age])
            ### save gallery
            if (frame_idx) % 4 == 0:
                for i in range(4):
                    if bbox[i] < 0 :
                        bbox[i] = 0
                img = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
                temp = img.copy()
                transform_test = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                img = transform_test(img)
                img = torch.unsqueeze(img, 0)
##sex age forward
                track_id = track.track_id
                img = img.cuda()
                sex_output, age_output = model_sex_age(img)
                pred1 = sex_output.data.max(1)[1]
                pred2 = age_output.data.max(1)[1]
                age = age_dict[str(int(pred2))]
                sex = sex_dict[str(int(pred1))]
                track.person_age = age
                track.sex = sex
##end
                f1 = myexactor(img)
                a1 = normalize(pool2d(f1[0], type='max'))
                if str(track_id) not in dic_feature.keys():
                    dic_feature[str(track_id)] = []
                    dic_feature[str(track_id)].append((a1, angle, (bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3])))
                else:
                    if len(dic_feature[str(track_id)]) > 100:
                        del(dic_feature[str(track_id)][0])
                    dic_feature[str(track_id)].append((a1, angle, (bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3])))
    if display:
        visualizer = visualization.Visualization(seq_info_list, update_ms=5,angle_length=angle_length)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback,angle_length)

    # Store results.
    f = open("./hypothese1.txt", 'w')
    for row in results1:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,%s,%s,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]),file=f)
    f2 = open("./hypothese2.txt", 'w')
    for row in results2:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,%s,%s,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]), file=f2)
    f3 = open("./hypothese3.txt", 'w')
    for row in results3:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,%s, %s, 1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]), file=f3)


def parse_args():
    """ Parse command line arguments.
    """
    print("in parse_function")
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    print("start parsing")
    args = parse_args()
    print("parse complete")
    
    run(
        args.sequence_dir.split(','), args.detection_file.split(','), args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
    '''
    run(
        args.sequence_dir, args.sequence_dir2, args.detection_file, args.detection_file2, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
    '''
