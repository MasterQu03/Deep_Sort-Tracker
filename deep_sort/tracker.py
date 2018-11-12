# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import time
import cv2
import math
from reflect import hit_pos
import matplotlib.pyplot as plt
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
import torch
from torchvision.transforms import *
import os
from util.utils import *
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine, euclidean
import datetime
def pool2d(tensor, type= 'max'):
    sz = tensor.size()
    if type == 'max':
        x = torch.nn.functional.max_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    if type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(sz[2]/8, sz[3]))
    x = x[0].cpu().data.numpy()
    x = np.transpose(x,(2,1,0))[0]
    return x
class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, image, myexactor, dic_feature,  frame_idx, gloabl_next_id, angle):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], image, myexactor, dic_feature, frame_idx, gloabl_next_id, angle)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, image, myexactor, dic_feature, frame_idx, gloabl_next_id, angle):
        print("person number:", len(dic_feature.keys()))
        for key in dic_feature.keys():
            print("id image number:", len(dic_feature[key]))
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # print(detection.tlwh)
        # extract feature for new tracker
        bbox = detection.tlwh
        area = bbox[2] * bbox[3]
        if area < 10000:
            return
        for i in range(4):
            if bbox[i] < 0:
                bbox[i] = 0
        img = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        img = cv2.resize(img, (128, 256))
        temp = img.copy()
        transform_test = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transform_test(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        # print(img.shape)
        f1 = myexactor(img)
        a1 = normalize(pool2d(f1[0], type='max'))
        info1 = (a1, angle, (bbox[0], bbox[1]))
        result = [-1]
        if len(dic_feature.keys()) != 0:
            # plt.figure()
            # plt.imshow(temp)
            # plt.show()
            result = self._checkid_(info1, dic_feature)
        if result[0] > 0:
            track = Track(
                mean, covariance, result[0], self.n_init, self.max_age,
                detection.feature)
            current_time = datetime.datetime.now()
            track.update_num = current_time.timestamp()
            track.rank3 = result
            self.tracks.append(track)
        else:
            print("next id", gloabl_next_id[0])
            self.tracks.append(Track(
                mean, covariance, (gloabl_next_id[0]), self.n_init, self.max_age,
                detection.feature))
            # print(self._next_id)
            dic_feature[str(gloabl_next_id[0])] = []
            dic_feature[str(gloabl_next_id[0])].append((a1, angle, (bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3])))
            gloabl_next_id[0] += 1

    def _comdist_(self, a1, a2):
        dist = np.zeros((8, 8))
        for i in range(8):
            temp_feat1 = a1[i]
            for j in range(8):
                temp_feat2 = a2[j]
                dist[i][j] = euclidean(temp_feat1, temp_feat2)
        d, D, sp = dtw(dist)
        return d

    def _get_punish_score_(self, geo_distance):
        if geo_distance < 400:
            return 0
        else:
            return (geo_distance-600)/100 * 0.1

    def _checkid_(self, info1, dic_feature):
        avg = {}
        imgs = []
        D  = []
        a1 = info1[0]
        for i in dic_feature.keys():
            dist = []
            for j in dic_feature[i]:
                if info1[1] == j[1]:
                    distance = 1000
                else:
                    distance = self._comdist_(a1, j[0])
                    # x1, y1 = hit_pos(info1[2][0], info1[2][1], info1[1])
                    # x2, y2 = hit_pos(j[2][0], j[2][1], j[1])
                    # print(info1[2][0], info1[2][1], info1[1])
                    # print(j[2][0], j[2][1], j[1])
                    # geo_distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
                    # geo_distance = math.sqrt(geo_distance)
                    # extra = self._get_punish_score_(geo_distance)
                    # distance += extra
                dist.append(distance)
                D.append(distance)
            avg_dist = np.min(np.array(dist))
            avg[i] = avg_dist
        min_avg = np.min(list(avg.values()))
        index = np.argsort(list(avg.values()))[0:3]
        ids = []
        if min_avg <= 0.6:
            for i in index:
                id = list(avg.keys())[i]
                id = int(id)
                ids.append(id)
            return ids
        else:
            return [-1]

