# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """
    def __init__(self, seq_info_list, update_ms,angle_length):
        image_shape = seq_info_list[0]["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.seq = seq_info_list
        self.viewer = ImageViewer(
            update_ms, (420,300), "Figure %s" % seq_info_list[0]["sequence_name"],angle_length)
        self.viewer.thickness = 2
        self.frame_idx = seq_info_list[0]["min_frame_idx"]
        self.last_idx = seq_info_list[0]["max_frame_idx"]

    def run(self, frame_callback,angle_length):
        self.viewer.run(lambda: self._update_fun(frame_callback,self.viewer))

    def _update_fun(self, frame_callback, viewer):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        index = 1
        for item in self.seq:
            frame_callback(self, self.frame_idx, item, viewer,index)
            index = index + 1
        #frame_callback(self, self.frame_idx, self.seq2, viewer,2)
        self.frame_idx += 1
        return True

    def set_image(self, image, viewer,angle):
        viewer.image_list[angle-1] = image
        #if angle == 1:
        #    viewer.image = image
        #elif angle == 2:
        #    viewer.image2 = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections,viewer,angle):
        #print("vis angle: "+str(angle))
        viewer.thickness = 2
        viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            viewer.rectangle(detection.tlwh[0],detection.tlwh[1],detection.tlwh[2],detection.tlwh[3],None,angle)

    def draw_trackers(self, tracks,viewer,angle):
        viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            viewer.color = create_unique_color_uchar(track.track_id)
            viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id) + "age" + track.person_age + "sex" + track.sex, angle=angle)
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#
