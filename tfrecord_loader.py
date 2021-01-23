import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.anchors import anchors_for_shape, AnchorParameters


class CTCGenerator:
    def __init__(self, num_classes, image_size, batch_size=4, training=False, test=False, anchor_parameters=None):
        super(CTCGenerator, self).__init__()
        self.batch_size = batch_size
        self.training = training
        self.test = test
        self.num_classes = num_classes
        if not anchor_parameters:
            self.anchor_parameters = AnchorParameters.default
        else:
            self.anchor_parameters = anchor_parameters
        self.anchors = anchors_for_shape((image_size, image_size), anchor_params=self.anchor_parameters,
                                         pyramid_levels=[3, 4, 5])
        self.image_size = image_size
        self.anchors_area = self.area(self.anchors)

    @tf.function
    def area(self, boxlist):
        """Computes area of boxes.
        Args:
          boxlist: BoxList holding N boxes
        Returns:
          a tensor with shape [N] representing box areas.
        """
        y_min, x_min, y_max, x_max = tf.split(boxlist, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    @tf.function
    def compute_overlap(self, boxlist1, boxlist2):
        """
            Args
                a: (N, 4) ndarray of float
                b: (K, 4) ndarray of float

            Returns
                overlaps: (N, K) ndarray of overlap between boxes and query_boxes
            """
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            boxlist1, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            boxlist2, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths

    @tf.function
    def compute_gt_annotations(self, bounding_boxes,
                               negative_overlap=0.3,
                               positive_overlap=0.85):
        overlaps = self.compute_overlap(self.anchors, bounding_boxes) / self.anchors_area[..., tf.newaxis]
        argmax_overlaps_inds = tf.cast(tf.math.argmax(overlaps, axis=-1), tf.int32)
        max_overlaps = tf.gather_nd(overlaps, tf.concat(
            [tf.range(overlaps.shape[0])[:, tf.newaxis], argmax_overlaps_inds[:, tf.newaxis]], axis=-1))
        positive_indices = max_overlaps >= positive_overlap
        ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
        return positive_indices, ignore_indices, argmax_overlaps_inds

    @tf.function
    def bbox_transform(self, gt_boxes):
        wa = self.anchors[:, 2] - self.anchors[:, 0]
        ha = self.anchors[:, 3] - self.anchors[:, 1]
        cxa = self.anchors[:, 0] + wa / 2.
        cya = self.anchors[:, 1] + ha / 2.
        w = gt_boxes[:, 2] - gt_boxes[:, 0]
        h = gt_boxes[:, 3] - gt_boxes[:, 1]
        cx = gt_boxes[:, 0] + w / 2.
        cy = gt_boxes[:, 1] + h / 2.
        # Avoid NaN in division and log below.
        ha += 1e-7
        wa += 1e-7
        h += 1e-7
        w += 1e-7
        tx = (cx - cxa) / wa
        ty = (cy - cya) / ha
        tw = tf.math.log(w / wa)
        th = tf.math.log(h / ha)
        return tf.stack([ty, tx, th, tw], axis=1)

    @tf.function
    def compute_regression(self, bounding_boxes, labels):
        positive_indices, ignore_indices, argmax_overlaps_inds = self.compute_gt_annotations(bounding_boxes)
        positive_indices_inf = tf.cast(tf.logical_not(positive_indices), tf.int64) * (self.num_classes * 2)
        positive_indices = tf.cast(positive_indices, tf.float32)
        ignore_indices = tf.cast(ignore_indices, tf.float32)
        class_indicator = positive_indices - ignore_indices

        all_labels = tf.gather(labels, argmax_overlaps_inds)

        all_labels = tf.clip_by_value(all_labels + positive_indices_inf, 0, self.num_classes)
        target_labels = tf.one_hot(all_labels, depth=self.num_classes + 1)[:, :-1]

        # target_labels[positive_indices, labels[argmax_overlaps_inds[positive_indices]].astype(int)] = 1
        target_labels = tf.concat([target_labels, class_indicator[:, tf.newaxis]], axis=-1)
        valid_bounding_boxes = tf.gather(bounding_boxes, argmax_overlaps_inds)
        target_regression_box = self.bbox_transform(valid_bounding_boxes)
        target_regression = tf.concat([target_regression_box, tf.cast(class_indicator[:, tf.newaxis], tf.float32)],
                                      axis=-1)
        return target_regression, target_labels

    @tf.function
    def return_empty_regression(self):
        target_regression = tf.zeros((tf.shape(self.anchors)[0], 4 + 1), dtype=tf.float32)
        target_labels = tf.zeros((tf.shape(self.anchors)[0], self.num_classes + 1), dtype=tf.float32)
        return target_regression, target_labels

    @tf.function
    def compute_targets(self, bounding_boxes, labels):
        if tf.shape(bounding_boxes)[0] > 0:
            return self.compute_regression(bounding_boxes, labels)
        else:
            return self.return_empty_regression()

    @tf.function
    def read_tfrecord(self, example):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "bounding_box": tf.io.VarLenFeature(tf.float32),
            "labels": tf.io.VarLenFeature(tf.int64),
            "index": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, features)

        image = tf.io.decode_png(example['image'])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        ratio = tf.cast(tf.shape(image)[0] / self.image_size, tf.float32)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        bounding_boxes = tf.sparse.to_dense(example['bounding_box'])
        labels = tf.sparse.to_dense(example['labels'])
        bounding_boxes = tf.reshape(bounding_boxes, [-1, 4])
        bounding_boxes = tf.math.divide(bounding_boxes, ratio)
        if self.test:
            return image, bounding_boxes, labels

        regression, classification = self.compute_targets(bounding_boxes, labels)
        return image, classification, regression

    @tf.function
    def augment_img(self, image):
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.5, 2.5)
        return image

    def generate_dataset(self, paths):
        if self.training:
            option_no_order = tf.data.Options()
            option_no_order.experimental_deterministic = False
            dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.experimental.AUTOTUNE) \
                .shuffle(5000).with_options(option_no_order)
            dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(
                lambda img, classification, regression: (self.augment_img(img), classification, regression),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(self.read_tfrecord,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    paths = [
        "/Users/kittipodpungcharoenkul/Downloads/CTC_data/bounding_box/bbox_tfrecord/output-seed42-val-23.tfrec"]
    ANCHORS_PARAMETERS = AnchorParameters(
        strides=(12, 16, 32),
        ratios=[0.5, 1, 2],
        sizes=(16, 32, 64),
        scales=[1])
    # ANCHORS_PARAMETERS = None
    ctc_generator = CTCGenerator(num_classes=1, image_size=512, batch_size=1, anchor_parameters=ANCHORS_PARAMETERS,
                                 test=True)
    train_generator = ctc_generator.generate_dataset(paths)
    anchors = ctc_generator.anchors
    # for image, overlap in train_generator:
    #     overlaps = overlap.numpy()[0][0]
    #     argmax_overlaps_inds = np.argmax(overlaps, axis=-1)
    #     max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
    #     positive_indices = max_overlaps >= 0.4
    #     ignore_indices = (max_overlaps > 0.5) & ~positive_indices
    #     plt.imshow(image[0].numpy())

    # for data in train_generator.take(15):
    #     test = data
    for image, bounding_boxes, labels in train_generator:
        # for image, classification, regression in train_generator:
        #     classification = classification.numpy()[0]
        #     regression = regression.numpy()[0]

        targets = ctc_generator.compute_targets(bounding_boxes[0], labels[0])
        regression, classification, = targets
        classification, regression = classification.numpy(), regression.numpy()
        image = image.numpy()[0]
        #
        valid_ids = np.array(np.where(regression[:, -1] == 1)).flatten()
        # if valid_ids.shape[0] == 0:
        #     continue
        #
        boxes = anchors[valid_ids]
        deltas = regression[valid_ids]
        class_ids = np.argmax(regression[valid_ids], axis=-1)
        mean_ = [0, 0, 0, 0]
        std_ = [0.2, 0.2, 0.2, 0.2]

        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        x1 = boxes[:, 0] + (deltas[:, 0] * std_[0] + mean_[0]) * width
        y1 = boxes[:, 1] + (deltas[:, 1] * std_[1] + mean_[1]) * height
        x2 = boxes[:, 2] + (deltas[:, 2] * std_[2] + mean_[2]) * width
        y2 = boxes[:, 3] + (deltas[:, 3] * std_[3] + mean_[3]) * height

        for x1_, y1_, x2_, y2_, class_id in zip(x1, y1, x2, y2, class_ids):
            x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
            start_point = (x1_, y1_)
            end_point = (x2_, y2_)
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        bounding_boxes = bounding_boxes[0].numpy()
        for x1_, y1_, x2_, y2_ in bounding_boxes:
            x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
            start_point = (x1_, y1_)
            end_point = (x2_, y2_)
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)
        plt.imshow(image)
        plt.show()
