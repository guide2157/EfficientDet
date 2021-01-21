import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.anchors import anchors_for_shape, anchor_targets_bbox_tfdataset, AnchorParameters, compute_overlap_tfdataset


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
        self.anchors = anchors_for_shape((image_size, image_size), anchor_params=self.anchor_parameters)

    def compute_targets(self, images, bounding_boxes, labels):
        batches_targets = anchor_targets_bbox_tfdataset(
            self.anchors,
            images.numpy(),
            bounding_boxes.numpy(),
            labels.numpy(),
            num_classes=self.num_classes
        )
        return list(batches_targets)

    def compute_overlap(self, bboxes):
        return compute_overlap_tfdataset(self.anchors, bboxes.numpy())

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
        bounding_boxes = tf.sparse.to_dense(example['bounding_box'])
        labels = tf.sparse.to_dense(example['labels'])
        bounding_boxes = tf.reshape(bounding_boxes, [-1, 4])
        if self.test:
            return image, bounding_boxes, labels

        classification, regression = tf.py_function(self.compute_targets, inp=[image, bounding_boxes, labels],
                                                    Tout=[tf.float32, tf.float32])
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
        "/Users/kittipodpungcharoenkul/Documents/ChulaXrayProject/Tumor/EfficientDet/datasets/VOC2007/"
        "tfrecord/all-5011.tfrec"]
    ANCHORS_PARAMETERS = AnchorParameters(
        ratios=(0.65, 1.),
        sizes=(16, 32, 64, 128, 256),
        scales=(1, 1.5))
    ctc_generator = CTCGenerator(num_classes=20, image_size=512, batch_size=5, anchor_parameters=ANCHORS_PARAMETERS,
                                 training=True)
    train_generator = ctc_generator.generate_dataset(paths)
    anchors = ctc_generator.anchors
    # for image, overlap in train_generator:
    #     overlaps = overlap.numpy()[0][0]
    #     argmax_overlaps_inds = np.argmax(overlaps, axis=-1)
    #     max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
    #     positive_indices = max_overlaps >= 0.4
    #     ignore_indices = (max_overlaps > 0.5) & ~positive_indices
    #     plt.imshow(image[0].numpy())

    # for image, bounding_boxes, labels in train_generator:
    for image, classification, regression in train_generator:
        classification = classification.numpy()[0]
        regression = regression.numpy()[0]
        # uniques = np.unique(regression[:, -1])

        # targets = ctc_generator.compute_targets(image[0], bounding_boxes[0], labels[0])
        # classification, regression = targets
        image = image.numpy()[0]
        #
        valid_ids = np.array(np.where(regression[:, -1] == 1)).flatten()
        if valid_ids.shape[0] == 0:
            continue
        #     # print(valid_ids)
        boxes = anchors[valid_ids]
        deltas = classification[valid_ids]
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
        # for x1_, y1_, x2_, y2_ in bounding_boxes:
        #     x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
        #     start_point = (x1_, y1_)
        #     end_point = (x2_, y2_)
        #     cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        plt.imshow(image)
        plt.show()
