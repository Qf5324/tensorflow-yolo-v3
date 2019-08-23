# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_video', 'test.mp4', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', 'result.mp4', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
# tf.app.flags.DEFINE_string(
#     'weights_file', 'darknet_weight/yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
# tf.app.flags.DEFINE_string(
#     'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', 'frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.3, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.45, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')



def main(argv=None):
    vid = cv2.VideoCapture(FLAGS.input_video)
    # vid = cv2.VideoCapture(0)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))
    # print('########', video_frame_cnt)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )
    t0 = time.time()
    frozenGraph = load_graph(FLAGS.frozen_model)
    with tf.Session(graph=frozenGraph, config=config) as sess:
        print("Loaded graph in {:.2f}s".format(time.time() - t0))

        for i in range(video_frame_cnt):
            #    j = j+1
            #   print('j:', j)
            ret, img = vid.read()
            if ret == False:
                break
            # cv -> plt
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
            img_resized = img_resized.astype(np.float32)
            classes = load_coco_names(FLAGS.class_names)

            boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

            filtered_boxes = non_max_suppression(detected_boxes,
                                            confidence_threshold=FLAGS.conf_threshold,
                                            iou_threshold=FLAGS.iou_threshold)
            print("Predictions found in {:.2f}s".format(time.time() - t0))

            draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


            cv2.imshow('image', img)
            videoWriter.write(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()

        videoWriter.release()

if __name__ == '__main__':
    tf.app.run()
