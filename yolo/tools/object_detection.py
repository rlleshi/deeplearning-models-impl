import argparse
import os
import os.path as osp
import sys
import time
from threading import Thread

import cv2
import numpy as np
from pytube import YouTube
from rich.console import Console

CONSOLE = Console()


class ThreadedCamera(object):
    """https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-
    video-capture-is-slow-how-to-drop-frames-or-get-sync."""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.status = False
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None


def get_classes(file):
    with open(file, 'r') as f:
        return f.read().splitlines()


def bootstrap_model(config, weights, scale, size, swapRB):
    """Initialize the model with the config & weight file.

    Set the pre-processing layer.
    """
    model = cv2.dnn.readNetFromDarknet(config, weights)
    model = cv2.dnn_DetectionModel(model)
    # set pre-processing params for frame, including mean to normalize
    model.setInputParams(scale=scale, size=size, swapRB=swapRB)
    return model


def get_results(model, img, conf_thr, nms_thr):
    """" Get detection results of the model.

    Returns:
        classId: [0, 79], e.g. 0 for human
        scores: [0, 1], e.g. 97% confidence for yolo_human.jpg
        boxes: [(top_left-x, top_left-y), (width, height)],
                e.g. [(193, 52), (218, 511)] for yolo_human.jpg
    """
    return model.detect(img, confThreshold=conf_thr, nmsThreshold=nms_thr)


def show_image(img):
    """Show an image to the screen."""
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='object detection using YOLO')
    parser.add_argument('--image', help='path to image file')
    parser.add_argument('--video', help='path to video file or youtube link')
    parser.add_argument('--webcam',
                        action='store_true',
                        help='whether to use webcam')
    parser.add_argument('--out-dir', default=None, help='out dir')
    parser.add_argument(
        '--find',
        nargs='+',
        default=None,
        help='class to find in the input. For images it highlights only these '
        'classes. For video/webcam it includes only those frames where '
        'the specified classes are found')
    parser.add_argument('--device',
                        default='cpu',
                        choices=['gpu', 'cpu'],
                        help='device')
    parser.add_argument('--classes',
                        default='resources/coco.names',
                        help='path to the file containing the classes')
    parser.add_argument('--config',
                        default='model/yolov4.cfg',
                        help='model config file')
    parser.add_argument('--weights',
                        default='model/yolov4.weights',
                        help='model weights')
    parser.add_argument('--conf-thr',
                        type=float,
                        default=0.5,
                        help='confidence threshold')
    parser.add_argument('--nms-thr',
                        type=float,
                        default=0.4,
                        help='non-maximum suppression threshold')
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=[416, 416],
                        help='input image size')
    args = parser.parse_args()
    return args


def add_results_to_img(img, results, classes):
    class_ids, scores, boxes = results
    for (class_id, score, box) in zip(class_ids, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0),
                      thickness=2)
        text = '%s: %.2f' % (classes[class_id], score)
        cv2.putText(img,
                    text, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color=(0, 255, 0),
                    thickness=2)
    return img


def filter_results(results, find):
    class_ids, scores, boxes = results
    new_class_ids, new_scores, new_boxes = [], [], []
    for (class_id, score, box) in zip(class_ids, scores, boxes):
        if class_id in find:
            new_class_ids.append(class_id)
            new_scores.append(score)
            new_boxes.append(box)
    return [new_class_ids, new_scores, new_boxes]


def main():
    args = parse_args()
    classes = get_classes(args.classes)
    class_to_id = {label: i for i, label in enumerate(classes)}
    model = bootstrap_model(args.config,
                            args.weights,
                            scale=1 / 255,
                            size=args.shape,
                            swapRB=True)

    if args.find is not None:
        find = [class_to_id.get(f, None) for f in args.find]
        if not any(find) and 0 not in find:
            CONSOLE.print(
                f'Category not found. Please check {args.classes} '
                'for a list of supported categories.',
                style='yellow')
            sys.exit()
        # stack to store frames if detector loses the object temporarily
        stack, stack_len = [], 150

    if args.image:
        img = cv2.imread(args.image)
        results = get_results(model, img, args.conf_thr, args.nms_thr)
        if args.find is not None:
            results = filter_results(results, find)
            if not results[0]:
                CONSOLE.print(f'{args.find} was/were not found on the image',
                              style='red')
                sys.exit()

        img = add_results_to_img(img, results, classes)
        out_f = f'yolo_{args.image.split(os.sep)[-1]}'
        if args.out_dir is not None:
            out_f = osp.join(args.out_dir, out_f)
        cv2.imwrite(out_f, img.astype(np.uint8))
        CONSOLE.print(f'Finished. Saved {out_f}', style='green')
        show_image(img)

    if args.video:
        if args.video.startswith('https://www.youtube.com/'):
            CONSOLE.print('Downloading youtube video...', style='green')
            args.video = YouTube(
                args.video).streams.filter(res='720p').first().download()

        video = cv2.VideoCapture(args.video)
        CONSOLE.print(
            f'Processing {args.video} with fps '
            f'{video.get(cv2.CAP_PROP_FPS)} and '
            f'{video.get(cv2.CAP_PROP_FRAME_COUNT)} frames\n',
            'On cpu it takes ~0.22s to process a frame',
            style='green')
        out_f = f'yolo_{args.video.split(os.sep)[-1]}'

        last_find_frame, skip_mode = 0, True  # used for args.find
        if args.out_dir is not None:
            out_f = osp.join(args.out_dir, out_f)

        video_writer = cv2.VideoWriter(
            out_f, cv2.VideoWriter_fourcc(*'MP4V'),
            video.get(cv2.CAP_PROP_FPS),
            (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
             round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        counter = 0
        start_time = time.time()
        while cv2.waitKey(1) < 0:
            success, frame = video.read()
            counter += 1
            if not success:
                CONSOLE.print(
                    f'Finished {round((time.time() - start_time)/60, 2)} '
                    f'minutes. Saved {out_f}',
                    style='green')
                video.release()
                break

            if args.find:
                if len(stack) > stack_len:
                    # add the last frames after the object has been found
                    # to smooth out the results
                    if not skip_mode:
                        for future_frame in stack:
                            video_writer.write(future_frame.astype(np.uint8))

                    # stack cannot be more than `stack_len`
                    stack.pop(0)

                    # enable skip mode if no object was found for `stack_len`
                    # frames
                    skip_mode = True

                stack.append(frame)
                last_find_frame += 1

                # skip every 2 out of 3 frames to speed things up
                if skip_mode & counter % 3 != 0:
                    continue

            results = get_results(model, frame, args.conf_thr, args.nms_thr)

            if args.find:
                results = filter_results(results, find)
                if not results[0]:
                    continue

                # temporarily disable skip mode since object was found
                skip_mode = False

                # add some previous frames before the current discovered frame
                if len(stack) == stack_len + 1:
                    stack.pop()
                    for past_frame in stack:
                        video_writer.write(past_frame.astype(np.uint8))

                # it might be that the model lost track of the object
                # momentarily, so add past frames from stack
                if last_find_frame < stack_len:
                    # remove current frame
                    stack.pop()

                    for past_frame in stack:
                        video_writer.write(past_frame.astype(np.uint8))

                stack.clear()
                last_find_frame = 0

            # do not add bbox for finding-objects mode
            if not args.find:
                frame = add_results_to_img(frame, results, classes)
            video_writer.write(frame.astype(np.uint8))

    if args.webcam:
        camera = ThreadedCamera()
        CONSOLE.print('Processing input from webcam...', style='green')
        out_f = 'webcam.avi'
        if args.out_dir is not None:
            out_f = osp.join(args.out_dir, out_f)
        video_writer = cv2.VideoWriter(
            filename=out_f,
            fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            # fps=camera.capture.get(cv2.CAP_PROP_FPS),
            fps=5,
            frameSize=(round(camera.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       round(camera.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:
            frame = camera.grab_frame()
            if frame is None:
                continue

            # * cpu inference speed is roughly 0.22s, which means max 5 frames
            # can be inferred per second
            results = get_results(model, frame, args.conf_thr, args.nms_thr)
            if args.find is not None:
                results = filter_results(results, find)

            frame = add_results_to_img(frame, results, classes)
            video_writer.write(frame.astype(np.uint8))
            cv2.imshow('webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                CONSOLE.print(f'Saved {out_f}', style='green')
                camera.capture.release()
                video_writer.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    main()
