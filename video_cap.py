import cv2
import colorsys
import random
import numpy as np


class VideoCaptureGen:

    def __init__(self, cap: cv2.VideoCapture, width: int = None, height: int = None,
                 aspect_ratio: bool = True):
        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize_ = not (width is None)

        if width is not None:
            if height is not None:
                self.height = height
            if aspect_ratio:
                self.height = int(self.height * width / self.width)

            self.width = width

        self.fps = cap.get(cv2.CAP_PROP_FPS)  # Получить частоту кадров прочитанного видео
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img = None
        self.objects_count = 0

    @staticmethod
    def random_colors(N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def display_boxes(self, box_score_result, thres=0.5):
        # Оставляем значения прямоугольников и вероятностей класса, если вероятность выше порога
        boxes, scores = zip(*(
            (i, j) for i, j in zip(box_score_result[0]['rois'], box_score_result[0]['scores']) if j >= thres
        ))

        # разные цвета для прямоугольников
        self.objects_count = len(boxes)
        colors = self.random_colors(self.objects_count)
        img_out = self.img
        title = '{}: {}'.format('humans', self.objects_count)

        for i, color in enumerate(colors):
            if not np.any(boxes[i]):
                continue

            # рисуем рамку
            y1, x1, y2, x2 = boxes[i]

            color_rec = [int(c) for c in np.array(colors[i]) * 255]
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), color_rec, 1)

            # выводим надписи
            # score = scores[i] if scores is not None else None
            # caption = '{} {:.2f}'.format('human', score) if score else 'human'
            # img_out = cv2.putText(img_out, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, color=(255, 255, 255))
            img_out = cv2.putText(img_out, title, (int(self.width / 3), self.height - 3), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                  color=(255, 255, 255))

        return img_out

    # генератор фреймов
    def get_frames_gen(self):
        frames = 0
        while self.cap.isOpened():
            frames += 1
            _, img = self.cap.read()
            if img is None:
                break
            if self.resize_:
                img = cv2.resize(img, (self.width, self.height))

            self.img = img
            yield frames, img
