import cv2
from tqdm import tqdm
from video_cap import VideoCaptureGen
from obj_recogn import object_detection_on_an_image


def human_detect_on_video(file: str, frame_per_second: int):
    # создадим объект VideoCapture для захвата видео
    video_cap = cv2.VideoCapture(file)
    vc = VideoCaptureGen(video_cap, width=None, aspect_ratio=True)

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # Формат кодирования
    video_out = cv2.VideoWriter(r'exit_.avi', fourcc, vc.fps, (vc.width, vc.height), True)

    detect_per_second = int(vc.fps) // frame_per_second
    df = []  # покадрово данные о числе распознанных

    for frame, img in tqdm(vc.get_frames_gen(), total=vc.frames):
        # распознавание раз в несколько секунд
        if frame % detect_per_second == 0:
            # Запускаем распознавание и определяем координаты прямоугольников и вероятности классов
            img_ = img.copy()  # При распознавании входящая картинка портится
            # результат распознавания, содержит координаты и вер-ти
            boxes_result = object_detection_on_an_image(img_, infer_speed=None)

        # распознавание делаем раз в несколько кадров, а рамки рисуем всё время
        img = vc.display_boxes(boxes_result, 0.4)  # рисуем на фрейме прямоугольники, в которых определены люди
        # cv2.imwrite('output_.jpg', img_out)

        video_out.write(img)
        df.append(vc.objects_count)
        # cv2.imshow("Cat", img)

        # при нажатии клавиши "q", совершаем выход
        if cv2.waitKey(25) == ord('q'):
            break

    # освобождаем память от переменной cap
    video_cap.release()
    video_out.release()
    # закрываем все открытые opencv окна
    cv2.destroyAllWindows()


if __name__ == '__main__':
    file = 'VID_20220912_191010.mp4'  # заменить на открытике файла
    human_detect_on_video(file=file, frame_per_second=1)
