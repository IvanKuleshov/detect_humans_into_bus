import cv2
from video_cap import VideoCaptureGen
from obj_recogn import object_detection_on_an_image


def main1():
    file = 'VID_20220912_191010.mp4'  # заменить на открытике файла
    # создадим объект VideoCapture для захвата видео
    video_cap = cv2.VideoCapture(file)
    vc = VideoCaptureGen(video_cap)

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # Формат кодирования
    video_out = cv2.VideoWriter(r'exit_.avi', fourcc, vc.fps, (vc.width, vc.height), True)

    for frame, img in vc.get_frames_gen():
        # Запускаем распознавание и определяем координаты прямоугольников и вероятности классов
        img_ = img.copy()  # При распознавании входящая картинка портится
        boxes_result = object_detection_on_an_image(img_)  # результат распознавания, содержит координаты и вер-ти
        img_out = vc.display_boxes(boxes_result, 0.8)  # рисуем на фрейме прямоугольники, в которых определены люди

        cv2.imwrite('output_.jpg', img_out)
        print(f"Objects{vc.objects_count}")
        return img_out
        # video_out.write(img)
        cv2.imshow("Cat", img)
        # при нажатии клавиши "q", совершаем выход
        if cv2.waitKey(25) == ord('q'):
            break

    # освобождаем память от переменной cap
    video_cap.release()
    video_out.release()
    # закрываем все открытые opencv окна
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dd = main1()
