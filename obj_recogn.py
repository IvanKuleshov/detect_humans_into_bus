import os
from typing_extensions import Literal # так как python 3.7
from pixellib.instance import instance_segmentation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def object_detection_on_an_image(img, infer_speed: Literal['average', 'fast', 'rapid'] = None):
    segment_image = instance_segmentation(infer_speed=infer_speed)
    segment_image.load_model("mask_rcnn_coco.h5")

    target_class = segment_image.select_target_classes(person=True)

    result = segment_image.segmentFrame(
        # image_path="1city.jpg",
        frame=img,
        show_bboxes=True,
        segment_target_classes=target_class,
        # extract_segmented_objects=True,
        # save_extracted_objects=True,
        # mask_points_values=True,
        # output_image_name="output.jpg"
    )
    return result
