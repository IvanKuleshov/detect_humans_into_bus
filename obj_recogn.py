import os
from pixellib.instance import instance_segmentation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def object_detection_on_an_image(img):
    segment_image = instance_segmentation()
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
        #output_image_name="output.jpg"
    )

    objects_count = len(result[0]["scores"])
    print(f"Objects{objects_count}")
    return result
