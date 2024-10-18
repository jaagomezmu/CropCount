from utils.processor import ImageAnnotatorController

controller = ImageAnnotatorController(
    path_images_cropped="data/cropped_images",
    path_labels_cropped="data/cropped_labels"
)
controller.run_multiple_annotations()
print("Done!")
