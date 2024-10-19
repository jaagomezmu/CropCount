import argparse
from utils.processor import ImageAnnotatorController

def main():
    parser = argparse.ArgumentParser(
        description="Controlador para anotación de imágenes"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Tamaño del batch para la anotación de imágenes"
    )

    args = parser.parse_args()

    controller = ImageAnnotatorController(
        path_images_cropped="data/cropped_images",
        path_labels_cropped="data/cropped_labels",
        batch_size=args.batch_size
    )

    controller.run_multiple_annotations()
    print("Done!")

if __name__ == "__main__":
    main()
