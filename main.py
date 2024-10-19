import argparse
from pathlib import Path

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
    parser.add_argument(
        "--harcoded",
        action="store_true",
        help="Automaticamente harcodear el ultimo archivo editado, si es True"
    )

    args = parser.parse_args()
    harcoded = None
    ####### Validar harcoded
    if args.harcoded:
        list_labels = list(Path("data/cropped_labels").glob("*.txt"))
        max_path = max(list_labels, key=lambda f: f.stat().st_mtime)
        harcoded = str(max_path.name)
    print(harcoded)

    controller = ImageAnnotatorController(
        path_images_cropped="data/cropped_images",
        path_labels_cropped="data/cropped_labels",
        batch_size=args.batch_size,
        harcoded=harcoded
    )

    controller.run_multiple_annotations()
    print("Done!")

if __name__ == "__main__":
    main()
