from pathlib import Path
from typing import List, Tuple

from PIL import Image


def crop_images(image_paths: List[Path],
                output_dir: Path,
                crop_size: Tuple[int, int] = (640, 640),
                overlap: int = 0) -> None:
    """Crops all images in the provided list of paths into smaller 640x640
       tiles and saves them.

    Args:
        image_paths (List[Path]): List of paths to the images.
        output_dir (Path): Directory where the cropped images will be saved.
        crop_size (Tuple[int, int], optional): The target size for cropping
          (width, height)  Defaults to (640, 640).
        overlap (int, optional): Overlap size between crops to achieve
          additional tiles. Defaults to 0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_width, crop_height = crop_size
    for img_path in image_paths:
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            crop_count = 0
            step_x = crop_width - overlap
            step_y = crop_height - overlap
            for top in range(0, img_height - crop_height + 1, step_y):
                for left in range(0, img_width - crop_width + 1, step_x):
                    right = left + crop_width
                    bottom = top + crop_height
                    cropped_img = img.crop((left, top, right, bottom))
                    crop_count += 1
                    save_path = (
                        output_dir / f"{img_path.stem}_crop_{crop_count}.jpg"
                    )
                    cropped_img.save(save_path)
                    print(f"Cropped and saved image: {save_path}")

            print(f"Total crops for {img_path.name}: {crop_count}")

if __name__ == "__main__":
    RAW_DATA_PATH = Path("data/selection")
    image_paths = list(RAW_DATA_PATH.glob('*.JPG'))
    output_directory = Path("data/cropped_images")
    crop_images(image_paths, output_directory, overlap=0)
