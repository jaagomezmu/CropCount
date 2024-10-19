import cv2
import os
from pathlib import Path
import random


class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_coordinates(self):
        return (self.x1, self.y1, self.x2, self.y2)


class ImageAnnotatorModel:
    def __init__(self, image_path, output_dir, class_id=0):
        self.image_path = image_path
        # Cambiar la carpeta de salida
        self.output_dir = output_dir
        # Asegurarse de que la carpeta exista
        os.makedirs(self.output_dir, exist_ok=True)
        self.class_id = class_id
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        self.bounding_boxes = []

    def add_bounding_box(self, bbox):
        """Agregar una nueva caja delimitadora a la lista"""
        self.bounding_boxes.append(bbox)

    def save_annotations(self, mode="w"):
        """Guardar las anotaciones en formato YOLO"""
        # Usar el mismo nombre de imagen pero con extensión .txt
        filename = os.path.basename(self.image_path).replace(
            '.jpg', '.txt')  # Cambiar según la extensión de la imagen
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, mode=mode) as f:
            for bbox in self.bounding_boxes:
                # Obtener coordenadas de la caja y normalizarlas
                x1, y1, x2, y2 = bbox.get_coordinates()
                x_center = ((x1 + x2) / 2) / self.width
                y_center = ((y1 + y2) / 2) / self.height
                box_width = (x2 - x1) / self.width
                box_height = (y2 - y1) / self.height

                # Escribir en formato YOLO
                f.write(f"{self.class_id} {x_center:.6f} {y_center:.6f} {
                        box_width:.6f} {box_height:.6f}\n")


class ImageAnnotatorView:
    def __init__(self, model, harcoded=False):
        self.model = model
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_image = model.image.copy()
        self.current_box = None
        self.harcoded = harcoded

    def draw_existing_boxes(self):
        """Dibujar todas las cajas ya existentes"""
        if self.harcoded:
            with open("data/cropped_labels/" + self.harcoded, mode="r") as f:
                lines = f.read().splitlines()

                for line in lines:
                    xc, yc, w, h = str(line).split(" ")[1:]
                    w = float(w) * 640
                    h = float(h) * 640
                    x1 = float(xc)*640 - (w/2)
                    y1 = float(yc)*640 - (h/2)
                    x2 = x1 + float(w)
                    y2 = y1 + float(h)
                    self.model.bounding_boxes.append(
                        BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
                    )

            for bbox in self.model.bounding_boxes:
                x1, y1, x2, y2 = bbox.get_coordinates()
                cv2.rectangle(self.current_image, (x1, y1),
                              (x2, y2), (0, 255, 255), 1)
            self.harcoded = None
        else:
            for bbox in self.model.bounding_boxes:
                x1, y1, x2, y2 = bbox.get_coordinates()
                cv2.rectangle(self.current_image, (x1, y1),
                              (x2, y2), (0, 255, 255), 1)


    def draw_box(self, event, x, y, flags, param):
        """Función callback del mouse para dibujar cajas delimitadoras"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            print(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_image = self.model.image.copy()
                self.draw_existing_boxes()
                cv2.rectangle(self.current_image,
                              self.start_point, (x, y), (0, 0, 255), 1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            cv2.rectangle(self.current_image, self.start_point,
                          self.end_point, (0, 0, 255), 1)
            self.current_box = BoundingBox(
                self.start_point[0],
                self.start_point[1],
                self.end_point[0],
                self.end_point[1]
            )

    def display_image(self):
        """Mostrar la imagen y registrar el callback del mouse"""
        cv2.namedWindow('Image Annotator')
        cv2.setMouseCallback('Image Annotator', self.draw_box)

        if self.harcoded:
            self.draw_existing_boxes()

        while True:
            cv2.imshow('Image Annotator', self.current_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Presiona 'q' para salir
                break
            elif key == ord('c'):
                self.command = True
                if self.current_box:
                    self.model.add_bounding_box(self.current_box)
                    self.model.save_annotations('a')
                    self.current_box = None
            elif key == ord('s'):  # Presiona 's' para guardar anotaciones
                self.model.save_annotations()
                print("Annotations saved!")
                break

        cv2.destroyAllWindows()


class ImageAnnotatorController:
    def __init__(
        self,
        path_images_cropped,
        path_labels_cropped,
        batch_size=1,
        harcoded=None
    ):
        self.batch_size = batch_size
        self.harcoded = harcoded
        self.path_labels_cropped = path_labels_cropped
        self.path_images_cropped = self.validate_images_path(
            path_images_cropped)
        self.list_images = self.find_available_paths(path_labels_cropped)

    def validate_images_path(self, path_images_cropped, format="*.jpg"):
        """Valida que el path usado contenga imagenes"""
        path_ = Path(path_images_cropped)

        if not path_.exists:
            raise FileNotFoundError(f"El path {path_} no existe")
        if format:
            if not list(path_.glob(format)):
                raise FileNotFoundError(f"El path {path_} no tiene")

        return path_

    def find_available_paths(self, path_labels_cropped):

        if self.harcoded:
            print(f"{[self.path_images_cropped / self.harcoded]}")
            return [self.path_images_cropped / self.harcoded]
        else:
            path_ = self.validate_images_path(path_labels_cropped, format=None)

            labels_list = [x.stem for x in list(path_.glob("*.txt"))]
            path_images = [x.stem for x in list(
                self.path_images_cropped.glob("*.jpg"))]
            filenames = [x for x in path_images if x not in labels_list]
            list_images = [
                self.path_images_cropped / f"{x}.jpg" for x in filenames
            ]
            list_images = [x for x in list_images if x.exists()]
            return random.sample(list_images, self.batch_size)

    def run(self):
        """Ejecutar la herramienta de anotación"""
        self.view.display_image()

    def run_multiple_annotations(self):

        for path_image in self.list_images:
            print(f"The path image is: {path_image}")
            if self.harcoded:
                image_path = path_image.stem + ".jpg"
                image_path = self.path_images_cropped / image_path
                self.model = ImageAnnotatorModel(
                    image_path, self.path_labels_cropped)
                self.view = ImageAnnotatorView(self.model, harcoded=self.harcoded)
            else:
                self.model = ImageAnnotatorModel(
                    path_image, self.path_labels_cropped)
                self.view = ImageAnnotatorView(self.model, harcoded=False)
            self.run()
