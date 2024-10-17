import cv2
import os

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
        self.output_dir = os.path.join(output_dir, 'labels')  # Cambiar la carpeta de salida
        os.makedirs(self.output_dir, exist_ok=True)  # Asegurarse de que la carpeta exista
        self.class_id = class_id
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        self.bounding_boxes = []  # Lista para almacenar las cajas delimitadoras

    def add_bounding_box(self, bbox):
        """Agregar una nueva caja delimitadora a la lista"""
        self.bounding_boxes.append(bbox)
    
    def save_annotations(self):
        """Guardar las anotaciones en formato YOLO"""
        # Usar el mismo nombre de imagen pero con extensión .txt
        filename = os.path.basename(self.image_path).replace('.jpg', '.txt')  # Cambiar según la extensión de la imagen
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w') as f:
            for bbox in self.bounding_boxes:
                # Obtener coordenadas de la caja y normalizarlas
                x1, y1, x2, y2 = bbox.get_coordinates()
                x_center = ((x1 + x2) / 2) / self.width
                y_center = ((y1 + y2) / 2) / self.height
                box_width = (x2 - x1) / self.width
                box_height = (y2 - y1) / self.height

                # Escribir en formato YOLO
                f.write(f"{self.class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

class ImageAnnotatorView:
    def __init__(self, model):
        self.model = model
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_image = model.image.copy()

    def draw_box(self, event, x, y, flags, param):
        """Función callback del mouse para dibujar cajas delimitadoras"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_image = self.model.image.copy()
                cv2.rectangle(self.current_image, self.start_point, (x, y), (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            cv2.rectangle(self.current_image, self.start_point, self.end_point, (0, 255, 0), 2)
            bbox = BoundingBox(self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1])
            self.model.add_bounding_box(bbox)

    def display_image(self):
        """Mostrar la imagen y registrar el callback del mouse"""
        cv2.namedWindow('Image Annotator')
        cv2.setMouseCallback('Image Annotator', self.draw_box)

        while True:
            cv2.imshow('Image Annotator', self.current_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Presiona 'q' para salir
                break
            elif key == ord('s'):  # Presiona 's' para guardar anotaciones
                self.model.save_annotations()
                print("Annotations saved!")
                break

        cv2.destroyAllWindows()

class ImageAnnotatorController:
    def __init__(self, image_path, output_dir):
        self.model = ImageAnnotatorModel(image_path, output_dir)
        self.view = ImageAnnotatorView(self.model)

    def run(self):
        """Ejecutar la herramienta de anotación"""
        self.view.display_image()

if __name__ == "__main__":
    image_path = 'data/cropped_images/73_crop_88.jpg'
    output_dir = 'data/cropped_labels'
    os.makedirs(output_dir, exist_ok=True)

    annotator = ImageAnnotatorController(image_path, output_dir)
    annotator.run()