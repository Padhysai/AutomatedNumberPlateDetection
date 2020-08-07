from PIL import Image
import cv2
import tensorflow as tf
from base2designs.utils import label_map_util
from base2designs.plates.plateFinder import PlateFinder
from base2designs.plates.predicter import Predicter
from base2designs.plates.plateDisplay import PlateDisplay
from PIL import ImageEnhance

class DetectVehicleNumberPlate:
    def __init__(self):
        # self.modelArg = "datasets/experiment_faster_rcnn/2018_08_02/exported_model/frozen_inference_graph.pb"
        self.modelArg = "datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb"
        self.labelsArg = "datasets/records/classes.pbtxt"
        self.num_classesArg = 37
        self.min_confidenceArg = 0.5

        # initialize the model
        self.model = tf.Graph()
        # create a context manager that makes this model the default one for
        # execution
        with self.model.as_default():
            # initialize the graph definition
            self.graphDef = tf.GraphDef()

            # load the graph from disk
            with tf.gfile.GFile(self.modelArg, "rb") as f:
                self.serializedGraph = f.read()
                self.graphDef.ParseFromString(self.serializedGraph)
                tf.import_graph_def(self.graphDef, name="")

        # load the class labels from disk
        self.labelMap = label_map_util.load_labelmap(self.labelsArg)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.labelMap, max_num_classes=self.num_classesArg,
            use_display_name=True)
        self.categoryIdx = label_map_util.create_category_index(self.categories)

        # create a plateFinder
        self.plateFinder = PlateFinder(self.min_confidenceArg, self.categoryIdx,
                                       rejectPlates=False, charIOUMax=0.3)

        # create plate displayer
        self.plateDisplay = PlateDisplay()

    def predictImages(self, imagePathArg, pred_stagesArg, croppedImagepath, numPlateOrg):
        # create a session to perform inference
        with numPlateOrg.model.as_default():
            with tf.Session(graph=numPlateOrg.model) as sess:
                # create a predicter, used to predict plates and chars
                predicter = Predicter(numPlateOrg.model, sess, numPlateOrg.categoryIdx)
                # load the image from disk
                # print("[INFO] Loading image \"{}\"".format(imagePaths))
                image = cv2.imread(imagePathArg)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image = ImageEnhance.Sharpness(imagePathArg)
                # If prediction stages == 2, then perform prediction on full image, find the plates, crop the plates from the image,
                # and then perform prediction on the plate images

                if pred_stagesArg == 2:
                    # Perform inference on the full image, and then select only the plate boxes
                    boxes, scores, labels = predicter.predictPlates(image, preprocess=False)
                    licensePlateFound_pred, plateBoxes_pred, plateScores_pred = self.plateFinder.findPlatesOnly(
                        boxes,
                        scores,
                        labels)
                    imageLabelled = self.getBoundingBox(image, plateBoxes_pred, imagePathArg, croppedImagepath)

                else:
                    print("[ERROR] --pred_stages {}. The number of prediction stages must be either 1 or 2".format(
                        pred_stagesArg))
                    quit()

                return imageLabelled

    def getBoundingBox(self, image, plateBoxes, imagePath, croppedImagepath):
        (H, W) = image.shape[:2]

        for plateBox in plateBoxes:
            # Draw the plate box rectangle in red
            # scale the bounding box from the range [0, 1] to [W, H]
            (startY, startX, endY, endX) = plateBox
            startX = int(startX * W)
            startY = int(startY * H)
            endX = int(endX * W)
            endY = int(endY * H)
            # croppedimage = crop(imagePath, (startX, startY, endX, endY), croppedImagepath)

            try:
                image_obj = Image.open(imagePath)

                cropped_image = image_obj.crop((startX, startY, endX, endY))
                cropped_image = cropped_image.convert("L")
                # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cropped_image.save(croppedImagepath)
                return cropped_image
            except Exception as e:
                print(e)
