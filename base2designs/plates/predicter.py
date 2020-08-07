import cv2
import numpy as np
from base2designs.utils import label_map_util
import tensorflow as tf


num_classesArg = 37
modelArg = "datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb"
labelsArg = "datasets/records/classes.pbtxt"
# initialize the model
model = tf.Graph()
# create a context manager that makes this model the default one for
# execution
with model.as_default():
    # initialize the graph definition
    graphDef = tf.GraphDef()

    # load the graph from disk
    with tf.gfile.GFile(modelArg, "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")
labelMap = label_map_util.load_labelmap(labelsArg)
categories = label_map_util.convert_label_map_to_categories(
            labelMap, max_num_classes=num_classesArg,
            use_display_name=True)
# categoryIdx = label_map_util.create_category_index(categories)
# imageTensor = model.get_tensor_by_name("image_tensor:0")
# boxesTensor = model.get_tensor_by_name("detection_boxes:0")
# scoresTensor = model.get_tensor_by_name("detection_scores:0")
# classesTensor = model.get_tensor_by_name("detection_classes:0")
# numDetections = model.get_tensor_by_name("num_detections:0")


class Predicter():

    def __init__(self, model, sess, categoryIdx):
        # self.model = model
        self.sess = sess
        self.categoryIdx = categoryIdx
        self.imageTensor = model.get_tensor_by_name("image_tensor:0")
        self.boxesTensor = model.get_tensor_by_name("detection_boxes:0")
        self.scoresTensor = model.get_tensor_by_name("detection_scores:0")
        self.classesTensor = model.get_tensor_by_name("detection_classes:0")
        self.numDetections = model.get_tensor_by_name("num_detections:0")

    def scaleBoxes(self, boxes, hScale, wScale):
        boxes_scaled = []
        for box in boxes:
            box_scaled = (box[0] * hScale,
                          box[1] * wScale,
                          box[2] * hScale,
                          box[3] * wScale)
            boxes_scaled.append(np.array(box_scaled))
        return boxes_scaled

    def genSquareImage(self, image):
        # separate the channels
        blue = image[..., 0]
        green = image[..., 1]
        red = image[..., 2]

        # find the average pixel value for each channel
        blueAv = np.sum(blue) / blue.size
        greenAv = np.sum(green) / green.size
        redAv = np.sum(red) / red.size

        # create a square matrix large enough to fit the largest dimension of the input image, and fill with average pixels
        # save the scaling factors for H and W
        dimMax = max(image.shape[0], image.shape[1])
        hScale = dimMax / image.shape[0]
        wScale = dimMax / image.shape[1]
        imageOut = np.zeros((dimMax, dimMax, 3), dtype=np.uint8)
        imageOut[..., 0] = np.uint8(blueAv)
        imageOut[..., 1] = np.uint8(greenAv)
        imageOut[..., 2] = np.uint8(redAv)

        # copy the input image to the new square image
        if image.shape[0] <= image.shape[1]:
            imageOut[0:image.shape[0], ...] = image
        else:
            imageOut[:, 0:image.shape[1], ...] = image

        return imageOut, hScale, wScale

    # def predictChars(self, image, plateBox, image_display=False):
    #     # crop plate from the image, and predict chars
    #     H, W = image.shape[:2]
    #     (pbStartY, pbStartX, pbEndY, pbEndX) = (int(plateBox[0] * H),
    #                                             int(plateBox[1] * W),
    #                                             int(plateBox[2] * H),
    #                                             int(plateBox[3] * W))
    #     plateImage = image[pbStartY: pbEndY, pbStartX: pbEndX, ...]
    #     plateImage, hScale, wScale = self.genSquareImage(plateImage)
    #     if image_display:
    #         cv2.imshow("Plate Image", plateImage)
    #         cv2.waitKey(0)
    #     image_tf = cv2.cvtColor(plateImage.copy(), cv2.COLOR_BGR2RGB)
    #     image_tf = np.expand_dims(image_tf, axis=0)
    #
    #     (boxes, scores, labels, N) = self.sess.run(
    #         [self.boxesTensor, self.scoresTensor, self.classesTensor, self.numDetections],
    #         feed_dict={self.imageTensor: image_tf})
    #     # squeeze the lists into a single dimension
    #     boxes = np.squeeze(boxes)
    #     scores = np.squeeze(scores)
    #     labels = np.squeeze(labels)
    #
    #     # Adjust the box co-ordinates to account for the padding performed to square the image
    #     boxes = self.scaleBoxes(boxes, hScale, wScale)
    #
    #     return boxes, scores, labels

    def predictPlates(self, image, preprocess=True):

        if preprocess:
            image, hScale, wScale = self.genSquareImage(image)

        # prepare the image for inference input
        image_tf = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image_tf = np.expand_dims(image_tf, axis=0)

        # perform inference and compute the bounding boxes,
        # probabilities, and class labels
        # (boxes, scores, labels, N) = self.sess.run(
        #     [self.boxesTensor, self.scoresTensor, self.classesTensor, self.numDetections],
        #     feed_dict={self.imageTensor: image_tf})

        (boxes, scores, labels, N) = self.sess.run(
            [self.boxesTensor, self.scoresTensor, self.classesTensor, self.numDetections],
            feed_dict={self.imageTensor: image_tf})

        # squeeze the lists into a single dimension
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)

        # Adjust the box co-ordinates to account for the padding performed to square the image
        if preprocess:
            boxes = self.scaleBoxes(boxes, hScale, wScale)

        return boxes, scores, labels
