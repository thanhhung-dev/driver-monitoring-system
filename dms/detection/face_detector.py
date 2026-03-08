import cv2
import onnxruntime
import numpy as np

class FaceDetector:

    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)

    def detect(self, frame):

        img = cv2.resize(frame,(640,640))
        img = img.astype(np.float32)

        img = img.transpose(2,0,1)
        img = np.expand_dims(img,axis=0)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img})

        return outputs