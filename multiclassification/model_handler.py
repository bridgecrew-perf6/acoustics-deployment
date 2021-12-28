"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
import os
import sys
import glob
from mms.service import PredictionException

import json
import base64
import numpy as np
import torch
import librosa


class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.model = None
        self.device = None


    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # import model code
        if os.path.exists(os.path.join(model_dir, "code/")):
            sys.path.append(os.path.join(model_dir, "code/"))
        self.model = self.model_fn(model_dir)

    def model_fn(self, model_dir):
        """Loads a model. For PyTorch, a default function to load a model only if Elastic Inference is used.
        In other cases, users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.

        Returns: A PyTorch model.
        """
        model_paths = glob.glob(os.path.join(model_dir, "**/*.pth"), recursive=True)
        if len(model_paths) != 1:
            raise ValueError(
                "Exactly one .pth or .pt file is required for PyTorch models: {}".format(model_paths)
                )
        model_path = model_paths[0]
        try:
            model = torch.load(model_path, map_location=self.device)
        except RuntimeError:
            raise MemoryError
        model = model.to(self.device)
        return model

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        if len(request) != 1:
            raise PredictionException("Batch input not accepted", 503)

        # Read the b64 str of the audio from the input
        body = request[0].get("body")
        try:
            # read b64 encoded audio file 
            json_str = body.decode('utf8').replace("'", '"')
            audio_b64 = json.loads(json_str)['audio']
    
            # save audio file
            audio_file = base64.b64decode(audio_b64)
            with open("/tmp/audio.m4a", "wb+") as f:
                f.write(audio_file)

            # load as numpy arr
            audio_arr, _ = librosa.core.load("/tmp/audio.m4a", sr=32000, mono=True)

            return torch.FloatTensor(audio_arr).reshape([1, -1])
        except Exception as e:
            raise PredictionException("error loading audio file: "+str(e), 500)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output
        """
        # Do some inference call to engine here and return output
        try:
            with torch.no_grad():
                self.model.eval()
                return self.model(model_input)['framewise_output'][0]
        except Exception as e:
            raise PredictionException(str(e), 513)


    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        
        return [inference_output.numpy().tolist()]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
