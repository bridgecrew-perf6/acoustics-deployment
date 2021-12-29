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
import librosa.display
import matplotlib.pyplot as plt

AUDIO_EXTENSIONS = ["mp4", "mp3", "aac", "m4a", "wav", "flac"] 

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
        :return: list of preprocessed model input data, audio in NDarray
        """
        # Take the input data and pre-process it make it inference ready
        if len(request) != 1:
            raise PredictionException("Batch input not accepted", 503)

        # Read the b64 str of the audio from the input
        body = request[0].get("body")
        try:
            # read b64 encoded audio file 
            json_str = body.decode('utf8').replace("'", '"')
            data = json.loads(json_str)
            audio_b64 = data['audio']
            ext = data['ext']
            if ext not in AUDIO_EXTENSIONS:
                raise ValueError(f"Got extension {ext} which is not part of {AUDIO_EXTENSIONS}")
    
            # save audio file
            audio_file = base64.b64decode(audio_b64)
            with open(f"/tmp/audio.{ext}", "wb+") as f:
                f.write(audio_file)

            # load as numpy arr
            audio_arr, _ = librosa.core.load(f"/tmp/audio.{ext}", sr=32000, mono=True)

            return torch.FloatTensor(audio_arr).reshape([1, -1]), audio_arr
        except Exception as e:
            raise PredictionException("error loading audio file: "+str(e), 500)

    def inference(self, model_input, audio_arr):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output
        """
        # Do some inference call to engine here and return output
        audio_arr.to(self.device)
        try:
            with torch.no_grad():
                self.model.eval()
                return self.model(model_input)['framewise_output'][0], audio_arr
        except Exception as e:
            raise PredictionException(str(e), 513)


    def postprocess(self, inference_output, audio_arr):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # wavplot
        fig_wav, ax = plt.subplots(figsize=(15,4), dpi=100)
        librosa.display.waveshow(audio_arr, sr=32000, ax=ax)
        ax.set_xlim(xmin=0, xmax=librosa.get_duration(audio_arr,sr=32000))
        fig_wav.savefig('/tmp/wav.png')
        plt.close()
        fig_mel, ax = plt.subplots(figsize=(15,4), dpi=100)
        stft = librosa.core.stft(y=audio_arr, n_fft=1024, hop_length=320, window='hann', center=True)
        ax.matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        fig_mel.savefig('/tmp/mel.png')
        plt.close()
        fig_pred, ax = plt.subplots(figsize=(15,4), dpi=100)
        inference_output = inference_output.cpu().numpy()
        inference_output = self.clean_output(inference_output)
        framewise_output = np.zeros_like(inference_output)
        framewise_output[np.arange(len(inference_output)), inference_output.argmax(1)] = 1
        ax.matshow(framewise_output.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        ax.yaxis.set_ticks(np.arange(0,3))
        ax.yaxis.set_ticklabels(np.array(['bp', 'start', 'stable']))
        ax.yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        fig_pred.savefig('/tmp/pred.png')
        plt.close()

        wav_b64 = base64.b64encode(open("/tmp/wav.png", "rb").read())
        mel_b64 = base64.b64encode(open("/tmp/mel.png", "rb").read())
        pred_b64 = base64.b64encode(open('/tmp/pred.png', "rb").read())
        
        return [{
            "wav" : wav_b64.decode('utf-8'),
            "mel_spec": mel_b64.decode('utf-8'),
            "pred": pred_b64.decode('utf-8')
            }]

    def clean_output(self, predicted_result):
        '''
        add weights to the result 
        '''
        is_bg, is_estart, is_estable = 0, 0, 0
        results = np.copy(predicted_result)

        if np.argmax(results[0]) == 0:
            is_bg = 1
            for idx, result in enumerate(results[1:]):
                predicted_class = np.argmax(result)
                if predicted_class == 1 and is_estart == 0:
    #                 print(f'found estart at index {idx}')
                    results[:idx+1][:, 0] = 1
                    results[:idx+1][:, 1] = 0
                    results[:idx+1][:, 2] = 0
                    is_estart = 1
                elif predicted_class == 0 and is_estart == 1:
    #                 print('found bg but estart already existed')
                    results[idx:][:, 2] = 1
                    results[idx:][:, 0] = 0
                    results[idx:][:, 1] = 0
                    is_estable = 1
                elif predicted_class == 2 and is_bg == 1 and is_estart == 0:
                    results[:idx+1][:, 0] = 1
                    results[:idx+1][:, 1] = 0
                    results[:idx+1][:, 2] = 0
    #                 print('stable changed to bg')
                    is_bg = 1
                elif predicted_class == 2 and is_estable == 0:
    #                 print(f'found estable at index{idx}')
                    results[idx:][:, 2] = 1
                    results[idx:][:, 0] = 0
                    results[idx:][:, 1] = 0
                    is_estable = 1
    #         print('is estart', is_estart)
            if is_estart == 0:
                results[:, 2] = 1
                results[:, 0] = 0
                results[:, 1] = 0
                
        else:
            results[:,2] = 1
            results[:, 0] = 0
            results[:, 1] = 0

        return results

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        model_input, audio_arr = self.preprocess(data)
        model_out, audio_arr = self.inference(model_input, audio_arr)
        return self.postprocess(model_out, audio_arr)


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
