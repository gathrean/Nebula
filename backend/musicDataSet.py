import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class MusicDataSet(Dataset):

    #Annotations file path (csv)
    #Audio dir path (dir)
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        #Setting the paths for the annotations and the audio files
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation #transformation is the mel spectrogram
        self.target_sample_rate = target_sample_rate

    #get the length of the dataset
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        #getting the path to the audio sample
        audio_sample_path = self._get_audio_sample_path(index)

        #get the label of the audio sample
        label = self._get_audio_sample_label(index)
        #returns the signal and the sample rate
        try:
            signal, sr = torchaudio.load(audio_sample_path)
        except Exception as e:
            print(f"Error loading audio file {audio_sample_path}: {e}")
            return None, None

        
        # the signal is a tensor that contains the audio samples
        # signal -> (num_channels, num_samples) -> (2, 16000) -> (1, 16000)
        
        #converting all audio sample rates to the target sample rate
        signal = self._resample_if_necessary(signal, sr)
        #take the signal and mix it down to mono
        signal = self._mix_down_if_necessary(signal)
        
        
        #passing the mel spectrogram and passing it the original signal 
        signal = self.transformation(signal)
        return signal, label

    #converting all audio sample rates to the target sample rate
    def _resample_if_necessary(self, signal, sr):
        #resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    #take the signal and mix it down to mono
    def _mix_down_if_necessary(self, signal):
        #mix down if necessary
        #only if there is more than one channel
        if signal.shape[0] > 1: # (2, 16000) -> (1, 16000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    #get the path to the audio file
    def _get_audio_sample_path(self, index):
        # Use forward slashes consistently for file paths
        file_name = self.annotations.iloc[index, 0].replace("\\", "/")
        path = os.path.join(self.audio_dir, file_name).replace("\\", "/")
        return path



    
    #get the instrument name (the correct answer) from the annotations file
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]


if __name__ == "__main__":
    ANNOTATIONS_FILE = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\archive\Metadata_Test.csv"
    AUDIO_DIR = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\archive\Test_submission\Test_submission"
    
    #sample rate of the audio files
    SAMPLE_RATE = 16000
    
    #mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    
    #it passes the path to the annotations file and the audio files
    usd = MusicDataSet(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                       SAMPLE_RATE)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]