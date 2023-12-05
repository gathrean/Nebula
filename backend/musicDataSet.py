import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class MusicDataSet(Dataset):

    #Annotations file path (csv)
    #Audio dir path (dir)
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        
        # the signal is registered to the device
        signal = signal.to(self.device)

        # the signal is a tensor that contains the audio samples
        # signal -> (num_channels, num_samples) -> (2, 16000) -> (1, 16000)
        
        #converting all audio sample rates to the target sample rate
        signal = self._resample_if_necessary(signal, sr)
        #take the signal and mix it down to mono
        signal = self._mix_down_if_necessary(signal)
        #number of samples per signal is more than expected then truncate
        signal = self._cut_if_necessary(signal)
        #number of samples per signal is less than expected then apply right padding
        signal = self._right_pad_if_necessary(signal)
        #passing the mel spectrogram and passing it the original signal 
        signal = self.transformation(signal)
        
        #convert label to numerical index
        label = torch.tensor(label).float()
        
        return signal, label
    
    def get_class_counts(self):
        # Assuming the labels are in the second column and onward in your annotations file
        label_columns = self.annotations.columns[1:]
        class_counts = {label: 0 for label in label_columns}

        for index in range(len(self.annotations)):
            labels = self._get_audio_sample_label(index)
            for label in label_columns:
                class_counts[label] += labels[label_columns.get_loc(label)]

        return class_counts
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1, 50000) -> (1, 22050)
        length_signal = signal.shape[1]
        if length_signal > self.num_samples:
            #Take the whole first dimension and leave it untouched until number of samples
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            #this is the number of missing samples we want to append
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) #0 is number to left pad other param is number for right pad
            # (0, 2) -> [1, 1, 1] -> [1, 1, 1, 0, 0] 
            # (1, num_samples) 
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    #converting all audio sample rates to the target sample rate
    def _resample_if_necessary(self, signal, sr):
        #resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            # Move resampler to the same device as the signal
            resampler = resampler.to(signal.device)
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
        return os.path.join(self.audio_dir, self.annotations.iloc[index, 0])



    
    #get the instrument name (the correct answer) from the annotations file
    def _get_audio_sample_label(self, index):
        # Assuming labels start from the 2nd column
        return self.annotations.iloc[index, 1:].tolist()


if __name__ == "__main__":
    ANNOTATIONS_FILE = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\SpotifyTrain.csv"
    AUDIO_DIR = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\Spotify"
    
    #sample rate of the audio files
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    
    #choosing device, if cuda is available then use cuda else use cpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    
    #mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    
    #it passes the path to the annotations file and the audio files
    usd = MusicDataSet(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                       SAMPLE_RATE, 
                       NUM_SAMPLES,
                       device)
    
    # After creating an instance of MusicDataSet, say 'usd', check its length
    print("Length of the dataset:", len(usd))

    # Access a valid index
    # Make sure 'index' is less than the length of the dataset
    index = 2  # for example
    if index < len(usd):
        signal, label = usd[index]
    else:
        print(f"Index {index} is out of bounds for the dataset.")

    