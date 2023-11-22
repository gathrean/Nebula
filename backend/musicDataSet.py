import os

from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class MusicDataSet(Dataset):

    #Annotations file path (csv)
    #Audio dir path (dir)
    def __init__(self, annotations_file, audio_dir):
        #Setting the paths for the annotations and the audio files
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    #get the length of the dataset
    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        #getting the path to the audio sample
        audio_sample_path = self._get_audio_sample_path(index)
        #get the label of the audio sample
        label = self._get_audio_sample_label(index)
        #returns the signal and the sample rate
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    #get the path to the audio file
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    
    #get the instrument name (the correct answer) from the annotations file
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:/Users/bardi/OneDrive/Documents/CST_Sem3/Nebula/Nebula/dataset/archive/Metadata_Test.csv"
    AUDIO_DIR = "C:/Users/bardi/OneDrive/Documents/CST_Sem3/Nebula/Nebula/dataset/archive/Test_submission"
    #it passes the path to the annotations file and the audio files
    usd = MusicDataSet(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
