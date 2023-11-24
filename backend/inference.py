import torch
import torchaudio
from neuralNet import CNNNetwork
from musicDataSet import MusicDataSet
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES




class_mapping = [
    "Sound_Guitar",
    "Sound_Drum",
    "Sound_Violin",
    "Sound_Piano"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("Nebula.pth")
    cnn.load_state_dict(state_dict)

    # load music sound dataset validation dataset
    # instantiating our dataset object and create data loader
    #mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    #it passes the path to the annotations file and the audio files
    usd = MusicDataSet(ANNOTATIONS_FILE, 
                       AUDIO_DIR, 
                       mel_spectrogram,
                       SAMPLE_RATE, 
                       NUM_SAMPLES,
                       "cpu")

    # get a sample from the music dataset for inference
    input, target = usd[0][0], usd[0][1] # [batch_size, num_channels, freq, time]
    input.unsqueeze_(0) # [1, num_channels, freq, time]

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")