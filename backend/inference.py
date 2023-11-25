import torch
import os
import torchaudio
from neuralNet import CNNNetwork
from train import SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "Sound_Guitar",
    "Sound_Drum",
    "Sound_Violin",
    "Sound_Piano"
]

def resample_if_necessary(signal, sr, target_sample_rate):
    # Resample if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def mix_down_if_necessary(signal):
    # Mix down if necessary (only if there is more than one channel)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def cut_if_necessary(signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal > num_samples:
        signal = signal[:, :num_samples]
    return signal

def right_pad_if_necessary(signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def predict_single_file(model, audio_file_path, class_mapping):
    model.eval()
    
    # Load the audio file
    try:
        signal, sr = torchaudio.load(audio_file_path)
    except Exception as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return
    
    # Apply the transformations
    target_sample_rate = SAMPLE_RATE  # Modify this if needed
    signal = signal.to("cpu")
    signal = resample_if_necessary(signal, sr, target_sample_rate)
    signal = mix_down_if_necessary(signal)
    signal = cut_if_necessary(signal, NUM_SAMPLES)
    signal = right_pad_if_necessary(signal, NUM_SAMPLES)
    
    # Apply mel spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    signal = mel_spectrogram(signal)
    
    # Prepare the input tensor
    input = signal.unsqueeze(0)  # [1, num_channels, freq, time]
    
    # Make an inference and print the prediction
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        
    file_name = os.path.basename(audio_file_path)  # Extract the file name
    print(f"Predicted for file '{file_name}': '{predicted}'")
    return predicted

if __name__ == "__main__":
    # Load the model
    cnn = CNNNetwork()
    state_dict = torch.load("Nebula.pth")
    cnn.load_state_dict(state_dict)

    # Path to the audio file you want to predict
    audio_file_path = "C:/Users/bardi/OneDrive/Documents/CST_Sem3/Nebula/Nebula/dataset/archive/Test_submission/Test_submission/darbuka-drum-percussion-64018.wav"

    # Make a prediction for the single audio file
    predicted = predict_single_file(cnn, audio_file_path, class_mapping)
