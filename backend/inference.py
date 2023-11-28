import torch
import os
import torchaudio
from neuralNet import CNNNetwork


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

def predict_single_file(model, audio_file_path, class_mapping, threshold=0.2):
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
    
    # Make an inference
    with torch.no_grad():
        predictions = model(input)
        probabilities = torch.sigmoid(predictions).squeeze(0)

        # Define predicted_labels based on a threshold
        predicted_labels = (probabilities > threshold).int()

        # Create a list of (probability, label) pairs
        prob_label_pairs = [(prob.item(), label) for prob, label in zip(probabilities, class_mapping)]

        # Sort the pairs based on probability in descending order
        sorted_prob_label_pairs = sorted(prob_label_pairs, reverse=True, key=lambda x: x[0])

        # Print the sorted probabilities and labels
        print("Sorted probabilities and labels:")
        for prob, label in sorted_prob_label_pairs:
            print(f"{label}: {prob:.4f}")

        # If you still need to use predicted_labels
        print(f"Predictions for file '{os.path.basename(audio_file_path)}':")
        for i, label in enumerate(predicted_labels):
            if label.item() == 1:
                print(f"  - {class_mapping[i]}")
        
    return predicted_labels

if __name__ == "__main__":
    # Define constants
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    class_mapping = [
        "cello",
        "clarinet",
        "flute",
        "acoustic guitar",
        "electric guitar",
        "piano",
        "saxophone",
        "trumpet",
        "violin",
        "human singing voice"
    ]

    # Load the model
    cnn = CNNNetwork()
    state_dict = torch.load("Nebula.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)
    cnn.eval()

    # Path to the audio file you want to predict
    audio_file_path = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\Drake.wav"

    # Make a prediction for the single audio file
    predicted_labels = predict_single_file(cnn, audio_file_path, class_mapping)
