import sys
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

def predict_full_song(model, audio_file_path, class_mapping, segment_length):
    model.eval()
    
    # Load the full song
    try:
        signal, sr = torchaudio.load(audio_file_path)
        signal = signal.to("cpu")
    except Exception as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return
    
    # Resample and mix down the song
    target_sample_rate = SAMPLE_RATE
    signal = resample_if_necessary(signal, sr, target_sample_rate)
    signal = mix_down_if_necessary(signal)
    
    # Split the song into 1-second segments
    total_samples = signal.shape[1]
    segments = torch.split(signal, segment_length, dim=1)
    aggregated_predictions = []

    # Process each segment
    for segment in segments:
        segment = right_pad_if_necessary(segment, segment_length)
        
        # Apply mel spectrogram transformation
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        input_segment = mel_spectrogram(segment).unsqueeze(0)  # [1, num_channels, freq, time]
        
        # Make an inference
        with torch.no_grad():
            predictions = model(input_segment)
            probabilities = torch.sigmoid(predictions).squeeze(0)
            aggregated_predictions.append(probabilities)

    # Aggregate predictions over all segments
    # Example: averaging the predictions
    average_predictions = torch.mean(torch.stack(aggregated_predictions), dim=0)

    # Calculate mean and standard deviation
    mean = torch.mean(average_predictions).item()
    std_dev = torch.std(average_predictions).item()

    # Determine the cutoff (mean + standard deviation)
    criteria = 0.5
    cutoff = mean + std_dev * criteria

    # Pair each label with its average prediction and sort by probability
    sorted_predictions = sorted(zip(class_mapping, average_predictions), key=lambda x: x[1], reverse=True)

    # Filter based on the cutoff
    filtered_predictions = [(label, prob.item()) for label, prob in sorted_predictions if prob.item() >= cutoff]

    # Generate final output
    print("Filtered predictions for the full song (sorted by accuracy):")
    for label, prob in filtered_predictions:
        print(f"{label}: {prob:.4f}")

    return filtered_predictions


if __name__ == "__main__":
    # Define constants
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050 * 5
    class_mapping = [
        "bass",
        "cello",
        "clarinet",
        "drums",
        "flute",
        "guitar",
        "piano",
        "trumpet",
        "violin",
        "voice"
    ]

    # Load the model
    cnn = CNNNetwork()
    state_dict = torch.load("Nebula.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)
    cnn.eval()

    # Path to the audio file you want to predict
    audio_file_path = sys.argv[1]

    # Make a prediction for the single audio file
    predicted_labels = predict_full_song(cnn, audio_file_path, class_mapping, NUM_SAMPLES)
