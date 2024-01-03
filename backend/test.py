import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from musicDataSet import MusicDataSet
from neuralNet import CNNNetwork
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

# Constants
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.5  # 10% of the data will be used for validation

ANNOTATIONS_FILE = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\FinalData.csv"
AUDIO_DIR = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\Spotify"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 * 10

def create_data_loaders(dataset, batch_size, validation_split):
    total_samples = len(dataset)
    validation_size = int(validation_split * total_samples)
    training_size = total_samples - validation_size

    train_set, val_set = random_split(dataset, [training_size, validation_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            # If targets are one-hot encoded, convert to class indices
            if len(target.shape) > 1:
                target = torch.argmax(target, dim=1)

            prediction = model(input)
            loss = loss_fn(prediction, target)

            total_loss += loss.item() * input.size(0)
            total_samples += input.size(0)

            # Save predictions and true labels
            all_predictions.extend(torch.argmax(prediction, dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    average_loss = total_loss / total_samples
    print(f"Validation Loss: {average_loss}")
    return average_loss, np.array(all_predictions), np.array(all_targets)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = MusicDataSet(ANNOTATIONS_FILE,
                       AUDIO_DIR,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device)

    _, val_loader = create_data_loaders(usd, BATCH_SIZE, VALIDATION_SPLIT)

    # Load your trained model
    cnn = CNNNetwork()
    cnn.load_state_dict(torch.load(r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\Best.pth"))
    cnn.to(device)

    # Use CrossEntropyLoss for evaluation (without class weights)
    loss_fn = nn.CrossEntropyLoss()

    # Perform validation (testing)
    val_loss, val_predictions, val_targets = validate(cnn, val_loader, loss_fn, device)

    # Compute confusion matrix and F1 score
    confusion_mat = confusion_matrix(val_targets, val_predictions)
    f1 = f1_score(val_targets, val_predictions, average='weighted')

    print(f"Confusion Matrix:\n{confusion_mat}")
    print(f"F1 Score: {f1}")
    print(f"Validation Loss: {val_loss}")