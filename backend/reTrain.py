
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from musicDataSet import MusicDataSet
from neuralNet import CNNNetwork

# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model

loss_fn = nn.BCEWithLogitsLoss()
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1  # 10% of the data will be used for validation

ANNOTATIONS_FILE = r"C:\\Users\\bardi\\OneDrive\\Documents\\CST_Sem3\\Nebula\\Nebula\\dataset\\MultiTraining.csv"
AUDIO_DIR = r"C:\\Users\\bardi\\OneDrive\\Documents\\CST_Sem3\\Nebula\\Nebula\\dataset\\BIGDATA"
# sample rate of the audio files
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loaders(dataset, batch_size, validation_split):
    # Split the dataset into training and validation sets
    total_samples = len(dataset)
    validation_size = int(validation_split * total_samples)
    training_size = total_samples - validation_size

    train_set, val_set = random_split(dataset, [training_size, validation_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()

    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # Forward pass
        prediction = model(input)

        # Calculate loss
        loss = loss_fn(prediction, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Training Loss: {loss.item()}")

def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            # Forward pass
            prediction = model(input)

            # Calculate loss
            loss = loss_fn(prediction, target)

            total_loss += loss.item() * input.size(0)
            total_samples += input.size(0)

    average_loss = total_loss / total_samples
    print(f"Validation Loss: {average_loss}")
    return average_loss

def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_single_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

        print("---------------------------")

    print("Finished training")

if __name__ == "__main__":
    # choosing device, if cuda is available then use cuda else use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
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

    train_loader, val_loader = create_data_loaders(usd, BATCH_SIZE, VALIDATION_SPLIT)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # Load pre-trained model if available
    try:
        cnn.load_state_dict(torch.load(r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\Nebula.pth"))
        print("Loaded pre-trained model.")
    except FileNotFoundError:
        print("No pre-trained model found, starting from scratch.")

    # initialize loss function + optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_loader, val_loader, loss_fn, optimizer, device, EPOCHS)


    # save the final trained model
    torch.save(cnn.state_dict(), "Nebula.pth")
    print("Final trained model saved.")