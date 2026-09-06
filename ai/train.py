import os
import sys
import re
import glob
import copy

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 48000

N_MFCC = 20
N_MELS = 40

EPOCHS = 500
LEARNING_RATE = 0.0005

BASE_DIR = "/Users/facebook/Anukrama/ai"
VERSE = "v02"
REFERENCE_DIR = os.path.join(BASE_DIR, "data", "ch15", VERSE)
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")
MODEL_PATH = os.path.join(MODEL_DIR, f"anukrama_{VERSE}_score.pt")
SCALER_PATH = os.path.join(SCALER_DIR, f"anukrama_{VERSE}_scaler.npz")

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def load_pcm(path):
    audio = np.fromfile(
        path,
        dtype=np.int16
    ).astype(np.float32)

    if len(audio) == 0:
        raise ValueError(
            f"Empty audio file: {path}"
        )

    audio /= 32768.0

    return audio

def extract_features(audio):
    audio = audio - np.mean(audio)

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = audio / peak

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512,
        n_mels=N_MELS
    )

    delta = librosa.feature.delta(
        mfcc
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512
    )

    zero_crossing = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=2048,
        hop_length=512
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )

    features = np.vstack([
        mfcc,
        delta,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing,
        rms
    ])

    mean = np.mean(
        features,
        axis=1,
        keepdims=True
    )

    std = np.std(
        features,
        axis=1,
        keepdims=True
    )

    std[std < 1e-6] = 1.0

    features = (
        features - mean
    ) / std

    return features.astype(
        np.float32
    )

def dtw_distance(
    reference,
    recording
):
    D, wp = librosa.sequence.dtw(
        X=reference,
        Y=recording,
        metric="euclidean"
    )

    distance = D[-1, -1]

    if len(wp) > 0:
        distance /= len(wp)

    return float(distance)

def compare_features(
    reference,
    recording
):
    distance = dtw_distance(
        reference,
        recording
    )

    ref_mean = np.mean(
        reference,
        axis=1
    )

    rec_mean = np.mean(
        recording,
        axis=1
    )

    ref_std = np.std(
        reference,
        axis=1
    )

    rec_std = np.std(
        recording,
        axis=1
    )

    mean_difference = np.mean(
        np.abs(
            ref_mean - rec_mean
        )
    )

    std_difference = np.mean(
        np.abs(
            ref_std - rec_std
        )
    )

    ref_frames = reference.shape[1]
    rec_frames = recording.shape[1]

    duration_ratio = (
        rec_frames /
        max(ref_frames, 1)
    )

    duration_difference = abs(
        1.0 - duration_ratio
    )

    return np.array([
        distance,
        mean_difference,
        std_difference,
        duration_difference
    ], dtype=np.float32)

def load_references():
    pattern = os.path.join(
        REFERENCE_DIR,
        "good",
        "*.pcm"
    )

    files = sorted(
        glob.glob(pattern)
    )

    if not files:
        raise RuntimeError(
            f"No reference recordings "
            f"found in {REFERENCE_DIR}"
        )

    references = []

    print()
    print(
        "Loading good reference recordings..."
    )

    for path in files:
        filename = os.path.basename(
            path
        )

        audio = load_pcm(
            path
        )

        features = extract_features(
            audio
        )

        references.append(
            features
        )

        print(
            f"  {filename:20s}"
            f" {len(audio) / SAMPLE_RATE:.2f}s"
            f" {features.shape}"
        )

    return references

def load_labeled_recordings():
    pattern = os.path.join(
        REFERENCE_DIR,
        "labeled",
        "bad_*.pcm"
    )

    files = sorted(
        glob.glob(pattern)
    )

    if not files:
        raise RuntimeError(
            f"No bad_XX.pcm recordings "
            f"found in {REFERENCE_DIR}"
        )

    recordings = []

    print()
    print(
        "Loading labeled recordings..."
    )

    for path in files:
        filename = os.path.basename(
            path
        )

        match = re.match(
            r"bad_(\d+(?:\.\d+)?)\.pcm$",
            filename
        )

        if not match:
            print(
                f"Skipping {filename}: "
                f"invalid filename"
            )
            continue

        score = float(
            match.group(1)
        )

        if score < 0 or score > 100:
            print(
                f"Skipping {filename}: "
                f"score must be 0-100"
            )
            continue

        audio = load_pcm(
            path
        )

        features = extract_features(
            audio
        )

        recordings.append(
            (
                filename,
                features,
                score
            )
        )

        print(
            f"  {filename:20s}"
            f" {len(audio) / SAMPLE_RATE:.2f}s"
            f" target={score:.1f}"
        )

    if not recordings:
        raise RuntimeError(
            "No valid bad_XX.pcm recordings found."
        )

    return recordings

class PronunciationDataset(
    Dataset
):
    def __init__(
        self,
        X,
        y
    ):
        self.X = torch.tensor(
            np.asarray(X),
            dtype=torch.float32
        )

        self.y = torch.tensor(
            np.asarray(y),
            dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(
            self.X
        )

    def __getitem__(
        self,
        index
    ):
        return (
            self.X[index],
            self.y[index]
        )

class PronunciationModel(
    nn.Module
):
    def __init__(
        self,
        input_size
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            )
        )

    def forward(
        self,
        x
    ):
        return self.network(x)

def fit_scaler(X):
    X = np.asarray(
        X,
        dtype=np.float32
    )

    mean = np.mean(
        X,
        axis=0
    )

    std = np.std(
        X,
        axis=0
    )

    std[std < 1e-6] = 1.0

    return mean, std

def apply_scaler(
    X,
    mean,
    std
):
    X = np.asarray(
        X,
        dtype=np.float32
    )

    return (
        (X - mean) /
        std
    ).astype(
        np.float32
    )

def build_dataset(
    references,
    recordings
):
    X = []
    y = []

    print()
    print(
        "Building comparison features..."
    )

    for (
        filename,
        recording,
        score
    ) in recordings:

        comparisons = []

        for reference in references:
            features = compare_features(
                reference,
                recording
            )

            comparisons.append(
                features
            )

        comparisons = np.asarray(
            comparisons,
            dtype=np.float32
        )

        mean_features = np.mean(
            comparisons,
            axis=0
        )

        min_features = np.min(
            comparisons,
            axis=0
        )

        max_features = np.max(
            comparisons,
            axis=0
        )

        median_features = np.median(
            comparisons,
            axis=0
        )

        final_features = np.concatenate([
            mean_features,
            min_features,
            max_features,
            median_features
        ])

        X.append(
            final_features
        )

        y.append(
            score
        )

        print(
            f"  {filename:20s}"
            f" target={score:5.1f}"
        )

    return X, y

def train_model(
    X,
    y
):
    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    scaler_mean, scaler_std = fit_scaler(
        X
    )

    X_scaled = apply_scaler(
        X,
        scaler_mean,
        scaler_std
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    np.savez(
        SCALER_PATH,
        mean=scaler_mean,
        std=scaler_std
    )

    dataset = PronunciationDataset(
        X_scaled,
        y
    )

    loader = DataLoader(
        dataset,
        batch_size=min(
            8,
            len(dataset)
        ),
        shuffle=True
    )

    model = PronunciationModel(
        input_size=X.shape[1]
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    loss_function = nn.MSELoss()

    best_loss = float(
        "inf"
    )

    best_state = None

    print()
    print(
        "Using device:",
        DEVICE
    )

    print(
        "Input features:",
        X.shape[1]
    )

    print()
    print(
        "Starting training..."
    )

    print()

    for epoch in range(
        EPOCHS
    ):
        model.train()

        total_loss = 0.0

        for inputs, targets in loader:
            inputs = inputs.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            optimizer.zero_grad()

            predictions = model(
                inputs
            )

            loss = loss_function(
                predictions,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(loader)
        )

        if average_loss < best_loss:
            best_loss = average_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

        if (
            epoch < 10
            or (epoch + 1) % 10 == 0
        ):
            print(
                f"Epoch "
                f"{epoch + 1:03d}/{EPOCHS}"
                f" | Loss: "
                f"{average_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    best_model_path = MODEL_PATH.replace(
        ".pt",
        "_best.pt"
    )

    torch.save(
        model.state_dict(),
        best_model_path
    )

    print()
    print(
        "Training complete."
    )

    print(
        f"Best loss: "
        f"{best_loss:.4f}"
    )

    print(
        f"Saved: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved: "
        f"{best_model_path}"
    )

    print(
        f"Saved scaler: "
        f"{SCALER_PATH}"
    )

    return model

def load_scaler():
    if not os.path.exists(
        SCALER_PATH
    ):
        raise RuntimeError(
            f"Scaler not found: "
            f"{SCALER_PATH}\n"
            f"Retrain the model first."
        )

    data = np.load(
        SCALER_PATH
    )

    return (
        data["mean"],
        data["std"]
    )

def score_recording(
    model,
    references,
    path
):
    print()
    print(
        "Anukrama Pronunciation Test"
    )

    print(
        "==================================="
    )

    print(
        f"Loading test recording: "
        f"{path}"
    )

    audio = load_pcm(
        path
    )

    print(
        f"Duration: "
        f"{len(audio) / SAMPLE_RATE:.2f}s"
    )

    print()
    print(
        "Extracting features..."
    )

    recording = extract_features(
        audio
    )

    comparisons = []

    for reference in references:
        features = compare_features(
            reference,
            recording
        )

        comparisons.append(
            features
        )

    comparisons = np.asarray(
        comparisons,
        dtype=np.float32
    )

    mean_features = np.mean(
        comparisons,
        axis=0
    )

    min_features = np.min(
        comparisons,
        axis=0
    )

    max_features = np.max(
        comparisons,
        axis=0
    )

    median_features = np.median(
        comparisons,
        axis=0
    )

    final_features = np.concatenate([
        mean_features,
        min_features,
        max_features,
        median_features
    ])

    scaler_mean, scaler_std = load_scaler()

    final_features = apply_scaler(
        final_features,
        scaler_mean,
        scaler_std
    )

    x = torch.tensor(
        final_features,
        dtype=torch.float32
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():
        prediction = model(
            x
        )

    score = float(
        prediction.item()
    )

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    print()
    print(
        "Comparison:"
    )

    print(
        f"  Mean DTW: "
        f"{mean_features[0]:.4f}"
    )

    print(
        f"  Best DTW: "
        f"{min_features[0]:.4f}"
    )

    print(
        f"  Worst DTW: "
        f"{max_features[0]:.4f}"
    )

    print()
    print(
        "==================================="
    )

    print(
        "RESULT"
    )

    print(
        "==================================="
    )

    print(
        f"Pronunciation score: "
        f"{score:.1f}/100"
    )

    if score >= 90:
        print(
            "Excellent pronunciation."
        )

    elif score >= 75:
        print(
            "Good pronunciation."
        )

    elif score >= 50:
        print(
            "Some pronunciation differences detected."
        )

    elif score >= 25:
        print(
            "Significant pronunciation differences detected."
        )

    else:
        print(
            "Major pronunciation differences detected."
        )

    return score

def main():
    if len(sys.argv) >= 3:
        test_path = sys.argv[2]

        if not os.path.exists(
            test_path
        ):
            raise RuntimeError(
                f"File not found: "
                f"{test_path}"
            )

        references = load_references()

        best_model_path = MODEL_PATH.replace(
            ".pt",
            "_best.pt"
        )

        if not os.path.exists(
            best_model_path
        ):
            raise RuntimeError(
                "No trained model found.\n"
                "Run:\n"
                "python3 train.py\n"
                "first."
            )

        scaler_mean, scaler_std = load_scaler()

        input_size = len(
            scaler_mean
        )

        model = PronunciationModel(
            input_size=input_size
        ).to(
            DEVICE
        )

        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=DEVICE
            )
        )

        score_recording(
            model,
            references,
            test_path
        )

        return

    print()

    print(
        f"Anukrama {VERSE} "
        "Pronunciation Model"
    )

    print(
        "==================================="
    )

    if not os.path.isdir(
        REFERENCE_DIR
    ):
        raise RuntimeError(
            f"Folder not found: "
            f"{REFERENCE_DIR}"
        )

    references = load_references()

    recordings = load_labeled_recordings()

    print()
    print(
        f"Good references: "
        f"{len(references)}"
    )

    print(
        f"Labeled recordings: "
        f"{len(recordings)}"
    )

    X, y = build_dataset(
        references,
        recordings
    )

    print()
    print(
        "Training examples:"
    )

    for target in y:
        print(
            f"  target = "
            f"{target:.1f}"
        )

    train_model(
        X,
        y
    )

if __name__ == "__main__":
    main()