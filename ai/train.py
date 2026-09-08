import os
import sys
import re
import glob
import copy
import random

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

SAMPLE_RATE = 48000

N_MFCC = 20
N_MELS = 40

EPOCHS = 300
LEARNING_RATE = 0.0005
BATCH_SIZE = 8

PATIENCE = 30
LR_PATIENCE = 10
MIN_LR = 0.000001

VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ch15"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

SCALER_DIR = os.path.join(
    BASE_DIR,
    "scalers"
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def verse_dir(verse):
    return os.path.join(
        DATA_DIR,
        verse
    )


def model_path(verse):
    return os.path.join(
        MODEL_DIR,
        f"anukrama_{verse}_score.pt"
    )


def best_model_path(verse):
    return os.path.join(
        MODEL_DIR,
        f"anukrama_{verse}_score_best.pt"
    )


def scaler_path(verse):
    return os.path.join(
        SCALER_DIR,
        f"anukrama_{verse}_scaler.npz"
    )


def discover_verses():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(
            f"Data directory not found:\n{DATA_DIR}"
        )

    folders = []

    for path in glob.glob(
        os.path.join(DATA_DIR, "*")
    ):
        if not os.path.isdir(path):
            continue

        name = os.path.basename(path)

        if re.match(r"^v\d+$", name):
            folders.append(name)

    folders.sort(
        key=lambda x: int(x[1:])
    )

    return folders


def load_pcm(path):
    audio = np.fromfile(
        path,
        dtype=np.int16
    ).astype(np.float32)

    if len(audio) == 0:
        raise ValueError(
            f"Empty audio file:\n{path}"
        )

    audio /= 32768.0

    return audio


def is_valid_speech_sample(
    audio,
    min_rms=0.0025,
    min_peak=0.02,
    min_active_ratio=0.025,
    min_duration=0.3
):
    if len(audio) == 0:
        return False

    if len(audio) / SAMPLE_RATE < min_duration:
        return False

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(np.square(audio)))
    active = np.mean(np.abs(audio) > 0.01)

    return bool(
        peak >= min_peak and
        rms >= min_rms and
        active >= min_active_ratio
    )


def extract_features(audio):
    audio = audio.astype(
        np.float32
    )

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


def build_comparison_features(
    references,
    recording
):
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

    return np.concatenate([
        mean_features,
        min_features,
        max_features,
        median_features
    ]).astype(np.float32)


def load_references(verse):
    reference_dir = os.path.join(
        verse_dir(verse),
        "good"
    )

    files = sorted(
        glob.glob(
            os.path.join(
                reference_dir,
                "*.pcm"
            )
        )
    )

    if not files:
        raise RuntimeError(
            f"No good reference recordings found for {verse}:\n"
            f"{reference_dir}"
        )

    references = []

    print()
    print(
        f"Loading good references for {verse}..."
    )

    for path in files:
        filename = os.path.basename(path)

        audio = load_pcm(path)

        if not is_valid_speech_sample(audio):
            print(
                f"  Skipping {filename}: no clear speech signal"
            )
            continue

        features = extract_features(
            audio
        )

        references.append(
            features
        )

        print(
            f"  {filename:28s}"
            f" {len(audio) / SAMPLE_RATE:.2f}s"
            f" {features.shape}"
        )

    return references


def load_labeled_recordings(verse):
    labeled_dir = os.path.join(
        verse_dir(verse),
        "labeled"
    )

    files = sorted(
        glob.glob(
            os.path.join(
                labeled_dir,
                "bad_*.pcm"
            )
        )
    )

    if not files:
        raise RuntimeError(
            f"No bad_XX.pcm recordings found for {verse}:\n"
            f"{labeled_dir}"
        )

    recordings = []

    print()
    print(
        f"Loading labeled recordings for {verse}..."
    )

    for path in files:
        filename = os.path.basename(path)

        match = re.match(
            r"bad_(\d+(?:\.\d+)?)\.pcm$",
            filename
        )

        if not match:
            print(
                f"Skipping {filename}: invalid filename"
            )
            continue

        score = float(
            match.group(1)
        )

        if score < 0 or score > 100:
            print(
                f"Skipping {filename}: score must be 0-100"
            )
            continue

        audio = load_pcm(path)

        if not is_valid_speech_sample(audio):
            print(
                f"  Skipping {filename}: no clear speech signal"
            )
            continue

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
            f"  {filename:28s}"
            f" {len(audio) / SAMPLE_RATE:.2f}s"
            f" target={score:.1f}"
        )

    if not recordings:
        raise RuntimeError(
            f"No valid labeled recordings found for {verse}."
        )

    return recordings


class PronunciationDataset(Dataset):
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
        return len(self.X)

    def __getitem__(
        self,
        index
    ):
        return (
            self.X[index],
            self.y[index]
        )


class PronunciationModel(nn.Module):
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
            nn.Dropout(0.15),
            nn.Linear(
                64,
                32
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
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

        final_features = build_comparison_features(
            references,
            recording
        )

        X.append(
            final_features
        )

        y.append(
            score
        )

        print(
            f"  {filename:28s}"
            f" target={score:5.1f}"
        )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.float32
        )
    )


def train_model(
    verse,
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

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        SCALER_DIR,
        exist_ok=True
    )

    np.savez(
        scaler_path(verse),
        mean=scaler_mean,
        std=scaler_std
    )

    dataset = PronunciationDataset(
        X_scaled,
        y
    )

    use_validation = len(dataset) >= 5

    if use_validation:
        validation_size = max(
            1,
            int(
                len(dataset) *
                VALIDATION_SPLIT
            )
        )

        training_size = (
            len(dataset) -
            validation_size
        )

        if training_size < 2:
            use_validation = False

    if use_validation:
        generator = torch.Generator().manual_seed(
            RANDOM_SEED
        )

        train_dataset, validation_dataset = random_split(
            dataset,
            [
                training_size,
                validation_size
            ],
            generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(
                BATCH_SIZE,
                len(train_dataset)
            ),
            shuffle=True
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=min(
                BATCH_SIZE,
                len(validation_dataset)
            ),
            shuffle=False
        )

    else:
        train_loader = DataLoader(
            dataset,
            batch_size=min(
                BATCH_SIZE,
                len(dataset)
            ),
            shuffle=True
        )

        validation_loader = None

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

    loss_function = nn.SmoothL1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=MIN_LR
    )

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print()
    print(
        "==================================="
    )
    print(
        f"Training {verse}"
    )
    print(
        "==================================="
    )
    print(
        "Device:",
        DEVICE
    )
    print(
        "Training examples:",
        len(dataset)
    )

    if use_validation:
        print(
            "Validation examples:",
            len(validation_dataset)
        )

    print(
        "Input features:",
        X.shape[1]
    )

    print()

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(inputs)

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

            total_train_loss += loss.item()

        train_loss = (
            total_train_loss /
            len(train_loader)
        )

        if validation_loader is not None:
            model.eval()

            total_validation_loss = 0.0

            with torch.no_grad():
                for inputs, targets in validation_loader:
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)

                    predictions = model(inputs)

                    loss = loss_function(
                        predictions,
                        targets
                    )

                    total_validation_loss += loss.item()

            validation_loss = (
                total_validation_loss /
                len(validation_loader)
            )

            monitor_loss = validation_loss

        else:
            validation_loss = train_loss
            monitor_loss = train_loss

        scheduler.step(
            monitor_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if monitor_loss < best_loss:
            best_loss = monitor_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epoch < 10
            or
            (epoch + 1) % 10 == 0
        ):
            if validation_loader is not None:
                print(
                    f"Epoch "
                    f"{epoch + 1:03d}/{EPOCHS}"
                    f" | Train: {train_loss:.4f}"
                    f" | Val: {validation_loss:.4f}"
                    f" | LR: {current_lr:.7f}"
                )
            else:
                print(
                    f"Epoch "
                    f"{epoch + 1:03d}/{EPOCHS}"
                    f" | Loss: {train_loss:.4f}"
                    f" | LR: {current_lr:.7f}"
                )

        if (
            use_validation
            and
            epochs_without_improvement >= PATIENCE
        ):
            print(
                f"Early stopping at epoch {epoch + 1}."
            )
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    torch.save(
        model.state_dict(),
        model_path(verse)
    )

    torch.save(
        model.state_dict(),
        best_model_path(verse)
    )

    print()
    print(
        "Training complete."
    )
    print(
        f"Best loss: {best_loss:.4f}"
    )
    print(
        f"Model: {model_path(verse)}"
    )
    print(
        f"Best model: {best_model_path(verse)}"
    )
    print(
        f"Scaler: {scaler_path(verse)}"
    )

    return model


def train_verse(verse):
    if not os.path.isdir(
        verse_dir(verse)
    ):
        raise RuntimeError(
            f"Verse folder not found:\n"
            f"{verse_dir(verse)}"
        )

    references = load_references(
        verse
    )

    recordings = load_labeled_recordings(
        verse
    )

    print()
    print(
        f"Verse: {verse}"
    )
    print(
        f"Good references: {len(references)}"
    )
    print(
        f"Labeled recordings: {len(recordings)}"
    )

    X, y = build_dataset(
        references,
        recordings
    )

    train_model(
        verse,
        X,
        y
    )


def train_all_verses():
    verses = discover_verses()

    if not verses:
        raise RuntimeError(
            f"No verse folders found in:\n"
            f"{DATA_DIR}"
        )

    print()
    print(
        "==================================="
    )
    print(
        "ANUKRAMA MULTI-VERSE TRAINING"
    )
    print(
        "==================================="
    )
    print(
        f"Found {len(verses)} verses:"
    )
    print(
        " ".join(verses)
    )

    successful = []
    failed = []

    for verse in verses:
        try:
            train_verse(
                verse
            )

            successful.append(
                verse
            )

        except Exception as error:
            failed.append(
                (
                    verse,
                    str(error)
                )
            )

            print()
            print(
                f"ERROR TRAINING {verse}:"
            )
            print(
                error
            )
            print(
                "Continuing..."
            )

    print()
    print(
        "==================================="
    )
    print(
        "TRAINING SUMMARY"
    )
    print(
        "==================================="
    )

    print(
        f"Successful: "
        f"{len(successful)}/{len(verses)}"
    )

    for verse in successful:
        print(
            f"  {verse}"
        )

    if failed:
        print()
        print(
            "Failed:"
        )

        for verse, error in failed:
            print(
                f"  {verse}: {error}"
            )


def load_scaler(verse):
    path = scaler_path(
        verse
    )

    if not os.path.exists(path):
        raise RuntimeError(
            f"Scaler not found:\n"
            f"{path}\n"
            f"Train {verse} first."
        )

    data = np.load(path)

    return (
        data["mean"],
        data["std"]
    )


def load_model(verse):
    path = best_model_path(
        verse
    )

    if not os.path.exists(path):
        raise RuntimeError(
            f"No trained model found for {verse}.\n"
            f"Run:\n"
            f"python3 train.py train {verse}"
        )

    scaler_mean, scaler_std = load_scaler(
        verse
    )

    model = PronunciationModel(
        input_size=len(scaler_mean)
    ).to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


def score_recording(
    verse,
    test_path
):
    print()
    print(
        "==================================="
    )
    print(
        "Anukrama Pronunciation Test"
    )
    print(
        "==================================="
    )

    print(
        f"Verse: {verse}"
    )

    print(
        f"Recording: {test_path}"
    )

    references = load_references(
        verse
    )

    model = load_model(
        verse
    )

    scaler_mean, scaler_std = load_scaler(
        verse
    )

    audio = load_pcm(
        test_path
    )

    if not is_valid_speech_sample(audio):
        raise ValueError(
            "No clear verse pronunciation detected. "
            "Please record the verse more clearly."
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

    print(
        f"Comparing against "
        f"{len(references)} references..."
    )

    comparisons = []

    for index, reference in enumerate(
        references,
        start=1
    ):
        features = compare_features(
            reference,
            recording
        )

        comparisons.append(
            features
        )

        print(
            f"  Reference {index}: "
            f"DTW={features[0]:.4f}"
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
        prediction = model(x)

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
        "==================================="
    )
    print(
        "COMPARISON"
    )
    print(
        "==================================="
    )

    print(
        f"Mean DTW:   {mean_features[0]:.4f}"
    )

    print(
        f"Best DTW:   {min_features[0]:.4f}"
    )

    print(
        f"Worst DTW:  {max_features[0]:.4f}"
    )

    print(
        f"Median DTW: {median_features[0]:.4f}"
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
        f"Verse: {verse}"
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


def print_help():
    print()
    print(
        "Anukrama pronunciation trainer"
    )

    print()
    print(
        "Train all verses:"
    )

    print(
        "  python3 train.py train"
    )

    print()
    print(
        "Train one verse:"
    )

    print(
        "  python3 train.py train v01"
    )

    print()
    print(
        "Score a recording:"
    )

    print(
        "  python3 train.py score v01 recording.pcm"
    )

    print()
    print(
        "List verses:"
    )

    print(
        "  python3 train.py verses"
    )


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == "train":
        if len(sys.argv) == 2:
            train_all_verses()
        else:
            verse = sys.argv[2]

            if not re.match(
                r"^v\d+$",
                verse
            ):
                raise RuntimeError(
                    f"Invalid verse: {verse}"
                )

            train_verse(
                verse
            )

        return

    if command == "score":
        if len(sys.argv) < 4:
            print(
                "Usage:"
            )
            print(
                "  python3 train.py score v01 recording.pcm"
            )
            return

        verse = sys.argv[2]
        test_path = sys.argv[3]

        if not re.match(
            r"^v\d+$",
            verse
        ):
            raise RuntimeError(
                f"Invalid verse: {verse}"
            )

        if not os.path.exists(
            test_path
        ):
            raise RuntimeError(
                f"Recording not found:\n"
                f"{test_path}"
            )

        score_recording(
            verse,
            test_path
        )

        return

    if command == "verses":
        verses = discover_verses()

        print()

        for verse in verses:
            good_count = len(
                glob.glob(
                    os.path.join(
                        verse_dir(verse),
                        "good",
                        "*.pcm"
                    )
                )
            )

            labeled_count = len(
                glob.glob(
                    os.path.join(
                        verse_dir(verse),
                        "labeled",
                        "bad_*.pcm"
                    )
                )
            )

            trained = os.path.exists(
                best_model_path(verse)
            )

            status = (
                "trained"
                if trained
                else "not trained"
            )

            print(
                f"{verse}: "
                f"{good_count} references, "
                f"{labeled_count} labeled, "
                f"{status}"
            )

        return

    if command in (
        "help",
        "-h",
        "--help"
    ):
        print_help()
        return

    print(
        f"Unknown command: {command}"
    )

    print_help()


if __name__ == "__main__":
    main()