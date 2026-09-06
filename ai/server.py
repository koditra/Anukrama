import base64
import glob
import io
import json
import os
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from train import (
    PronunciationModel,
    apply_scaler,
    compare_features,
    extract_features,
    load_pcm,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ch15")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

REFERENCE_CACHE = {}
MODEL_CACHE = {}
SCALER_CACHE = {}


def aggregate_comparisons(comparisons):
    if not comparisons:
        raise ValueError("No reference comparisons available.")

    count = len(comparisons)
    mean = np.zeros(4, dtype=np.float32)
    minimum = np.zeros(4, dtype=np.float32)
    maximum = np.zeros(4, dtype=np.float32)
    median = np.zeros(4, dtype=np.float32)

    for c in range(4):
        values = [row[c] for row in comparisons]
        total = float(sum(values))
        mean[c] = total / count
        values_sorted = sorted(values)
        minimum[c] = values_sorted[0]
        maximum[c] = values_sorted[-1]
        middle = count // 2
        median[c] = values_sorted[middle] if count % 2 else (values_sorted[middle - 1] + values_sorted[middle]) / 2.0

    output = np.zeros(16, dtype=np.float32)
    output[0:4] = mean
    output[4:8] = minimum
    output[8:12] = maximum
    output[12:16] = median
    return output


def load_reference_features(verse):
    if verse in REFERENCE_CACHE:
        return REFERENCE_CACHE[verse]

    pattern = os.path.join(DATA_DIR, verse, "good", "*.pcm")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No reference recordings found for {verse}")

    references = []
    for path in files:
        audio = load_pcm(path)
        references.append(extract_features(audio))

    REFERENCE_CACHE[verse] = references
    return references


def load_model_and_scaler(verse):
    if verse in MODEL_CACHE and verse in SCALER_CACHE:
        return MODEL_CACHE[verse], SCALER_CACHE[verse]

    model_path = os.path.join(MODEL_DIR, f"anukrama_{verse}_score_best.pt")
    scaler_path = os.path.join(SCALER_DIR, f"anukrama_{verse}_scaler.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for {verse}: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found for {verse}: {scaler_path}")

    scaler_data = np.load(scaler_path)
    mean = scaler_data["mean"]
    std = scaler_data["std"]

    model = PronunciationModel(input_size=len(mean)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    MODEL_CACHE[verse] = model
    SCALER_CACHE[verse] = (mean, std)
    return model, (mean, std)


def score_waveform(verse, wav_bytes):
    if verse not in {"v01", "v02"}:
        raise ValueError(f"Unsupported verse: {verse}")

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            frames = wav_file.readframes(wav_file.getnframes())
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        if channels != 1:
            mono = pcm.reshape(-1, channels)
            pcm = mono.mean(axis=1).astype(np.float32)
    except wave.Error:
        pcm = np.frombuffer(wav_bytes, dtype="<i2").astype(np.float32)
        sample_rate = 48000

    if pcm.size == 0:
        raise ValueError("The audio payload is empty.")

    if sample_rate != 48000:
        pcm = pcm.astype(np.float32)

    pcm /= 32768.0

    recording = extract_features(pcm)
    references = load_reference_features(verse)
    comparisons = [compare_features(reference, recording) for reference in references]
    features = aggregate_comparisons(comparisons)

    model, (mean, std) = load_model_and_scaler(verse)
    scaled = apply_scaler(features, mean, std)
    input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_tensor)

    score = float(prediction.item())
    score = max(0.0, min(100.0, score))
    return float(score)


class AIHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "service": "anukrama-ai",
                "verses": ["v01", "v02"],
                "device": str(DEVICE),
            })
            return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/score":
            self._send_json({"error": "not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))

            verse = payload.get("verse")
            audio_b64 = payload.get("audio")

            if verse not in {"v01", "v02"}:
                raise ValueError("Unsupported verse. Expected v01 or v02.")

            if not audio_b64:
                raise ValueError("Missing audio payload.")

            audio_bytes = base64.b64decode(audio_b64)
            score = score_waveform(verse, audio_bytes)
            self._send_json({"score": score, "verse": verse})
        except Exception as exc:  # pragma: no cover - server error path
            self._send_json({"error": str(exc)}, 400)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8123
    server = ThreadingHTTPServer((host, port), AIHandler)
    print(f"Anukrama AI backend listening on http://{host}:{port}")
    server.serve_forever()
