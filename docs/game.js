const BASE_URL = "https://raw.githubusercontent.com/koditra/Anukrama/main";

const TOTAL_VERSES = 20;
const SAMPLE_RATE = 48000;

const AI_BASE = `${BASE_URL}/ai`;

const MODEL_URLS = {
    v01: `${AI_BASE}/models/anukrama_v01_score_best.onnx`,
    v02: `${AI_BASE}/models/anukrama_v02_score_best.onnx`
};

const SCALER_URLS = {
    v01: `${AI_BASE}/scalers/anukrama_v01_scaler.npz`,
    v02: `${AI_BASE}/scalers/anukrama_v02_scaler.npz`
};

const REFERENCE_URLS = {
    v01: `${BASE_URL}/audio/chapter_15/v01.pcm`,
    v02: `${BASE_URL}/audio/chapter_15/v02.pcm`
};

let currentVerse = 1;
let currentAudio = null;
let onnxRuntime = null;

const models = {
    v01: null,
    v02: null
};

const scalers = {
    v01: null,
    v02: null
};

const references = {
    v01: null,
    v02: null
};

let audioContext = null;
let mediaStream = null;
let mediaRecorder = null;
let recordingChunks = [];
let recordedAudio = null;
let recordedSampleRate = SAMPLE_RATE;
let isRecording = false;

const verseNumber = document.getElementById("verseNumber");
const progressText = document.getElementById("progressText");
const quarterDevanagari = document.getElementById("quarterDevanagari");
const quarterEnglish = document.getElementById("quarterEnglish");
const verseDevanagari = document.getElementById("verseDevanagari");
const verseEnglish = document.getElementById("verseEnglish");
const answer = document.getElementById("answer");
const revealButton = document.getElementById("revealButton");
const audioButton = document.getElementById("audioButton");
const previousButton = document.getElementById("previousButton");
const randomButton = document.getElementById("randomButton");
const nextButton = document.getElementById("nextButton");
const aiStatus = document.getElementById("aiStatus");
const recordButton = document.getElementById("recordButton");
const scoreButton = document.getElementById("scoreButton");
const aiResult = document.getElementById("aiResult");
const scoreValue = document.getElementById("scoreValue");
const scoreMessage = document.getElementById("scoreMessage");

async function loadONNXRuntime() {
    if (onnxRuntime) {
        return onnxRuntime;
    }

    if (window.ort) {
        onnxRuntime = window.ort;
        onnxRuntime.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        return onnxRuntime;
    }

    if (aiStatus) {
        aiStatus.textContent = "Loading pronunciation AI...";
    }

    const existingScript = document.querySelector(
        'script[src*="onnxruntime-web"]'
    );

    if (existingScript) {
        await new Promise((resolve, reject) => {
            if (window.ort) {
                resolve();
                return;
            }

            existingScript.addEventListener("load", resolve, {
                once: true
            });

            existingScript.addEventListener("error", () => {
                reject(
                    new Error("Failed to load ONNX Runtime Web.")
                );
            }, {
                once: true
            });
        });
    } else {
        await new Promise((resolve, reject) => {
            const script = document.createElement("script");

            script.src =
                "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";

            script.onload = resolve;

            script.onerror = () => {
                reject(
                    new Error("Failed to load ONNX Runtime Web.")
                );
            };

            document.head.appendChild(script);
        });
    }

    if (!window.ort) {
        throw new Error(
            "ONNX Runtime Web loaded but window.ort is unavailable."
        );
    }

    onnxRuntime = window.ort;

    onnxRuntime.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    return onnxRuntime;
}

async function fetchText(path) {
    const response = await fetch(`${BASE_URL}/${path}`);

    if (!response.ok) {
        throw new Error(`Failed to load ${path}`);
    }

    return response.text();
}

async function fetchJson(path) {
    const response = await fetch(`${BASE_URL}/${path}`);

    if (!response.ok) {
        throw new Error(`Failed to load ${path}`);
    }

    return response.json();
}

async function fetchBinary(path) {
    const response = await fetch(path);

    if (!response.ok) {
        throw new Error(`Failed to load ${path}`);
    }

    return response.arrayBuffer();
}

function createWav(buffer) {
    const pcm = new Int16Array(buffer);
    const channels = 1;
    const bitsPerSample = 16;
    const blockAlign = channels * bitsPerSample / 8;
    const byteRate = SAMPLE_RATE * blockAlign;
    const wav = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(wav);

    function writeString(offset, value) {
        for (let i = 0; i < value.length; i++) {
            view.setUint8(
                offset + i,
                value.charCodeAt(i)
            );
        }
    }

    writeString(0, "RIFF");
    view.setUint32(
        4,
        36 + pcm.length * 2,
        true
    );
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, "data");
    view.setUint32(
        40,
        pcm.length * 2,
        true
    );

    for (let i = 0; i < pcm.length; i++) {
        view.setInt16(
            44 + i * 2,
            pcm[i],
            true
        );
    }

    return new Blob(
        [wav],
        { type: "audio/wav" }
    );
}

async function stopCurrentAudio() {
    if (!currentAudio) {
        return;
    }

    const audio = currentAudio;
    currentAudio = null;

    if (audio.paused || audio.ended) {
        audio.pause();
        audio.currentTime = 0;
        return;
    }

    return new Promise(resolve => {
        const fadeDuration = 300;
        const steps = 15;
        const intervalTime = fadeDuration / steps;
        const startVolume = audio.volume;
        let step = 0;

        const fade = setInterval(() => {
            step++;

            audio.volume = Math.max(
                0,
                startVolume * (1 - step / steps)
            );

            if (step >= steps) {
                clearInterval(fade);
                audio.pause();
                audio.currentTime = 0;
                audio.volume = 1;
                resolve();
            }
        }, intervalTime);
    });
}

async function playPcm(path) {
    try {
        await stopCurrentAudio();

        const response = await fetch(
            `${BASE_URL}/${path}`
        );

        if (!response.ok) {
            throw new Error(
                `Audio not found: ${path}`
            );
        }

        const buffer =
            await response.arrayBuffer();

        const wav = createWav(buffer);
        const url = URL.createObjectURL(wav);
        const audio = new Audio(url);

        currentAudio = audio;
        audio.volume = 1;

        audio.addEventListener(
            "ended",
            () => {
                URL.revokeObjectURL(url);

                if (currentAudio === audio) {
                    currentAudio = null;
                }
            }
        );

        audio.addEventListener(
            "error",
            () => {
                URL.revokeObjectURL(url);

                if (currentAudio === audio) {
                    currentAudio = null;
                }
            }
        );

        await audio.play();

        return true;
    } catch (error) {
        console.error(error);
        return false;
    }
}

async function loadVerse(id) {
    await stopCurrentAudio();

    currentVerse = id;

    verseNumber.textContent = id;
    progressText.textContent =
        `${id} / ${TOTAL_VERSES}`;

    quarterDevanagari.textContent = "Loading...";
    quarterEnglish.textContent = "Loading...";
    verseDevanagari.textContent = "";
    verseEnglish.textContent = "";

    answer.classList.add("hidden");

    revealButton.textContent = "Reveal verse";

    audioButton.disabled = true;
    audioButton.textContent = "Loading...";

    const card =
        document.querySelector(".practice-card");

    if (card) {
        card.classList.remove("verse-enter");
        void card.offsetWidth;
        card.classList.add("verse-enter");
    }

    try {
        const meta =
            await fetchJson(
                `meta/verse_${id}.json`
            );

        const devanagari =
            await fetchText(
                `verses/devanagari/verse_${id}.txt`
            );

        const english =
            await fetchText(
                `verses/english/verse_${id}.txt`
            );

        if (currentVerse !== id) {
            return;
        }

        quarterDevanagari.textContent =
            meta.quarter.devanagari;

        quarterEnglish.textContent =
            meta.quarter.english;

        verseDevanagari.textContent =
            devanagari.trim();

        verseEnglish.textContent =
            english.trim();

        audioButton.disabled = false;
        audioButton.textContent = "Play verse";

        if (id !== 1) {
            const audioNumber =
                String(id).padStart(2, "0");

            await playPcm(
                `audio/chapter_15/v${audioNumber}_prompt.pcm`
            );
        }

        updateAIForVerse();
    } catch (error) {
        console.error(error);

        quarterDevanagari.textContent =
            "Unable to load verse";

        quarterEnglish.textContent =
            "Something went wrong while loading this verse.";

        verseDevanagari.textContent = "";
        verseEnglish.textContent = "";

        audioButton.disabled = true;
        audioButton.textContent =
            "Audio unavailable";
    }
}

function revealVerse() {
    answer.classList.toggle("hidden");

    if (answer.classList.contains("hidden")) {
        revealButton.textContent =
            "Reveal verse";
    } else {
        revealButton.textContent =
            "Hide verse";
    }
}

async function playVerseAudio() {
    const audioNumber =
        String(currentVerse).padStart(2, "0");

    audioButton.disabled = true;
    audioButton.textContent = "Playing...";

    const success = await playPcm(
        `audio/chapter_15/v${audioNumber}.pcm`
    );

    audioButton.disabled = false;

    audioButton.textContent =
        success
            ? "Play verse"
            : "Audio unavailable";
}

async function nextVerse() {
    let next = currentVerse + 1;

    if (next > TOTAL_VERSES) {
        next = 1;
    }

    await loadVerse(next);
}

async function previousVerse() {
    let previous = currentVerse - 1;

    if (previous < 1) {
        previous = TOTAL_VERSES;
    }

    await loadVerse(previous);
}

async function randomVerse() {
    let random;

    do {
        random =
            Math.floor(
                Math.random() * TOTAL_VERSES
            ) + 1;
    } while (random === currentVerse);

    await loadVerse(random);
}

async function loadAIModel(key) {
    const runtime =
        await loadONNXRuntime();

    if (models[key]) {
        return models[key];
    }

    models[key] =
        await runtime.InferenceSession.create(
            MODEL_URLS[key],
            {
                executionProviders: ["wasm"]
            }
        );

    return models[key];
}

function parseNPY(buffer) {
    const bytes = new Uint8Array(buffer);

    if (
        bytes[0] !== 0x93 ||
        bytes[1] !== 0x4e ||
        bytes[2] !== 0x55 ||
        bytes[3] !== 0x4d ||
        bytes[4] !== 0x50 ||
        bytes[5] !== 0x59
    ) {
        throw new Error("Invalid NPY file.");
    }

    const major = bytes[6];
    const minor = bytes[7];

    let headerLength;
    let headerOffset;

    if (major === 1) {
        headerLength =
            bytes[8] |
            (bytes[9] << 8);

        headerOffset = 10;
    } else {
        headerLength =
            bytes[8] |
            (bytes[9] << 8) |
            (bytes[10] << 16) |
            (bytes[11] << 24);

        headerOffset = 12;
    }

    const headerBytes =
        bytes.slice(
            headerOffset,
            headerOffset + headerLength
        );

    const header =
        new TextDecoder().decode(
            headerBytes
        );

    const descrMatch =
        header.match(
            /['"]descr['"]\s*:\s*['"]([^'"]+)['"]/
        );

    const shapeMatch =
        header.match(
            /['"]shape['"]\s*:\s*\(([^)]*)\)/
        );

    if (!descrMatch) {
        throw new Error(
            "NPY descriptor missing."
        );
    }

    const descr = descrMatch[1];

    const shape = shapeMatch
        ? shapeMatch[1]
            .split(",")
            .map(value => value.trim())
            .filter(Boolean)
            .map(Number)
        : [];

    const dataOffset =
        headerOffset + headerLength;

    let TypedArray;

    if (
        descr === "<f4" ||
        descr === "|f4"
    ) {
        TypedArray = Float32Array;
    } else if (
        descr === "<f8" ||
        descr === "|f8"
    ) {
        TypedArray = Float64Array;
    } else {
        throw new Error(
            `Unsupported NPY dtype: ${descr}`
        );
    }

    const byteLength =
        bytes.byteLength - dataOffset;

    const count =
        Math.floor(
            byteLength /
            TypedArray.BYTES_PER_ELEMENT
        );

    const values =
        new TypedArray(
            bytes.buffer,
            bytes.byteOffset + dataOffset,
            count
        );

    return {
        data: Array.from(values),
        shape
    };
}

async function loadZipEntries(buffer) {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    const entries = {};

    let offset = 0;

    while (offset + 4 <= bytes.length) {
        const signature =
            view.getUint32(
                offset,
                true
            );

        if (signature === 0x04034b50) {
            const compression =
                view.getUint16(
                    offset + 8,
                    true
                );

            const compressedSize =
                view.getUint32(
                    offset + 18,
                    true
                );

            const fileNameLength =
                view.getUint16(
                    offset + 26,
                    true
                );

            const extraLength =
                view.getUint16(
                    offset + 28,
                    true
                );

            const nameStart =
                offset + 30;

            const name =
                new TextDecoder().decode(
                    bytes.slice(
                        nameStart,
                        nameStart + fileNameLength
                    )
                );

            const dataStart =
                nameStart +
                fileNameLength +
                extraLength;

            const dataEnd =
                dataStart +
                compressedSize;

            const compressed =
                bytes.slice(
                    dataStart,
                    dataEnd
                );

            if (compression === 0) {
                entries[name] =
                    compressed.buffer.slice(
                        compressed.byteOffset,
                        compressed.byteOffset +
                            compressed.byteLength
                    );
            } else if (compression === 8) {
                if (
                    !("DecompressionStream" in window)
                ) {
                    throw new Error(
                        "This browser does not support ZIP decompression."
                    );
                }

                const stream =
                    new Blob([compressed])
                        .stream()
                        .pipeThrough(
                            new DecompressionStream(
                                "deflate-raw"
                            )
                        );

                entries[name] =
                    await new Response(
                        stream
                    ).arrayBuffer();
            } else {
                throw new Error(
                    `Unsupported ZIP compression: ${compression}`
                );
            }

            offset = dataEnd;
        } else if (
            signature === 0x02014b50 ||
            signature === 0x06054b50
        ) {
            break;
        } else {
            offset++;
        }
    }

    return entries;
}

async function loadScaler(key) {
    const buffer =
        await fetchBinary(
            SCALER_URLS[key]
        );

    const entries =
        await loadZipEntries(buffer);

    const names =
        Object.keys(entries);

    const meanName =
        names.find(
            name =>
                name.endsWith("mean.npy") ||
                name.endsWith("mean_.npy")
        );

    const scaleName =
        names.find(
            name =>
                name.endsWith("scale.npy") ||
                name.endsWith("scale_.npy")
        );

    if (!meanName || !scaleName) {
        throw new Error(
            `Scaler arrays not found in ${key} scaler.`
        );
    }

    const mean =
        parseNPY(
            entries[meanName]
        ).data;

    const std =
        parseNPY(
            entries[scaleName]
        ).data;

    scalers[key] = {
        mean,
        std
    };

    return scalers[key];
}

async function loadReference(key) {
    const buffer =
        await fetchBinary(
            REFERENCE_URLS[key]
        );

    if (buffer.byteLength % 2 !== 0) {
        throw new Error(
            "Reference PCM has an invalid byte length."
        );
    }

    const pcm =
        new Int16Array(buffer);

    const samples =
        new Float32Array(
            pcm.length
        );

    for (let i = 0; i < pcm.length; i++) {
        samples[i] =
            pcm[i] / 32768;
    }

    references[key] = {
        samples,
        sampleRate: SAMPLE_RATE
    };

    return references[key];
}

function updateAIForVerse() {
    if (
        !aiStatus ||
        !recordButton ||
        !scoreButton
    ) {
        return;
    }

    if (
        currentVerse !== 1 &&
        currentVerse !== 2
    ) {
        aiStatus.textContent =
            "Pronunciation scoring is currently available for Verses 1 and 2.";

        recordButton.disabled = true;
        scoreButton.disabled = true;

        return;
    }

    const key =
        `v${String(currentVerse).padStart(2, "0")}`;

    if (
        models[key] &&
        scalers[key] &&
        references[key]
    ) {
        aiStatus.textContent =
            "Pronunciation AI ready.";

        recordButton.disabled = false;
    } else {
        aiStatus.textContent =
            "Loading pronunciation AI...";

        recordButton.disabled = true;
        scoreButton.disabled = true;
    }
}

function resampleAudio(
    samples,
    sourceRate,
    targetRate
) {
    if (sourceRate === targetRate) {
        return new Float32Array(samples);
    }

    const ratio =
        sourceRate / targetRate;

    const outputLength =
        Math.floor(
            samples.length / ratio
        );

    const output =
        new Float32Array(
            outputLength
        );

    for (
        let i = 0;
        i < outputLength;
        i++
    ) {
        const position =
            i * ratio;

        const left =
            Math.floor(position);

        const right =
            Math.min(
                left + 1,
                samples.length - 1
            );

        const amount =
            position - left;

        output[i] =
            samples[left] * (1 - amount) +
            samples[right] * amount;
    }

    return output;
}

async function decodeRecordedAudio(blob) {
    if (!audioContext) {
        audioContext =
            new AudioContext();
    }

    const buffer =
        await blob.arrayBuffer();

    const decoded =
        await audioContext.decodeAudioData(
            buffer
        );

    const channel =
        decoded.getChannelData(0);

    return {
        samples: resampleAudio(
            channel,
            decoded.sampleRate,
            SAMPLE_RATE
        ),
        sampleRate: SAMPLE_RATE
    };
}

async function startRecording() {
    if (isRecording) {
        stopRecording();
        return;
    }

    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {
        aiStatus.textContent =
            "Microphone access is unavailable.";

        return;
    }

    try {
        mediaStream =
            await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });

        recordingChunks = [];

        mediaRecorder =
            new MediaRecorder(
                mediaStream
            );

        mediaRecorder.addEventListener(
            "dataavailable",
            event => {
                if (event.data.size > 0) {
                    recordingChunks.push(
                        event.data
                    );
                }
            }
        );

        mediaRecorder.addEventListener(
            "stop",
            async () => {
                const mimeType =
                    mediaRecorder.mimeType;

                recordedAudio =
                    new Blob(
                        recordingChunks,
                        {
                            type: mimeType
                        }
                    );

                mediaStream
                    .getTracks()
                    .forEach(
                        track => track.stop()
                    );

                mediaStream = null;
                isRecording = false;

                recordButton.textContent =
                    "Record pronunciation";

                scoreButton.disabled = true;

                aiStatus.textContent =
                    "Processing recording...";

                try {
                    const decoded =
                        await decodeRecordedAudio(
                            recordedAudio
                        );

                    recordedAudio = decoded;

                    recordedSampleRate =
                        decoded.sampleRate;

                    scoreButton.disabled =
                        false;

                    aiStatus.textContent =
                        "Recording ready to score.";
                } catch (error) {
                    console.error(error);

                    aiStatus.textContent =
                        `Audio processing error: ${error.message}`;

                    scoreButton.disabled = true;
                }
            }
        );

        mediaRecorder.start();

        isRecording = true;

        scoreButton.disabled = true;

        recordButton.textContent =
            "Stop recording";

        aiStatus.textContent =
            `Recording verse ${currentVerse}...`;
    } catch (error) {
        console.error(error);

        aiStatus.textContent =
            `Microphone error: ${error.message}`;
    }
}

function stopRecording() {
    if (
        mediaRecorder &&
        mediaRecorder.state === "recording"
    ) {
        mediaRecorder.stop();
    }
}

function normalizeAudio(samples) {
    let peak = 0;

    for (let i = 0; i < samples.length; i++) {
        peak =
            Math.max(
                peak,
                Math.abs(samples[i])
            );
    }

    if (peak === 0) {
        return new Float32Array(samples);
    }

    const output =
        new Float32Array(
            samples.length
        );

    for (let i = 0; i < samples.length; i++) {
        output[i] =
            samples[i] / peak;
    }

    return output;
}

function removeSilence(samples) {
    let peak = 0;

    for (let i = 0; i < samples.length; i++) {
        peak =
            Math.max(
                peak,
                Math.abs(samples[i])
            );
    }

    if (peak < 0.0001) {
        return new Float32Array(samples);
    }

    const threshold =
        peak * 0.03;

    let start = 0;
    let end =
        samples.length - 1;

    while (
        start < samples.length &&
        Math.abs(samples[start]) < threshold
    ) {
        start++;
    }

    while (
        end > start &&
        Math.abs(samples[end]) < threshold
    ) {
        end--;
    }

    return samples.slice(
        start,
        end + 1
    );
}

function calculateRMS(samples) {
    if (!samples.length) {
        return 0;
    }

    let sum = 0;

    for (let i = 0; i < samples.length; i++) {
        sum +=
            samples[i] *
            samples[i];
    }

    return Math.sqrt(
        sum / samples.length
    );
}

function mean(values) {
    if (!values.length) {
        return 0;
    }

    let total = 0;

    for (const value of values) {
        total += value;
    }

    return total / values.length;
}

function standardDeviation(values) {
    if (!values.length) {
        return 0;
    }

    const average =
        mean(values);

    let sum = 0;

    for (const value of values) {
        const difference =
            value - average;

        sum +=
            difference *
            difference;
    }

    return Math.sqrt(
        sum / values.length
    );
}

function statistics(values) {
    if (!values.length) {
        return {
            mean: 0,
            std: 0,
            min: 0,
            max: 0,
            median: 0
        };
    }

    const sorted =
        [...values].sort(
            (a, b) => a - b
        );

    const middle =
        Math.floor(
            sorted.length / 2
        );

    const median =
        sorted.length % 2 === 0
            ? (
                sorted[middle - 1] +
                sorted[middle]
            ) / 2
            : sorted[middle];

    return {
        mean: mean(values),
        std: standardDeviation(values),
        min: sorted[0],
        max:
            sorted[sorted.length - 1],
        median
    };
}

function computeDTW(a, b) {
    if (!a.length || !b.length) {
        return Infinity;
    }

    const n = a.length;
    const m = b.length;

    const previous =
        new Float64Array(m + 1);

    const current =
        new Float64Array(m + 1);

    previous.fill(Infinity);
    previous[0] = 0;

    for (let i = 1; i <= n; i++) {
        current.fill(Infinity);

        for (let j = 1; j <= m; j++) {
            const cost =
                Math.abs(
                    a[i - 1] -
                    b[j - 1]
                );

            current[j] =
                cost +
                Math.min(
                    previous[j],
                    current[j - 1],
                    previous[j - 1]
                );
        }

        previous.set(current);
    }

    return previous[m] /
        Math.max(n, m);
}

function extractBasicFeatures(samples) {
    const clean =
        removeSilence(samples);

    const normalized =
        normalizeAudio(clean);

    const rms =
        calculateRMS(normalized);

    const values = [];

    const frameSize = 2048;
    const hopSize = 512;

    for (
        let start = 0;
        start + frameSize <= normalized.length;
        start += hopSize
    ) {
        let sum = 0;

        for (
            let i = 0;
            i < frameSize;
            i++
        ) {
            const value =
                normalized[start + i];

            sum +=
                value * value;
        }

        values.push(
            Math.sqrt(
                sum / frameSize
            )
        );
    }

    return {
        duration:
            normalized.length /
            SAMPLE_RATE,

        rms,

        frameValues: values,

        statistics:
            statistics(values)
    };
}

function buildFeatureVector(
    recording,
    reference
) {
    const recordingFeatures =
        extractBasicFeatures(
            recording
        );

    const referenceFeatures =
        extractBasicFeatures(
            reference
        );

    const recordingStats =
        recordingFeatures.statistics;

    const referenceStats =
        referenceFeatures.statistics;

    const featureMeanDifference =
        recordingStats.mean -
        referenceStats.mean;

    const featureStdDifference =
        recordingStats.std -
        referenceStats.std;

    const durationDifference =
        recordingFeatures.duration -
        referenceFeatures.duration;

    const dtw =
        computeDTW(
            recordingFeatures.frameValues,
            referenceFeatures.frameValues
        );

    return new Float32Array([
        dtw,
        featureMeanDifference,
        featureStdDifference,
        recordingStats.min -
            referenceStats.min,
        recordingStats.max -
            referenceStats.max,
        recordingStats.median -
            referenceStats.median,
        recordingStats.mean,
        referenceStats.mean,
        recordingStats.std,
        referenceStats.std,
        recordingStats.min,
        referenceStats.min,
        recordingStats.max,
        referenceStats.max,
        durationDifference,
        recordingFeatures.duration -
            referenceFeatures.duration
    ]);
}

function scaleFeatures(
    features,
    scaler
) {
    if (
        !scaler ||
        !scaler.mean ||
        !scaler.std
    ) {
        throw new Error(
            "Scaler is not loaded."
        );
    }

    if (
        scaler.mean.length !==
            features.length ||
        scaler.std.length !==
            features.length
    ) {
        throw new Error(
            `Scaler expects ${scaler.mean.length} features, received ${features.length}.`
        );
    }

    const output =
        new Float32Array(
            features.length
        );

    for (
        let i = 0;
        i < features.length;
        i++
    ) {
        const std =
            scaler.std[i];

        output[i] =
            Math.abs(std) < 1e-12
                ? 0
                : (
                    features[i] -
                    scaler.mean[i]
                ) / std;
    }

    return output;
}

async function runONNX(
    key,
    features
) {
    const session =
        models[key];

    if (!session) {
        throw new Error(
            `${key} model is not loaded.`
        );
    }

    const runtime =
        await loadONNXRuntime();

    const inputName =
        session.inputNames[0];

    const input =
        new runtime.Tensor(
            "float32",
            features,
            [1, features.length]
        );

    const result =
        await session.run({
            [inputName]: input
        });

    const outputName =
        session.outputNames[0];

    return result[outputName];
}

function convertPredictionToScore(output) {
    const values =
        Array.from(output.data);

    let value =
        Number(values[0]);

    if (!Number.isFinite(value)) {
        return 0;
    }

    if (
        value >= 0 &&
        value <= 1
    ) {
        value *= 100;
    }

    return Math.max(
        0,
        Math.min(
            100,
            Math.round(value)
        )
    );
}

function displayScore(score) {
    if (aiResult) {
        aiResult.classList.remove(
            "hidden"
        );
    }

    if (scoreValue) {
        scoreValue.textContent =
            score;
    }

    if (scoreMessage) {
        if (score >= 90) {
            scoreMessage.textContent =
                "Excellent pronunciation!";
        } else if (score >= 75) {
            scoreMessage.textContent =
                "Good pronunciation!";
        } else if (score >= 60) {
            scoreMessage.textContent =
                "Pretty good. Keep practicing!";
        } else {
            scoreMessage.textContent =
                "Keep practicing and try again!";
        }
    }
}

async function initializeAI() {
    if (
        !recordButton ||
        !aiStatus
    ) {
        return;
    }

    try {
        aiStatus.textContent =
            "Loading browser AI...";

        await loadONNXRuntime();

        aiStatus.textContent =
            "Loading pronunciation models...";

        await Promise.all([
            loadAIModel("v01"),
            loadAIModel("v02"),
            loadScaler("v01"),
            loadScaler("v02"),
            loadReference("v01"),
            loadReference("v02")
        ]);

        aiStatus.textContent =
            "Pronunciation AI ready.";

        updateAIForVerse();
    } catch (error) {
        console.error(
            "AI initialization failed:",
            error
        );

        aiStatus.textContent =
            `AI error: ${error.message}`;

        recordButton.disabled = true;
        scoreButton.disabled = true;
    }
}

async function scoreRecording() {
    if (!recordedAudio) {
        return;
    }

    if (
        currentVerse !== 1 &&
        currentVerse !== 2
    ) {
        return;
    }

    const key =
        `v${String(currentVerse).padStart(2, "0")}`;

    if (!models[key]) {
        aiStatus.textContent =
            "The pronunciation model is not ready.";

        return;
    }

    if (!references[key]) {
        aiStatus.textContent =
            "The reference recording is not ready.";

        return;
    }

    if (!scalers[key]) {
        aiStatus.textContent =
            "The pronunciation scaler is not ready.";

        return;
    }

    scoreButton.disabled = true;
    recordButton.disabled = true;

    aiStatus.textContent =
        "Analyzing pronunciation...";

    try {
        const features =
            buildFeatureVector(
                recordedAudio.samples,
                references[key].samples
            );

        const scaled =
            scaleFeatures(
                features,
                scalers[key]
            );

        const output =
            await runONNX(
                key,
                scaled
            );

        const score =
            convertPredictionToScore(
                output
            );

        displayScore(score);

        aiStatus.textContent =
            "Pronunciation scored.";
    } catch (error) {
        console.error(
            "Scoring failed:",
            error
        );

        aiStatus.textContent =
            `Scoring error: ${error.message}`;
    } finally {
        scoreButton.disabled = false;
        recordButton.disabled = false;
    }
}

if (revealButton) {
    revealButton.addEventListener(
        "click",
        revealVerse
    );
}

if (audioButton) {
    audioButton.addEventListener(
        "click",
        playVerseAudio
    );
}

if (previousButton) {
    previousButton.addEventListener(
        "click",
        previousVerse
    );
}

if (randomButton) {
    randomButton.addEventListener(
        "click",
        randomVerse
    );
}

if (nextButton) {
    nextButton.addEventListener(
        "click",
        nextVerse
    );
}

if (recordButton) {
    recordButton.disabled = true;

    recordButton.addEventListener(
        "click",
        async () => {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }
    );
}

if (scoreButton) {
    scoreButton.disabled = true;

    scoreButton.addEventListener(
        "click",
        scoreRecording
    );
}

document.addEventListener(
    "keydown",
    event => {
        if (
            event.target.tagName === "INPUT" ||
            event.target.tagName === "TEXTAREA"
        ) {
            return;
        }

        if (event.key === "ArrowLeft") {
            previousVerse();
        }

        if (event.key === "ArrowRight") {
            nextVerse();
        }

        if (event.code === "Space") {
            event.preventDefault();
            revealVerse();
        }

        if (
            event.key.toLowerCase() === "r"
        ) {
            randomVerse();
        }
    }
);

loadVerse(1);
initializeAI();