const BASE_URL = "https://raw.githubusercontent.com/koditra/Anukrama/main";

const TOTAL_VERSES = 20;
const SAMPLE_RATE = 48000;
const N_MFCC = 20;
const N_MELS = 40;
const N_FFT = 2048;
const HOP_LENGTH = 512;

const AI_BASE = `${BASE_URL}/ai`;
const GITHUB_API = "https://api.github.com/repos/koditra/Anukrama/contents";

function resolveAssetUrl(path) {
    const normalized = path.replace(/^\.?\//, "");
    const host = typeof window !== "undefined" && window.location ? window.location.hostname : "";
    const localHost = host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";

    if (localHost) {
        const relative = `../${normalized}`;
        return new URL(relative, window.location.href).toString();
    }

    return `${BASE_URL}/${normalized}`;
}

const MODEL_URLS = {
    v01: resolveAssetUrl("ai/models/anukrama_v01_score_best.onnx"),
    v02: resolveAssetUrl("ai/models/anukrama_v02_score_best.onnx")
};

const SCALER_URLS = {
    v01: resolveAssetUrl("ai/scalers/anukrama_v01_scaler.npz"),
    v02: resolveAssetUrl("ai/scalers/anukrama_v02_scaler.npz")
};

const REFERENCE_DIRS = {
    v01: "ai/data/ch15/v01/good",
    v02: "ai/data/ch15/v02/good"
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

const referenceFeatures = {
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

    window.Module = window.Module || {};
    window.Module.MountedFiles = window.Module.MountedFiles || {};

    const existingScript = document.querySelector(
        'script[src*="onnxruntime-web"]'
    );

    if (existingScript) {
        await new Promise((resolve, reject) => {
            if (window.ort) {
                resolve();
                return;
            }

            existingScript.addEventListener(
                "load",
                resolve,
                { once: true }
            );

            existingScript.addEventListener(
                "error",
                () => {
                    reject(
                        new Error(
                            "Failed to load ONNX Runtime Web."
                        )
                    );
                },
                { once: true }
            );
        });
    } else {
        await new Promise((resolve, reject) => {
            const script = document.createElement("script");

            script.src =
                "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";

            script.onload = resolve;

            script.onerror = () => {
                reject(
                    new Error(
                        "Failed to load ONNX Runtime Web."
                    )
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
    const normalized = path.replace(/^\.?\//, "");
    const response = await fetch(`${BASE_URL}/${normalized}`);

    if (!response.ok) {
        throw new Error(`Failed to load ${path}`);
    }

    return response.text();
}

async function fetchJson(path) {
    const normalized = path.replace(/^\.?\//, "");
    const response = await fetch(`${BASE_URL}/${normalized}`);

    if (!response.ok) {
        throw new Error(`Failed to load ${path}`);
    }

    return response.json();
}

async function fetchBinary(url) {
    const normalized = url.replace(/^https?:\/\//i, "");
    const finalUrl = normalized.startsWith("raw.githubusercontent.com/")
        ? `https://${normalized}`
        : /^https?:\/\//i.test(url)
            ? url
            : `${BASE_URL}/${url.replace(/^\.?\//, "")}`;

    const response = await fetch(finalUrl);

    if (!response.ok) {
        throw new Error(`Failed to load ${url}`);
    }

    return response.arrayBuffer();
}

function createWav(buffer) {
    const pcm = new Int16Array(buffer);
    const channels = 1;
    const bitsPerSample = 16;
    const blockAlign = channels * bitsPerSample / 8;
    const byteRate = SAMPLE_RATE * blockAlign;

    const wav = new ArrayBuffer(
        44 + pcm.length * 2
    );

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

    quarterDevanagari.textContent =
        "Loading...";

    quarterEnglish.textContent =
        "Loading...";

    verseDevanagari.textContent = "";
    verseEnglish.textContent = "";

    answer.classList.add("hidden");

    revealButton.textContent =
        "Reveal verse";

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
        audioButton.textContent =
            "Play verse";

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

    revealButton.textContent =
        answer.classList.contains("hidden")
            ? "Reveal verse"
            : "Hide verse";
}

async function playVerseAudio() {
    const audioNumber =
        String(currentVerse).padStart(2, "0");

    audioButton.disabled = true;
    audioButton.textContent =
        "Playing...";

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
    } while (
        random === currentVerse
    );

    await loadVerse(random);
}

async function loadAIModel(key) {
    const runtime =
        await loadONNXRuntime();

    if (models[key]) {
        return models[key];
    }

    const modelUrl = MODEL_URLS[key];

    try {
        models[key] = await runtime.InferenceSession.create(modelUrl, {
            executionProviders: ["wasm"]
        });
        return models[key];
    } catch (err) {
        console.warn(`WASM model load failed for ${key}:`, err && err.message ? err.message : err);
    }

    try {
        console.warn(`Falling back to WebGL provider for ${key}`);
        models[key] = await runtime.InferenceSession.create(modelUrl, {
            executionProviders: ["webgl"]
        });
        return models[key];
    } catch (err2) {
        console.warn(`WebGL fallback failed for ${key}:`, err2 && err2.message ? err2.message : err2);
    }

    try {
        models[key] = await runtime.InferenceSession.create(modelUrl);
        return models[key];
    } catch (finalErr) {
        throw finalErr;
    }
}

function readUInt16(view, offset) {
    return view.getUint16(
        offset,
        true
    );
}

function readUInt32(view, offset) {
    return view.getUint32(
        offset,
        true
    );
}

function parseNPY(buffer) {
    const bytes =
        new Uint8Array(buffer);

    if (
        bytes.length < 10 ||
        bytes[0] !== 0x93 ||
        bytes[1] !== 0x4e ||
        bytes[2] !== 0x55 ||
        bytes[3] !== 0x4d ||
        bytes[4] !== 0x50 ||
        bytes[5] !== 0x59
    ) {
        throw new Error(
            "Invalid NPY file."
        );
    }

    const major = bytes[6];
    const minor = bytes[7];

    let headerLength;
    let headerOffset;

    if (major === 1) {
        headerLength =
            readUInt16(
                new DataView(buffer),
                8
            );

        headerOffset = 10;
    } else if (major === 2 || major === 3) {
        headerLength =
            readUInt32(
                new DataView(buffer),
                8
            );

        headerOffset = 12;
    } else {
        throw new Error(
            `Unsupported NPY version ${major}.${minor}.`
        );
    }

    const headerEnd =
        headerOffset +
        headerLength;

    if (headerEnd > bytes.length) {
        throw new Error(
            "NPY header extends beyond file."
        );
    }

    const header =
        new TextDecoder().decode(
            bytes.slice(
                headerOffset,
                headerEnd
            )
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
            "NPY dtype descriptor missing."
        );
    }

    const descr =
        descrMatch[1];

    let shape = [];

    if (shapeMatch) {
        shape =
            shapeMatch[1]
                .split(",")
                .map(value => value.trim())
                .filter(Boolean)
                .map(Number);
    }

    const fortranMatch =
        header.match(
            /['"]fortran_order['"]\s*:\s*(True|False)/
        );

    const fortranOrder =
        fortranMatch
            ? fortranMatch[1] === "True"
            : false;

    if (fortranOrder) {
        throw new Error(
            "Fortran-order NPY arrays are unsupported."
        );
    }

    const dataOffset =
        headerEnd;

    let bytesPerElement;

    if (descr.endsWith("f4")) {
        bytesPerElement = 4;
    } else if (descr.endsWith("f8")) {
        bytesPerElement = 8;
    } else {
        throw new Error(
            `Unsupported NPY dtype: ${descr}`
        );
    }

    const count =
        Math.floor(
            (
                bytes.length -
                dataOffset
            ) /
            bytesPerElement
        );

    const output =
        new Array(count);

    const dataView =
        new DataView(buffer);

    const firstChar = descr[0];

    let littleEndian;

    if (firstChar === ">") {
        littleEndian = false;
    } else if (firstChar === "<" || firstChar === "|") {
        littleEndian = true;
    } else {
        littleEndian = true;
    }

    for (
        let i = 0;
        i < count;
        i++
    ) {
        const offset =
            dataOffset +
            i *
            bytesPerElement;

        if (bytesPerElement === 4) {
            output[i] =
                dataView.getFloat32(
                    offset,
                    littleEndian
                );
        } else {
            output[i] =
                dataView.getFloat64(
                    offset,
                    littleEndian
                );
        }
    }

    return {
        data: output,
        shape
    };
}

async function loadZipEntries(buffer) {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    const entries = {};

    let eocdOffset = -1;
    const maxComment = 0xffff;
    const startSearch = Math.max(0, bytes.length - (maxComment + 22));

    for (let i = bytes.length - 22; i >= startSearch; i--) {
        if (view.getUint32(i, true) === 0x06054b50) {
            eocdOffset = i;
            break;
        }
    }

    if (eocdOffset === -1) {
        throw new Error("Invalid ZIP: EOCD record not found.");
    }

    const centralDirSize = view.getUint32(eocdOffset + 12, true);
    const centralDirOffset = view.getUint32(eocdOffset + 16, true);

    let offset = centralDirOffset;

    while (offset + 46 <= bytes.length) {
        const sig = view.getUint32(offset, true);

        if (sig !== 0x02014b50) break;

        const compression = view.getUint16(offset + 10, true);
        const compressedSize = view.getUint32(offset + 20, true);
        const uncompressedSize = view.getUint32(offset + 24, true);
        const fileNameLength = view.getUint16(offset + 28, true);
        const extraLength = view.getUint16(offset + 30, true);
        const commentLength = view.getUint16(offset + 32, true);
        const relOffset = view.getUint32(offset + 42, true);

        const nameStart = offset + 46;
        const nameEnd = nameStart + fileNameLength;

        if (nameEnd > bytes.length) break;

        const name = new TextDecoder().decode(bytes.slice(nameStart, nameEnd));

        if (relOffset + 30 > bytes.length) {
            throw new Error(`Invalid local header offset for ${name}`);
        }

        const localSig = view.getUint32(relOffset, true);

        if (localSig !== 0x04034b50) {
            throw new Error(`Invalid local header for ${name}`);
        }

        const localNameLen = view.getUint16(relOffset + 26, true);
        const localExtraLen = view.getUint16(relOffset + 28, true);

        const dataStart = relOffset + 30 + localNameLen + localExtraLen;
        const dataEnd = dataStart + compressedSize;

        if (dataEnd > bytes.length) {
            throw new Error(`Invalid ZIP entry: ${name}`);
        }

        const compressed = bytes.slice(dataStart, dataEnd);

        if (compression === 0) {
            entries[name] = compressed.buffer.slice(compressed.byteOffset, compressed.byteOffset + compressed.byteLength);
        } else if (compression === 8) {
            if (!('DecompressionStream' in window)) {
                throw new Error('This browser does not support ZIP decompression.');
            }

            const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate-raw'));

            entries[name] = await new Response(stream).arrayBuffer();
        } else {
            throw new Error(`Unsupported ZIP compression: ${compression}`);
        }

        offset = offset + 46 + fileNameLength + extraLength + commentLength;
    }

    return entries;
}

async function loadScaler(key) {
    if (scalers[key]) {
        return scalers[key];
    }

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
                name === "mean.npy" ||
                name.endsWith("/mean.npy")
        );

    const stdName =
        names.find(
            name =>
                name === "std.npy" ||
                name.endsWith("/std.npy")
        );

    if (!meanName || !stdName) {
        throw new Error(
            `Scaler arrays not found in ${key} scaler. Found: ${names.join(", ")}`
        );
    }

    const mean =
        parseNPY(
            entries[meanName]
        ).data;

    const std =
        parseNPY(
            entries[stdName]
        ).data;

    if (
        mean.length !== 16 ||
        std.length !== 16
    ) {
        throw new Error(
            `${key} scaler contains mean=${mean.length}, std=${std.length}; expected 16 and 16.`
        );
    }

    scalers[key] = {
        mean,
        std
    };

    return scalers[key];
}

async function listReferenceFiles(key) {
    const path =
        REFERENCE_DIRS[key];

    try {
        const response = await fetch(`${GITHUB_API}/${path}?per_page=100`, {
            headers: {
                "Accept": "application/vnd.github+json",
                "User-Agent": "Anukrama-GitHub-Pages"
            }
        });

        if (response.ok) {
            const files = await response.json();

            return files
                .filter(
                    file =>
                        file.type === "file" &&
                        file.name
                            .toLowerCase()
                            .endsWith(".pcm")
                )
                .sort((a, b) =>
                    a.name.localeCompare(b.name, undefined, {
                        numeric: true
                    })
                );
        }
    } catch (e) {
        // ignore and fall back to probing raw file names
    }

    // GitHub Pages and raw GitHub are the intended hosting model for this repo.
    // Avoid depending on the GitHub API because it can return 403 in anonymous browser contexts.
    // Filename pattern in the repo: e.g. v01_01.pcm, v01_02.pcm, ...
    const prefix = key; // 'v01' or 'v02'
    const results = [];

    let consecutiveMisses = 0;

    for (let i = 1; i <= 50; i++) {
        const name = `${prefix}_${String(i).padStart(2, "0")}.pcm`;
        const url = `${BASE_URL}/${path}/${name}`;

        try {
            // Prefer HEAD to avoid downloading full files; fall back to GET if HEAD fails.
            let ok = false;

            try {
                const head = await fetch(url, { method: "HEAD" });
                ok = head.ok;
            } catch (headErr) {
                // Some hosts/CORS configurations disallow HEAD; try GET.
                const get = await fetch(url, { method: "GET" });
                ok = get.ok;
            }

            if (ok) {
                results.push({ name, download_url: url, type: "file" });
                consecutiveMisses = 0;
            } else {
                consecutiveMisses++;
            }
        } catch (err) {
            consecutiveMisses++;
        }

        // Stop after several consecutive misses to avoid long probing.
        if (consecutiveMisses >= 6) break;
    }

    // Sort discovered files and return in the same shape as GitHub API results.
    results.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));

    return results;
}

function pcmToFloat32(buffer) {
    if (
        buffer.byteLength % 2 !== 0
    ) {
        throw new Error(
            "PCM file has an odd byte length."
        );
    }

    const view =
        new DataView(buffer);

    const count =
        buffer.byteLength / 2;

    const output =
        new Float32Array(count);

    for (
        let i = 0;
        i < count;
        i++
    ) {
        output[i] =
            view.getInt16(
                i * 2,
                true
            ) /
            32768;
    }

    return output;
}

async function loadReference(key) {
    if (references[key]) {
        return references[key];
    }

    const files =
        await listReferenceFiles(key);

    if (!files.length) {
        throw new Error(
            `No good reference recordings found for ${key}.`
        );
    }

    const loaded = [];

    for (const file of files) {
        const buffer =
            await fetchBinary(
                file.download_url
            );

        loaded.push({
            name: file.name,
            samples: pcmToFloat32(buffer),
            sampleRate: SAMPLE_RATE
        });
    }

    references[key] = loaded;

    return loaded;
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
        referenceFeatures[key]
    ) {
        aiStatus.textContent =
            `Pronunciation AI ready — ${referenceFeatures[key].length} reference recordings.`;

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
    if (
        sourceRate === targetRate
    ) {
        return new Float32Array(
            samples
        );
    }

    const ratio =
        sourceRate /
        targetRate;

    const outputLength =
        Math.floor(
            samples.length /
            ratio
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
            samples[left] *
                (1 - amount) +
            samples[right] *
                amount;
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
                if (
                    event.data.size > 0
                ) {
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

                const blob =
                    new Blob(
                        recordingChunks,
                        {
                            type: mimeType
                        }
                    );

                mediaStream
                    .getTracks()
                    .forEach(
                        track =>
                            track.stop()
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
                            blob
                        );

                    recordedAudio =
                        decoded;

                    recordedSampleRate =
                        decoded.sampleRate;

                    scoreButton.disabled =
                        false;

                    aiStatus.textContent =
                        "Recording ready to score.";
                } catch (error) {
                    console.error(error);

                    recordedAudio = null;

                    aiStatus.textContent =
                        `Audio processing error: ${error.message}`;

                    scoreButton.disabled =
                        true;
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

function prepareAudio(samples) {
    const audio =
        new Float64Array(
            samples.length
        );

    let mean = 0;

    for (
        let i = 0;
        i < samples.length;
        i++
    ) {
        mean += samples[i];
    }

    mean /=
        samples.length;

    for (
        let i = 0;
        i < samples.length;
        i++
    ) {
        audio[i] =
            samples[i] -
            mean;
    }

    let peak = 0;

    for (
        let i = 0;
        i < audio.length;
        i++
    ) {
        peak =
            Math.max(
                peak,
                Math.abs(audio[i])
            );
    }

    if (peak > 0) {
        for (
            let i = 0;
            i < audio.length;
            i++
        ) {
            audio[i] /=
                peak;
        }
    }

    return audio;
}

function hannWindow(length) {
    const window =
        new Float64Array(
            length
        );

    for (
        let i = 0;
        i < length;
        i++
    ) {
        window[i] =
            0.5 -
            0.5 *
                Math.cos(
                    2 *
                    Math.PI *
                    i /
                    length
                );
    }

    return window;
}

function fftReal(samples) {
    const n =
        samples.length;

    const real =
        new Float64Array(
            samples
        );

    const imag =
        new Float64Array(n);

    let j = 0;

    for (
        let i = 1;
        i < n;
        i++
    ) {
        let bit =
            n >> 1;

        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j) {
            const temp =
                real[i];

            real[i] =
                real[j];

            real[j] =
                temp;
        }
    }

    for (
        let size = 2;
        size <= n;
        size <<= 1
    ) {
        const angle =
            -2 *
            Math.PI /
            size;

        const wReal =
            Math.cos(angle);

        const wImag =
            Math.sin(angle);

        const half =
            size >> 1;

        for (
            let start = 0;
            start < n;
            start += size
        ) {
            let currentReal = 1;
            let currentImag = 0;

            for (
                let k = 0;
                k < half;
                k++
            ) {
                const even =
                    start + k;

                const odd =
                    even + half;

                const oddReal =
                    real[odd] *
                        currentReal -
                    imag[odd] *
                        currentImag;

                const oddImag =
                    real[odd] *
                        currentImag +
                    imag[odd] *
                        currentReal;

                const evenReal =
                    real[even];

                const evenImag =
                    imag[even];

                real[even] =
                    evenReal +
                    oddReal;

                imag[even] =
                    evenImag +
                    oddImag;

                real[odd] =
                    evenReal -
                    oddReal;

                imag[odd] =
                    evenImag -
                    oddImag;

                const nextReal =
                    currentReal *
                        wReal -
                    currentImag *
                        wImag;

                currentImag =
                    currentReal *
                        wImag +
                    currentImag *
                        wReal;

                currentReal =
                    nextReal;
            }
        }
    }

    return {
        real,
        imag
    };
}

function centerPadConstant(
    samples,
    size
) {
    const pad =
        Math.floor(
            size / 2
        );

    const output =
        new Float64Array(
            samples.length +
            pad * 2
        );

    output.set(
        samples,
        pad
    );

    return output;
}

function centerPadEdge(
    samples,
    size
) {
    const pad =
        Math.floor(
            size / 2
        );

    const output =
        new Float64Array(
            samples.length +
            pad * 2
        );

    for (
        let i = 0;
        i < output.length;
        i++
    ) {
        const source =
            Math.max(
                0,
                Math.min(
                    samples.length - 1,
                    i - pad
                )
            );

        output[i] =
            samples[source];
    }

    return output;
}

function frameAudio(
    samples,
    centerMode = "constant"
) {
    const padded =
        centerMode === "edge"
            ? centerPadEdge(
                samples,
                N_FFT
            )
            : centerPadConstant(
                samples,
                N_FFT
            );

    const frameCount =
        1 +
        Math.floor(
            (
                padded.length -
                N_FFT
            ) /
            HOP_LENGTH
        );

    const frames =
        new Array(
            frameCount
        );

    const window =
        hannWindow(N_FFT);

    for (
        let frame = 0;
        frame < frameCount;
        frame++
    ) {
        const start =
            frame *
            HOP_LENGTH;

        const values =
            new Float64Array(
                N_FFT
            );

        for (
            let i = 0;
            i < N_FFT;
            i++
        ) {
            values[i] =
                padded[start + i] *
                window[i];
        }

        frames[frame] =
            values;
    }

    return frames;
}

function computeSTFTPower(
    samples
) {
    const frames =
        frameAudio(
            samples,
            "constant"
        );

    const bins =
        N_FFT / 2 + 1;

    const output =
        new Array(
            frames.length
        );

    for (
        let frame = 0;
        frame < frames.length;
        frame++
    ) {
        const fft =
            fftReal(
                frames[frame]
            );

        const power =
            new Float64Array(
                bins
            );

        for (
            let k = 0;
            k < bins;
            k++
        ) {
            power[k] =
                (
                    fft.real[k] *
                    fft.real[k] +
                    fft.imag[k] *
                    fft.imag[k]
                ) /
                N_FFT;
        }

        output[frame] =
            power;
    }

    return output;
}

function hzToMelSlaney(hz) {
    if (hz < 1000) {
        return hz / 200;
    }

    return 15 +
        27 *
        Math.log10(
            hz / 1000
        );
}

function melToHzSlaney(mel) {
    if (mel < 15) {
        return mel * 200;
    }

    return 1000 *
        Math.pow(
            10,
            (mel - 15) / 27
        );
}

function createMelFilterBank() {
    const filters =
        new Array(
            N_MELS
        );

    const minMel =
        hzToMelSlaney(0);

    const maxMel =
        hzToMelSlaney(
            SAMPLE_RATE / 2
        );

    const melPoints =
        new Float64Array(
            N_MELS + 2
        );

    for (
        let i = 0;
        i < N_MELS + 2;
        i++
    ) {
        melPoints[i] =
            minMel +
            (
                maxMel -
                minMel
            ) *
            i /
            (N_MELS + 1);
    }

    const frequencies =
        new Float64Array(
            N_MELS + 2
        );

    for (
        let i = 0;
        i < frequencies.length;
        i++
    ) {
        frequencies[i] =
            melToHzSlaney(
                melPoints[i]
            );
    }

    const bins =
        N_FFT / 2 + 1;

    const fftFrequencies =
        new Float64Array(
            bins
        );

    for (
        let k = 0;
        k < bins;
        k++
    ) {
        fftFrequencies[k] =
            k *
            SAMPLE_RATE /
            N_FFT;
    }

    for (
        let m = 0;
        m < N_MELS;
        m++
    ) {
        const filter =
            new Float64Array(
                bins
            );

        const lower =
            frequencies[m];

        const center =
            frequencies[m + 1];

        const upper =
            frequencies[m + 2];

        for (
            let k = 0;
            k < bins;
            k++
        ) {
            const frequency =
                fftFrequencies[k];

            if (
                frequency >= lower &&
                frequency < center &&
                center > lower
            ) {
                filter[k] =
                    (
                        frequency -
                        lower
                    ) /
                    (
                        center -
                        lower
                    );
            } else if (
                frequency >= center &&
                frequency <= upper &&
                upper > center
            ) {
                filter[k] =
                    (
                        upper -
                        frequency
                    ) /
                    (
                        upper -
                        center
                    );
            }
        }

        let area = 0;

        for (
            let k = 0;
            k < bins;
            k++
        ) {
            area +=
                filter[k];
        }

        if (area > 0) {
            const binWidth =
                SAMPLE_RATE /
                N_FFT;

            for (
                let k = 0;
                k < bins;
                k++
            ) {
                filter[k] *=
                    2 /
                    (
                        upper -
                        lower
                    );
            }
        }

        filters[m] =
            filter;
    }

    return filters;
}

const MEL_FILTER_BANK =
    createMelFilterBank();

function dctTypeIIOrtho(
    values,
    outputCount
) {
    const n =
        values.length;

    const output =
        new Float64Array(
            outputCount
        );

    for (
        let k = 0;
        k < outputCount;
        k++
    ) {
        let sum = 0;

        for (
            let i = 0;
            i < n;
            i++
        ) {
            sum +=
                values[i] *
                Math.cos(
                    Math.PI *
                    k *
                    (
                        2 * i + 1
                    ) /
                    (
                        2 * n
                    )
                );
        }

        output[k] =
            k === 0
                ? sum *
                    Math.sqrt(
                        1 / n
                    )
                : sum *
                    Math.sqrt(
                        2 / n
                    );
    }

    return output;
}

function computeMFCC(
    powerFrames
) {
    const output =
        new Array(
            powerFrames.length
        );

    for (
        let frame = 0;
        frame < powerFrames.length;
        frame++
    ) {
        const mel =
            new Float64Array(
                N_MELS
            );

        for (
            let m = 0;
            m < N_MELS;
            m++
        ) {
            const filter =
                MEL_FILTER_BANK[m];

            let sum = 0;

            for (
                let k = 0;
                k < filter.length;
                k++
            ) {
                sum +=
                    powerFrames[frame][k] *
                    filter[k];
            }

            mel[m] =
                Math.max(
                    sum,
                    Number.EPSILON
                );
        }

        for (
            let m = 0;
            m < N_MELS;
            m++
        ) {
            mel[m] =
                Math.log(
                    mel[m]
                );
        }

        output[frame] =
            dctTypeIIOrtho(
                mel,
                N_MFCC
            );
    }

    return output;
}

function computeDelta(
    matrix
) {
    const frames =
        matrix.length;

    if (!frames) {
        return [];
    }

    const channels =
        matrix[0].length;

    const output =
        new Array(
            frames
        );

    const half = 4;

    let denominator = 0;

    for (
        let i = 1;
        i <= half;
        i++
    ) {
        denominator +=
            i * i;
    }

    denominator *= 2;

    for (
        let t = 0;
        t < frames;
        t++
    ) {
        const row =
            new Float64Array(
                channels
            );

        for (
            let c = 0;
            c < channels;
            c++
        ) {
            let value = 0;

            for (
                let n = 1;
                n <= half;
                n++
            ) {
                const left =
                    Math.max(
                        0,
                        t - n
                    );

                const right =
                    Math.min(
                        frames - 1,
                        t + n
                    );

                value +=
                    n *
                    (
                        matrix[right][c] -
                        matrix[left][c]
                    );
            }

            row[c] =
                value /
                denominator;
        }

        output[t] =
            row;
    }

    return output;
}

function computeSpectralCentroid(
    power,
    frequencies
) {
    let total = 0;
    let weighted = 0;

    for (
        let k = 0;
        k < power.length;
        k++
    ) {
        total += power[k];

        weighted +=
            power[k] *
            frequencies[k];
    }

    if (total <= 0) {
        return 0;
    }

    return weighted / total;
}

function computeSpectralBandwidth(
    power,
    frequencies,
    centroid
) {
    let total = 0;
    let weighted = 0;

    for (
        let k = 0;
        k < power.length;
        k++
    ) {
        total += power[k];

        const difference =
            frequencies[k] -
            centroid;

        weighted +=
            power[k] *
            difference *
            difference;
    }

    if (total <= 0) {
        return 0;
    }

    return Math.sqrt(
        weighted / total
    );
}

function computeSpectralRolloff(
    power,
    frequencies
) {
    let total = 0;

    for (
        let k = 0;
        k < power.length;
        k++
    ) {
        total += power[k];
    }

    if (total <= 0) {
        return 0;
    }

    const target =
        total * 0.85;

    let cumulative = 0;

    for (
        let k = 0;
        k < power.length;
        k++
    ) {
        cumulative += power[k];

        if (
            cumulative >= target
        ) {
            return frequencies[k];
        }
    }

    return frequencies[
        frequencies.length - 1
    ];
}

function computeZeroCrossingRate(
    samples
) {
    const padded =
        centerPadEdge(
            samples,
            N_FFT
        );

    const frameCount =
        1 +
        Math.floor(
            (
                padded.length -
                N_FFT
            ) /
            HOP_LENGTH
        );

    const output =
        new Float64Array(
            frameCount
        );

    for (
        let frame = 0;
        frame < frameCount;
        frame++
    ) {
        const start =
            frame *
            HOP_LENGTH;

        let crossings = 0;

        for (
            let i = 1;
            i < N_FFT;
            i++
        ) {
            const a =
                padded[
                    start + i - 1
                ];

            const b =
                padded[
                    start + i
                ];

            if (
                (
                    a < 0 &&
                    b >= 0
                ) ||
                (
                    a >= 0 &&
                    b < 0
                )
            ) {
                crossings++;
            }
        }

        output[frame] =
            crossings /
            N_FFT;
    }

    return output;
}

function computeRMS(
    samples
) {
    const padded =
        centerPadConstant(
            samples,
            N_FFT
        );

    const frameCount =
        1 +
        Math.floor(
            (
                padded.length -
                N_FFT
            ) /
            HOP_LENGTH
        );

    const output =
        new Float64Array(
            frameCount
        );

    for (
        let frame = 0;
        frame < frameCount;
        frame++
    ) {
        const start =
            frame *
            HOP_LENGTH;

        let sum = 0;

        for (
            let i = 0;
            i < N_FFT;
            i++
        ) {
            const value =
                padded[
                    start + i
                ];

            sum +=
                value * value;
        }

        output[frame] =
            Math.sqrt(
                sum / N_FFT
            );
    }

    return output;
}

function normalizeFeatureMatrix(
    matrix
) {
    const frames =
        matrix.length;

    if (!frames) {
        return matrix;
    }

    const channels =
        matrix[0].length;

    const mean =
        new Float64Array(
            channels
        );

    const std =
        new Float64Array(
            channels
        );

    for (
        let t = 0;
        t < frames;
        t++
    ) {
        for (
            let c = 0;
            c < channels;
            c++
        ) {
            mean[c] +=
                matrix[t][c];
        }
    }

    for (
        let c = 0;
        c < channels;
        c++
    ) {
        mean[c] /=
            frames;
    }

    for (
        let t = 0;
        t < frames;
        t++
    ) {
        for (
            let c = 0;
            c < channels;
            c++
        ) {
            const difference =
                matrix[t][c] -
                mean[c];

            std[c] +=
                difference *
                difference;
        }
    }

    for (
        let c = 0;
        c < channels;
        c++
    ) {
        std[c] =
            Math.sqrt(
                std[c] /
                frames
            );

        if (
            std[c] < 1e-6
        ) {
            std[c] = 1;
        }
    }

    const output =
        new Array(
            frames
        );

    for (
        let t = 0;
        t < frames;
        t++
    ) {
        const row =
            new Float32Array(
                channels
            );

        for (
            let c = 0;
            c < channels;
            c++
        ) {
            row[c] =
                (
                    matrix[t][c] -
                    mean[c]
                ) /
                std[c];
        }

        output[t] =
            row;
    }

    return output;
}

function extractFeatures(
    inputSamples
) {
    const audio =
        prepareAudio(
            inputSamples
        );

    if (!audio.length) {
        throw new Error(
            "Audio recording is empty."
        );
    }

    const powerFrames =
        computeSTFTPower(
            audio
        );

    const mfcc =
        computeMFCC(
            powerFrames
        );

    const delta =
        computeDelta(
            mfcc
        );

    const bins =
        N_FFT / 2 + 1;

    const frequencies =
        new Float64Array(
            bins
        );

    for (
        let k = 0;
        k < bins;
        k++
    ) {
        frequencies[k] =
            k *
            SAMPLE_RATE /
            N_FFT;
    }

    const centroids =
        new Float64Array(
            powerFrames.length
        );

    const bandwidths =
        new Float64Array(
            powerFrames.length
        );

    const rolloffs =
        new Float64Array(
            powerFrames.length
        );

    for (
        let t = 0;
        t < powerFrames.length;
        t++
    ) {
        centroids[t] =
            computeSpectralCentroid(
                powerFrames[t],
                frequencies
            );

        bandwidths[t] =
            computeSpectralBandwidth(
                powerFrames[t],
                frequencies,
                centroids[t]
            );

        rolloffs[t] =
            computeSpectralRolloff(
                powerFrames[t],
                frequencies
            );
    }

    const zeroCrossings =
        computeZeroCrossingRate(
            audio
        );

    const rms =
        computeRMS(
            audio
        );

    const frames =
        powerFrames.length;

    const featureMatrix =
        new Array(frames);

    for (
        let t = 0;
        t < frames;
        t++
    ) {
        const row =
            new Float64Array(45);

        let index = 0;

        for (
            let c = 0;
            c < 20;
            c++
        ) {
            row[index++] =
                mfcc[t][c];
        }

        for (
            let c = 0;
            c < 20;
            c++
        ) {
            row[index++] =
                delta[t][c];
        }

        row[index++] =
            centroids[t];

        row[index++] =
            bandwidths[t];

        row[index++] =
            rolloffs[t];

        row[index++] =
            zeroCrossings[t];

        row[index++] =
            rms[t];

        featureMatrix[t] =
            row;
    }

    return normalizeFeatureMatrix(
        featureMatrix
    );
}

function featureMean(
    matrix
) {
    const frames =
        matrix.length;

    const channels =
        matrix[0].length;

    const output =
        new Float64Array(
            channels
        );

    for (
        let t = 0;
        t < frames;
        t++
    ) {
        for (
            let c = 0;
            c < channels;
            c++
        ) {
            output[c] +=
                matrix[t][c];
        }
    }

    for (
        let c = 0;
        c < channels;
        c++
    ) {
        output[c] /=
            frames;
    }

    return output;
}

function featureStd(
    matrix,
    mean
) {
    const frames =
        matrix.length;

    const channels =
        matrix[0].length;

    const output =
        new Float64Array(
            channels
        );

    for (
        let t = 0;
        t < frames;
        t++
    ) {
        for (
            let c = 0;
            c < channels;
            c++
        ) {
            const difference =
                matrix[t][c] -
                mean[c];

            output[c] +=
                difference *
                difference;
        }
    }

    for (
        let c = 0;
        c < channels;
        c++
    ) {
        output[c] =
            Math.sqrt(
                output[c] /
                frames
            );
    }

    return output;
}

function euclideanDistance(
    a,
    b
) {
    let sum = 0;

    for (
        let i = 0;
        i < 45;
        i++
    ) {
        const difference =
            a[i] -
            b[i];

        sum +=
            difference *
            difference;
    }

    return Math.sqrt(sum);
}

function dtwDistance(
    reference,
    recording
) {
    const n =
        reference.length;

    const m =
        recording.length;

    if (
        n === 0 ||
        m === 0
    ) {
        return Infinity;
    }

    let previous =
        new Float64Array(
            m + 1
        );

    let current =
        new Float64Array(
            m + 1
        );

    previous.fill(
        Infinity
    );

    previous[0] = 0;

    for (
        let i = 1;
        i <= n;
        i++
    ) {
        current.fill(
            Infinity
        );

        for (
            let j = 1;
            j <= m;
            j++
        ) {
            const cost =
                euclideanDistance(
                    reference[i - 1],
                    recording[j - 1]
                );

            current[j] =
                cost +
                Math.min(
                    previous[j],
                    current[j - 1],
                    previous[j - 1]
                );
        }

        const temp =
            previous;

        previous =
            current;

        current =
            temp;
    }

    let i = n;
    let j = m;
    let pathLength = 1;

    while (
        i > 0 ||
        j > 0
    ) {
        if (i === 0) {
            j--;
        } else if (j === 0) {
            i--;
        } else {
            const diagonal =
                previous === null
                    ? Infinity
                    : Infinity;

            let bestDirection;

            if (
                i === 1 &&
                j === 1
            ) {
                bestDirection = 0;
            } else {
                const up =
                    getDTWCell(
                        reference,
                        recording,
                        i - 1,
                        j
                    );

                const left =
                    getDTWCell(
                        reference,
                        recording,
                        i,
                        j - 1
                    );

                const diag =
                    getDTWCell(
                        reference,
                        recording,
                        i - 1,
                        j - 1
                    );

                if (
                    diag <= up &&
                    diag <= left
                ) {
                    bestDirection = 0;
                } else if (
                    up <= left
                ) {
                    bestDirection = 1;
                } else {
                    bestDirection = 2;
                }
            }

            if (bestDirection === 0) {
                i--;
                j--;
            } else if (bestDirection === 1) {
                i--;
            } else {
                j--;
            }
        }

        pathLength++;
    }

    return (
        previous[m] /
        pathLength
    );
}

function getDTWCell(
    reference,
    recording,
    i,
    j
) {
    if (
        i < 0 ||
        j < 0
    ) {
        return Infinity;
    }

    const n =
        reference.length;

    const m =
        recording.length;

    const width =
        m + 1;

    const matrix =
        new Float64Array(
            (n + 1) *
            (m + 1)
        );

    matrix.fill(
        Infinity
    );

    matrix[0] = 0;

    for (
        let r = 1;
        r <= n;
        r++
    ) {
        for (
            let c = 1;
            c <= m;
            c++
        ) {
            const cost =
                euclideanDistance(
                    reference[r - 1],
                    recording[c - 1]
                );

            matrix[
                r * width + c
            ] =
                cost +
                Math.min(
                    matrix[
                        (r - 1) *
                        width + c
                    ],
                    matrix[
                        r *
                        width + c - 1
                    ],
                    matrix[
                        (r - 1) *
                        width + c - 1
                    ]
                );
        }
    }

    return matrix[
        i * width + j
    ];
}

function compareFeatures(
    reference,
    recording
) {
    const n =
        reference.length;

    const m =
        recording.length;

    const width =
        m + 1;

    const matrix =
        new Float64Array(
            (n + 1) *
            (m + 1)
        );

    matrix.fill(
        Infinity
    );

    matrix[0] = 0;

    for (
        let i = 1;
        i <= n;
        i++
    ) {
        for (
            let j = 1;
            j <= m;
            j++
        ) {
            const cost =
                euclideanDistance(
                    reference[i - 1],
                    recording[j - 1]
                );

            matrix[
                i * width + j
            ] =
                cost +
                Math.min(
                    matrix[
                        (i - 1) *
                        width + j
                    ],
                    matrix[
                        i *
                        width + j - 1
                    ],
                    matrix[
                        (i - 1) *
                        width + j - 1
                    ]
                );
        }
    }

    let i = n;
    let j = m;
    let pathLength = 1;

    while (
        i > 0 ||
        j > 0
    ) {
        if (i === 0) {
            j--;
        } else if (j === 0) {
            i--;
        } else {
            const up =
                matrix[
                    (i - 1) *
                    width + j
                ];

            const left =
                matrix[
                    i *
                    width + j - 1
                ];

            const diagonal =
                matrix[
                    (i - 1) *
                    width + j - 1
                ];

            if (
                diagonal <= up &&
                diagonal <= left
            ) {
                i--;
                j--;
            } else if (
                up <= left
            ) {
                i--;
            } else {
                j--;
            }
        }

        pathLength++;
    }

    const distance =
        matrix[
            n * width + m
        ] /
        pathLength;

    const referenceMean =
        featureMean(
            reference
        );

    const recordingMean =
        featureMean(
            recording
        );

    const referenceStd =
        featureStd(
            reference,
            referenceMean
        );

    const recordingStd =
        featureStd(
            recording,
            recordingMean
        );

    let meanDifference = 0;
    let stdDifference = 0;

    for (
        let c = 0;
        c < 45;
        c++
    ) {
        meanDifference +=
            Math.abs(
                referenceMean[c] -
                recordingMean[c]
            );

        stdDifference +=
            Math.abs(
                referenceStd[c] -
                recordingStd[c]
            );
    }

    meanDifference /= 45;
    stdDifference /= 45;

    const durationRatio =
        m /
        Math.max(
            n,
            1
        );

    const durationDifference =
        Math.abs(
            1 -
            durationRatio
        );

    return new Float32Array([
        distance,
        meanDifference,
        stdDifference,
        durationDifference
    ]);
}

function aggregateComparisons(
    comparisons
) {
    const count =
        comparisons.length;

    if (!count) {
        throw new Error(
            "No reference comparisons available."
        );
    }

    const meanFeatures =
        new Float32Array(4);

    const minFeatures =
        new Float32Array(4);

    const maxFeatures =
        new Float32Array(4);

    const medianFeatures =
        new Float32Array(4);

    for (
        let c = 0;
        c < 4;
        c++
    ) {
        const values =
            new Array(count);

        for (
            let i = 0;
            i < count;
            i++
        ) {
            values[i] =
                comparisons[i][c];
        }

        let total = 0;

        for (
            let i = 0;
            i < count;
            i++
        ) {
            total += values[i];
        }

        meanFeatures[c] =
            total / count;

        values.sort(
            (a, b) =>
                a - b
        );

        minFeatures[c] =
            values[0];

        maxFeatures[c] =
            values[count - 1];

        const middle =
            Math.floor(
                count / 2
            );

        medianFeatures[c] =
            count % 2 === 0
                ? (
                    values[middle - 1] +
                    values[middle]
                ) / 2
                : values[middle];
    }

    const output =
        new Float32Array(16);

    output.set(
        meanFeatures,
        0
    );

    output.set(
        minFeatures,
        4
    );

    output.set(
        maxFeatures,
        8
    );

    output.set(
        medianFeatures,
        12
    );

    return output;
}

function scaleFeatures(
    features,
    scaler
) {
    if (
        features.length !== 16
    ) {
        throw new Error(
            `Expected 16 features, received ${features.length}.`
        );
    }

    const output =
        new Float32Array(16);

    for (
        let i = 0;
        i < 16;
        i++
    ) {
        output[i] =
            (
                features[i] -
                scaler.mean[i]
            ) /
            scaler.std[i];
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
            [1, 16]
        );

    const result =
        await session.run({
            [inputName]: input
        });

    const outputName =
        session.outputNames[0];

    return result[
        outputName
    ];
}

function convertPredictionToScore(
    output
) {
    const value =
        Number(
            output.data[0]
        );

    if (
        !Number.isFinite(value)
    ) {
        return 0;
    }

    let score = value;

    if (score >= 0 && score <= 1) {
        score *= 100;
    }

    return Math.max(
        0,
        Math.min(
            100,
            Math.round(score)
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
        } else if (score >= 50) {
            scoreMessage.textContent =
                "Pretty good. Keep practicing!";
        } else if (score >= 25) {
            scoreMessage.textContent =
                "Keep practicing and try again!";
        } else {
            scoreMessage.textContent =
                "Keep practicing and try again!";
        }
    }
}

async function prepareReferenceFeatures(
    key
) {
    if (referenceFeatures[key]) {
        return referenceFeatures[key];
    }

    const loaded =
        await loadReference(key);

    const features =
        new Array(
            loaded.length
        );

    for (
        let i = 0;
        i < loaded.length;
        i++
    ) {
        features[i] =
            extractFeatures(
                loaded[i].samples
            );
    }

    referenceFeatures[key] =
        features;

    return features;
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

        await loadAIModel("v01");
        await loadAIModel("v02");

        await Promise.all([
            loadScaler("v01"),
            loadScaler("v02"),
            loadReference("v01"),
            loadReference("v02")
        ]);

        aiStatus.textContent =
            "Extracting reference features...";

        await prepareReferenceFeatures("v01");
        await prepareReferenceFeatures("v02");

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

    if (!referenceFeatures[key]) {
        aiStatus.textContent =
            "The reference recordings are not ready.";

        return;
    }

    if (!scalers[key]) {
        aiStatus.textContent =
            "The pronunciation scaler is not ready.";

        return;
    }

    scoreButton.disabled = true;
    recordButton.disabled = true;

    try {
        aiStatus.textContent =
            "Extracting pronunciation features...";

        const recordingFeatures =
            extractFeatures(
                recordedAudio.samples
            );

        aiStatus.textContent =
            "Comparing with reference recordings...";

        const comparisons = [];

        for (
            let i = 0;
            i < referenceFeatures[key].length;
            i++
        ) {
            comparisons.push(
                compareFeatures(
                    referenceFeatures[key][i],
                    recordingFeatures
                )
            );
        }

        const features =
            aggregateComparisons(
                comparisons
            );

        if (
            features.length !== 16
        ) {
            throw new Error(
                `Expected 16 model features, received ${features.length}.`
            );
        }

        const scaled =
            scaleFeatures(
                features,
                scalers[key]
            );

        aiStatus.textContent =
            "Running pronunciation model...";

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

        if (
            event.key === "ArrowLeft"
        ) {
            previousVerse();
        }

        if (
            event.key === "ArrowRight"
        ) {
            nextVerse();
        }

        if (
            event.code === "Space"
        ) {
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