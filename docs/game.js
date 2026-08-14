const BASE_URL = "https://raw.githubusercontent.com/koditra/Anukrama/main";

const TOTAL_VERSES = 20;

let currentVerse = 1;
let currentAudio = null;

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

function createWav(buffer) {
    const pcm = new Int16Array(buffer);

    const sampleRate = 48000;
    const channels = 1;
    const bitsPerSample = 16;
    const blockAlign = channels * bitsPerSample / 8;
    const byteRate = sampleRate * blockAlign;

    const wav = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(wav);

    function writeString(offset, value) {
        for (let i = 0; i < value.length; i++) {
            view.setUint8(offset + i, value.charCodeAt(i));
        }
    }

    writeString(0, "RIFF");
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeString(8, "WAVE");

    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    writeString(36, "data");
    view.setUint32(40, pcm.length * 2, true);

    for (let i = 0; i < pcm.length; i++) {
        view.setInt16(44 + i * 2, pcm[i], true);
    }

    return new Blob([wav], {
        type: "audio/wav"
    });
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

        const response = await fetch(`${BASE_URL}/${path}`);

        if (!response.ok) {
            throw new Error(`Audio not found: ${path}`);
        }

        const buffer = await response.arrayBuffer();
        const wav = createWav(buffer);
        const url = URL.createObjectURL(wav);

        const audio = new Audio(url);

        currentAudio = audio;
        audio.volume = 1;

        audio.addEventListener("ended", () => {
            URL.revokeObjectURL(url);

            if (currentAudio === audio) {
                currentAudio = null;
            }
        });

        audio.addEventListener("error", () => {
            URL.revokeObjectURL(url);

            if (currentAudio === audio) {
                currentAudio = null;
            }
        });

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
    progressText.textContent = `${id} / ${TOTAL_VERSES}`;

    quarterDevanagari.textContent = "Loading...";
    quarterEnglish.textContent = "Loading...";

    verseDevanagari.textContent = "";
    verseEnglish.textContent = "";

    answer.classList.add("hidden");

    revealButton.textContent = "Reveal verse";

    audioButton.disabled = true;
    audioButton.textContent = "Loading...";

    const card = document.querySelector(".practice-card");

    if (card) {
        card.classList.remove("verse-enter");

        void card.offsetWidth;

        card.classList.add("verse-enter");
    }

    try {
        const meta = await fetchJson(`meta/verse_${id}.json`);

        const devanagari = await fetchText(
            `verses/devanagari/verse_${id}.txt`
        );

        const english = await fetchText(
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
            const audioNumber = String(id).padStart(2, "0");

            await playPcm(
                `audio/chapter_15/v${audioNumber}_prompt.pcm`
            );
        }
    } catch (error) {
        console.error(error);

        quarterDevanagari.textContent =
            "Unable to load verse";

        quarterEnglish.textContent =
            "Something went wrong while loading this verse.";

        verseDevanagari.textContent = "";
        verseEnglish.textContent = "";

        audioButton.disabled = true;
        audioButton.textContent = "Audio unavailable";
    }
}

function revealVerse() {
    answer.classList.toggle("hidden");

    if (answer.classList.contains("hidden")) {
        revealButton.textContent = "Reveal verse";
    } else {
        revealButton.textContent = "Hide verse";
    }
}

async function playVerseAudio() {
    const audioNumber = String(currentVerse).padStart(2, "0");

    audioButton.disabled = true;
    audioButton.textContent = "Playing...";

    const success = await playPcm(
        `audio/chapter_15/v${audioNumber}.pcm`
    );

    audioButton.disabled = false;

    if (success) {
        audioButton.textContent = "Play verse";
    } else {
        audioButton.textContent = "Audio unavailable";
    }
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
            Math.floor(Math.random() * TOTAL_VERSES) + 1;
    } while (random === currentVerse);

    await loadVerse(random);
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

document.addEventListener("keydown", event => {
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

    if (event.key.toLowerCase() === "r") {
        randomVerse();
    }
});

loadVerse(1);
