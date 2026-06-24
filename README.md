# Anukrama

A Nintendo Switch homebrew game that helps users memorize and practice chanting Bhagavad Gita verses. The game provides the previous quarter of a verse as a clue and challenges the player to recall and chant the next section from memory.

## Design Overview

### Main Menu

<img width="1456" height="819" alt="Screenshot 2026-06-24 at 4 49 26 PM" src="https://github.com/user-attachments/assets/5dd8ee09-9433-4d2a-966d-b8030ffe9c2c" />

### Verse Practice Screen

<img width="1456" height="819" alt="Screenshot 2026-06-24 at 4 49 48 PM" src="https://github.com/user-attachments/assets/a8b00f71-26c2-4736-8aa9-3fe914a48bb2" />

<img width="1456" height="819" alt="Screenshot 2026-06-24 at 4 50 40 PM" src="https://github.com/user-attachments/assets/71e68620-6701-46a4-ab26-2f32ab51c49e" />

## Features

* Chapter 15 verse practice
* Audio prompts for each verse
* Sanskrit (Devanagari) verse audio
* English transliteration (with unicode-fallback)
* Controller-based navigation
* A lightweight Nintendo Switch homebrew application
* Designed for memorization and chanting practice

## Verse Content

Current chapter included:

| Chapter                  | Verses |
| ------------------------ | ------ |
| Bhagavad Gita Chapter 15 | 1–20   |

## Audio System

Each verse includes:

* Full verse audio
* Prompt audio for guided practice
* Fast loading using preprocessed PCM files

Audio files are stored in:

```text
audio/chapter_15/
```

## Metadata

Verse information such as the lyrics, progression, and prompts are stored in:

```text
meta/
```

## Files

```text
source/  → main application source code
audio/   → PCM audio recordings
meta/    → verse metadata
verses/  → Sanskrit and English verse text
build/   → generated build files (you will get these after building/see next step)
```

## Building

Requirements:

* devkitPro
* libnx
* Nintendo Switch homebrew environment

Build:

```bash
make
```

Output:

```text
Anukrama.nro
```

## Controls

| Button        | Action            |
| ------------- | ----------------- |
| A             | Select / Continue |
| B             | Back              |
| D-Pad / Stick | Navigate menus    |
| +             | Exit              |

## Future Plans

* Additional Bhagavad Gita chapters
* Chanting score system
* Voice recognition
* Practice statistics
* A Better Multiplayer VS mode
* Improved UI and animations
* Progress tracking

## Notes

Anukrama was created as a way to make Bhagavad Gita memorization more interactive and accessible through Nintendo Switch homebrew. The goal is to combine traditional chanting practice with a modern gaming experience.

## About

A Nintendo Switch homebrew game that helps users memorize and practice Bhagavad Gita verses by challenging them to chant the next verse section from memory using guided prompts.
