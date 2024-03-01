# yt2logseq

Experiment to autogenerate logseq notes from a youtube video using machine learning.

The goal of this tool is to generate logseq notes with rich metadata and AI summary from a youtube video.

It uses yt-dlp to download youtube videos, ffmpeg to extract the audio, openAI whisper to generate subtitles locally and transformers (BART models) to generate a summary of the subtitles.
In additionl the tools can read an existing logseq graph to gather all `topic::` already in use and perform zero-shot classification to assign the input video to up to 3 existing topics.

## Installation

Install with poetry

```sh
poetry install
```

## Usage

```sh
# Without logseq
python yt2logseq/main.py <video-url> <output.md>

# With logseq
LOGSEQ_DIR=path/to/logseq/graph python yt2logseq/main.py <video-url> <output.md>
```
