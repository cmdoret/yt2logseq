#!/usr/bin/env python3
"""Main module for yt2logseq.

yt2logseq is an experiment to generate logseq notes from youtube videos.

It relies on:
* yt-dlp to download yt videos
* ffmpeg to extract the audio
* openai whisper to generate subtitles
* transformers to summarize the video and assign it to predefined topics.
"""

import os
from pathlib import Path
import sys
import tempfile
from typing import Iterator, Optional

from pytube import YouTube
from transformers import pipeline
import whisper
from whisper.utils import get_writer

from yt2logseq.chains import summarize_subtitles
from yt2logseq.logseq import gather_logseq_topics, generate_logseq_page

LOGSEQ_DIR = os.environ.get("LOGSEQ_DIR")
MAX_KEYWORDS=3
MAX_SUMMARY_WORDS = 200





def download_audio(url: str) -> tuple[Path, dict]:
    """Extract audio from target video and return output path+metadata."""

    yt = YouTube(url)
    audio_path = (
        yt
        .streams
        .filter(only_audio=True, file_extension='mp4')
        .order_by('abr')[-1]
        .download(output_path=tempfile.gettempdir())
    )

    meta = {
        "title": yt.title,
        "author": yt.author,
        "description": yt.description.split("\n")[0],
        "tags": yt.keywords,
        "publish_date": yt.publish_date.date() if yt.publish_date else None,
        "watch_url": yt.watch_url,
    }

    return (Path(audio_path), meta)


def extract_subtitles(audio_file: Path) -> tuple[str, str]:
    """Extract subtitles and language from an audio file."""

    model = whisper.load_model("base")
    result = model.transcribe(
        str(audio_file),
    )
    writer = get_writer(output_format='srt', output_dir=".")
    writer(
        result,
        audio_file,
        {"max_line_width":55, "max_line_count":2, "highlight_words":False}
    )
    subtitles = open(audio_file.with_suffix(".srt").name, 'r').read()
    return (subtitles, result['language'])


def assign_topics(
    text: str,
    topics: tuple[str],
    model: str="facebook/bart-large-mnli",
    max_keywords: int = MAX_KEYWORDS,
    min_score: float=0.95,
):
    """Assign topics to a text and return top keywords."""
    classifier = pipeline(
        "zero-shot-classification",
        model=model
    )
    kws = classifier(text, topics, multi_label=True)
    # Pick top <=3 fields above threshold
    return [
        kw
        for kw, score
        in zip(kws["labels"], kws["scores"])
        if score >= min_score][
        :max_keywords
    ]




if __name__ == '__main__':
    url = sys.argv[1]
    output_file = sys.argv[2]

    audio_path, meta = download_audio(url)
    subtitles, language = extract_subtitles(audio_path)
    meta['language'] = language
    if LOGSEQ_DIR:
        all_topics = gather_logseq_topics(LOGSEQ_DIR)
        meta["topics"] = assign_topics(subtitles, tuple(all_topics))
    summary = summarize_subtitles(subtitles)
    page = generate_logseq_page(summary, meta)
    with open(output_file, "w") as out:
        out.write(page)


