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
import re
import sys
from typing import Iterator, Optional

from transformers import pipeline
import whisper
import yt_dlp

LOGSEQ_DIR=os.environ.get("LOGSEQ_DIR")
MAX_KEYWORDS=3
MAX_SUMMARY_WORDS = 200


def parse_topics_line(line: str) -> Iterator[str]:
    """Parse a property line with topics in a line."""
    # functionally equivalent to ripgrep -f "topic::" $LOGSEQ_DIR
    values = line.split("::")[1].strip()
    for value in values.split(","):
        yield value.strip().strip("[]")


def gather_logseq_topics(logseq_dir: Path) -> set[str]:
    """Gather all topics from existing logseq notes."""
    topics = set()
    for file in Path(logseq_dir).rglob("*.md"):
        with open(file, "r") as f:
            for line in f:
                if "topic::" in line:
                    line_topics = parse_topics_line(line)
                    topics |= set(line_topics)

    return topics


def find_yt_dlp_file(filename: str) -> str:
    """Given input filename, find the corresponding yt-dlp file.
    This is meant to account for inconsistent filename generation by yt-dlp.
    """
    pattern = re.search(
        r'\[[-a-zA-Z0-9]*\]\.mp3', filename
    ).group(0)
    # NOTE: Skip first bracket to avoid problem with glob
    path = [p for p in Path().rglob(f"*{pattern[1:]}")][0]
    return str(path)


def download_audio(url: str) -> tuple[Path, dict]:
    """Extract audio from target video and return output path+metadata."""
    ydl_opts = {
    'format': 'mp3/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        info = ydl.sanitize_info(info)
        ydl.download([url])
        # Filename generation broken (inconsistent spaces)
        _filename = ydl.evaluate_outtmpl(
            ydl.params['outtmpl']['default'],
            info,
        ).removesuffix(".webm") + ".mp3"
        filename = find_yt_dlp_file(_filename)

    return (Path(filename), info)


def extract_subtitles(audio_file: Path) -> str:
    """Extract subtitles from an audio file."""

    model = whisper.load_model("base")
    result = model.transcribe(str(audio_file))
    return str(result["text"])


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


def generate_summary(
    text: str,
    model="sshleifer/distilbart-cnn-12-6",
    min_length=10,
    max_length=MAX_SUMMARY_WORDS,
) -> str:
    """Generate a summary of text."""

    summarizer = pipeline(
        "summarization",
        model=model,
        min_length=min_length,
        max_length=max_length,
    )
    summary = (
        summarizer(
            text,
            truncation=True
        )[0]["summary_text"]
        .strip(' ')
        .replace(' .', '.')
    )
    return summary


def generate_logseq_page(summary: str, meta: dict, topics: Optional[list[str]]=None):
    date = meta['upload_date']
    if not topics:
        topics_line = ""
    else:
        topics_line = "topic:: " + ', '.join([f"[[{t}]]" for t in topics]) 
    note = """
type:: [[video]]
title:: {title}
author:: [[{author}]]
date-published:: [[{date}]]
description:: {description}
tag:: {tags}
category:: {categories}
language:: {language}
link:: {original_url}
{topics}

- # Summary
        - {summary}

- # Video
        - {{{{video {original_url}}}}}


    """.format(
        summary=summary,
        topics=topics_line,
        title=meta['title'],
        description=meta['description'].split('\n')[0],
        author=meta['channel'],
        tags=', '.join(meta['tags']),
        categories=', '.join(meta['categories']),
        language=meta['language'],
        date=f"{date[:4]}-{date[4:6]}-{date[6:8]}",
        original_url=meta["original_url"],
    )
    return note 


if __name__ == '__main__':
    url = sys.argv[1]
    output_file = sys.argv[2]

    all_topics = gather_logseq_topics(LOGSEQ_DIR) if LOGSEQ_DIR else None
    audio_path, yt_meta = download_audio(url)
    sub_file = extract_subtitles(audio_path)

    if all_topics:
        topics = assign_topics(sub_file, tuple(all_topics))
    else:
        topics = None
    summary = generate_summary(sub_file)
    page = generate_logseq_page(summary, yt_meta, topics)
    with open(output_file, "w") as out:
        out.write(page)


