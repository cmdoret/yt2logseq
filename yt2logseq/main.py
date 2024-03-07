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

from yt2logseq.chains import make_mapreduce_chain
from yt2logseq.srt import StampedSRTLoader

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



def download_audio(url: str) -> tuple[Path, YouTube]:
    """Extract audio from target video and return output path+metadata."""

    yt = YouTube(url)
    audio_path = (
        yt
        .streams
        .filter(only_audio=True, file_extension='mp4')
        .order_by('abr')[-1]
        .download(output_path=tempfile.gettempdir())
    )

    return (Path(audio_path), yt)


def extract_subtitles(audio_file: Path) -> tuple[str, str]:
    """Extract subtitles and language from an audio file."""

    model = whisper.load_model("base")
    result = model.transcribe(str(audio_file))
    return (str(result["text"]), result['language'])


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


def generate_logseq_page(summary: str, yt: YouTube, topics: Optional[list[str]]=None):
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
language:: {language}
link:: {url}
{topics}

- # Summary
        - {summary}

- # Video
        - {{{{video {url}}}}}


    """.format(
        summary=summary,
        topics=topics_line,
        title=yt.title,
        description=yt.description.split('\n')[0],
        author=yt.author,
        tags=', '.join(yt.keywords),
        language=yt.language,
        date=yt.publish_date.date(),
        url=yt.watch_url,
    )
    return note 


if __name__ == '__main__':
    url = sys.argv[1]
    output_file = sys.argv[2]

    audio_path, yt = download_audio(url)
    subtitles, language = extract_subtitles(audio_path)
    if LOGSEQ_DIR:
        all_topics = gather_logseq_topics(LOGSEQ_DIR)
        topics = assign_topics(subtitles, tuple(all_topics))
    else:
        topics = None
    docs = StampedSRTLoader(file_path="", data=subtitles).load()
    mapreduce_chain = make_mapreduce_chain()
    summary = mapreduce_chain.invoke({'input_documents': docs})
    yt.language = language
    page = generate_logseq_page(summary, yt, topics)
    with open(output_file, "w") as out:
        out.write(page)


