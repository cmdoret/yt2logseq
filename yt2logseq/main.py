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

import openai
from pytube import YouTube
from transformers import pipeline
import whisper
from whisper.utils import get_writer

from yt2logseq.chains import summarize_subtitles
from yt2logseq.logseq import gather_logseq_topics, generate_logseq_page

LOGSEQ_DIR = os.environ.get("LOGSEQ_DIR")
MAX_KEYWORDS=3
MAX_SUMMARY_WORDS = 200





def extract_subtitles(url: str) -> tuple[str, dict]:
    """Download audio and extract subtitles and metadata
    from target video return output path+metadata. subtitles
    are generated using openai's whisper model.

    Parameters
    ----------
    url
        The url of the video to process.

    Returns
    -------
    tuple[str, dict]
        The subtitles text (srt format) and video metadata.
    """

    yt = YouTube(url)
    audio_path = Path(
        yt
        .streams
        .filter(only_audio=True, file_extension='mp4')
        .order_by('abr')[-1]
        .download(output_path=tempfile.gettempdir())
    )

    model = whisper.load_model("base")
    result = model.transcribe(
        str(audio_path),
    )
    writer = get_writer(output_format='srt', output_dir=str(audio_path.parent))
    writer(
        result,
        audio_path,
        {"max_line_width":55, "max_line_count":2, "highlight_words":False}
    )
    subtitle_path = audio_path.with_suffix(".srt")
    subtitles = open(subtitle_path, 'r').read()

    meta = {
        "title": yt.title,
        "author": yt.author,
        "description": yt.description.split("\n")[0],
        "tags": yt.keywords,
        "publish_date": yt.publish_date.date() if yt.publish_date else None,
        "watch_url": yt.watch_url,
        "language": result['language'],
    }
    os.remove(audio_path)
    os.remove(subtitle_path)

    return (subtitles, meta)



def assign_topics(
    text: str,
    topics: tuple[str],
    max_keywords: int = MAX_KEYWORDS,
    ) -> list[str]:
    """Assign topics to a text and return top keywords."""
    client = openai.OpenAI()
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Here is some text:\n{text}\nPlease assign a topic among the following: {topics}"
            }
        ],
        model="gpt-3.5-turbo",
        logprobs=True,
        top_logprobs=max_keywords + 5,
    )
    try:
        next_token_probs = completion.choices[0].logprobs.content[0].top_logprobs
    except (TypeError, IndexError):
        return []
    return [
        choice.token 
        for choice
        in sorted(next_token_probs, key=lambda x: x.logprob)
        if choice.token in topics
    ][:max_keywords]



if __name__ == '__main__':
    url = sys.argv[1]
    output_file = sys.argv[2]

    subtitles, meta = extract_subtitles(url)
    if LOGSEQ_DIR:
        all_topics = gather_logseq_topics(LOGSEQ_DIR)
        meta["topics"] = assign_topics(f'{meta["title"]}\n{meta["description"]}', tuple(all_topics))
    summary = summarize_subtitles(subtitles)
    page = generate_logseq_page(summary, meta)
    with open(output_file, "w") as out:
        out.write(page)


