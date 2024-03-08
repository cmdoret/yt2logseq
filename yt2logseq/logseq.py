from pathlib import Path
import re
from typing import Iterator, Optional

from pytube import YouTube

from yt2logseq.srt import extract_timestamp


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

def format_timestamps(summary: str) -> Iterator[str]:
    """Format lines with embedded timestamps for logseq."""
    for line in summary.split("\n"):
        time, text = extract_timestamp(line)
        text = re.sub("^ *- *", "", text)
        if time:
            yield f"        - {{{{youtube-timestamp {time.strftime('%H:%M:%S')}}}}} {text}"
        else:
            yield f"        - {text}"


def generate_logseq_page(summary: str, meta: dict):
    if "topics" in meta:
        topics_line = "topic:: " + ', '.join([f"[[{t}]]" for t in meta["topics"]]) 
    else:
        topics_line = ""
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

- {{{{video {url}}}}}
{summary}



    """.format(
        summary="\n".join([l for l in format_timestamps(summary)]),
        topics=topics_line,
        title=meta["title"],
        description=meta["description"],
        author=meta["author"],
        tags=', '.join(meta["tags"]),
        language=meta["language"],
        date=meta["publish_date"],
        url=meta["watch_url"],
    )
    return note 
