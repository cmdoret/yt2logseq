
MAP_TMPL = """The following are time-stamped captions in srt format.
{docs}
Based on these captions, please select and summarize the important information as bullet points in markdown format, making sure to always include the original timestamp from the original caption where the information comes from.
Markdown summary:"""

REDUCE_TMPL = """The following is set of summaries based on a video:
{docs}
Distill these into a final, consolidated summary. Discard redundant and irrelevant information. Always retain the original time-stamps.
Markdown summary:"""
