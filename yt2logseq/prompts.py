
MAP_TMPL = """The following are time-stamped captions in srt format.
{docs}
Based on these captions, please select and summarize the important information as bullet points in markdown format, making sure to include the original timestamp from the original caption where the information comes from..
Markdown summary:"""

REDUCE_TMPL = """The following is set of summaries based on a video:
{docs}
Take these and distill it into a final, consolidated summary. Discard information that is not relevant. Retain the original time-stamps whenever it makes sense.
Markdown summary:"""
