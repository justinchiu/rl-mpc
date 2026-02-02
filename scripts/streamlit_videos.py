from __future__ import annotations

from pathlib import Path
import re

import streamlit as st


def _sort_key(path: Path):
    match = re.search(r"_step(\\d+)", path.name)
    if match:
        return int(match.group(1))
    return path.name


st.set_page_config(page_title="SB3 Training Videos", layout="centered")
st.title("SB3 Training Videos")

video_dir = st.text_input("Video directory", "videos/sb3")
max_videos = st.number_input("Max videos", min_value=1, value=20, step=1)

root = Path(video_dir).expanduser()
if not root.exists():
    st.warning(f"Directory not found: {root}")
else:
    videos = sorted(root.glob("*.mp4"), key=_sort_key)
    if not videos:
        st.info("No .mp4 files found.")
    else:
        for path in videos[: int(max_videos)]:
            st.caption(path.name)
            st.video(str(path))
