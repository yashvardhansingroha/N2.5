"""Optional atmospheric video masthead for the PM2.5 Streamlit dashboard.

This module is intentionally self-contained. Set AQ_VIDEO_HERO=0, remove this
file, or remove its asset folder to make the main app use its built-in CSS
masthead without affecting data collection or modeling.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st


ASSET_DIR = Path(__file__).parent / "static" / "atmospheric-hero"
VIDEO_PATH = ASSET_DIR / "city-smog.mp4"
POSTER_PATH = ASSET_DIR / "city-smog-poster.jpg"
DISABLED_VALUES = {"0", "false", "no", "off"}


def feature_enabled() -> bool:
    return os.getenv("AQ_VIDEO_HERO", "1").strip().casefold() not in DISABLED_VALUES


def assets_available() -> bool:
    return POSTER_PATH.is_file()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .aqv-root {
            position: absolute;
            inset: 0;
            z-index: 3;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 1.7rem 1.8rem 1.35rem;
            color: #f3f7f5;
            background: rgba(6, 11, 13, 0.62);
            pointer-events: none;
        }
        .aqv-copy { max-width: 720px; }
        .aqv-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            color: #62d6d6;
            font-size: 0.73rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .aqv-kicker::before {
            content: "";
            width: 18px;
            height: 5px;
            background: #ff6b5f;
        }
        .aqv-root h1 {
            max-width: 700px;
            margin: 0.75rem 0 0.55rem;
            color: #f3f7f5;
            font-size: 2.55rem;
            line-height: 1.04;
            font-weight: 760;
            letter-spacing: 0;
            text-shadow: 0 2px 16px rgba(0, 0, 0, 0.7);
        }
        .aqv-root h1 span { color: #62d6d6; }
        .aqv-root p {
            margin: 0;
            color: rgba(243, 247, 245, 0.78);
            font-size: 0.96rem;
            line-height: 1.5;
        }
        .aqv-readouts {
            display: flex;
            gap: 0.65rem;
            align-items: stretch;
        }
        .aqv-readout {
            min-width: 126px;
            padding: 0.62rem 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            background: rgba(9, 15, 17, 0.7);
            backdrop-filter: blur(10px);
        }
        .aqv-readout small {
            display: block;
            color: rgba(243, 247, 245, 0.6);
            font-size: 0.62rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .aqv-readout strong {
            display: block;
            margin-top: 0.12rem;
            color: #f3f7f5;
            font-size: 1rem;
            font-weight: 650;
            line-height: 1.2;
        }
        .aqv-flow {
            position: absolute;
            left: -32%;
            z-index: 4;
            width: 28%;
            height: 2px;
            background: #62d6d6;
            opacity: 0.42;
            animation: aqv-drift 12s linear infinite;
        }
        .aqv-flow::after {
            content: "";
            position: absolute;
            right: -28px;
            top: -2px;
            width: 20px;
            height: 6px;
            background: #ff6b5f;
        }
        .aqv-flow.f1 { top: 25%; }
        .aqv-flow.f2 { top: 68%; animation-delay: -7s; animation-duration: 16s; }
        @keyframes aqv-drift {
            from { transform: translateX(0); }
            to { transform: translateX(500%); }
        }
        @media (max-width: 700px) {
            .aqv-root { padding: 1.45rem 1.2rem 1.05rem; }
            .aqv-root h1 { font-size: 2rem; line-height: 1.08; }
            .aqv-root p { max-width: 310px; font-size: 0.88rem; }
            .aqv-readouts { gap: 0.42rem; }
            .aqv-readout { min-width: 0; flex: 1; padding: 0.52rem; }
            .aqv-readout strong { font-size: 0.82rem; }
            .aqv-flow.f1 { top: 13%; }
            .aqv-flow.f2 { top: 92%; }
        }
        @media (prefers-reduced-motion: reduce) {
            .aqv-flow { display: none; }
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) {
            position: relative;
            height: 320px !important;
            min-height: 320px !important;
            overflow: hidden;
            padding: 0 !important;
            margin: 0 0 1.8rem;
            border: 1px solid #395158;
            border-radius: 6px;
            background: #101719;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) > div > div[data-testid="stVerticalBlock"] {
            position: static;
            gap: 0;
            padding: 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) > div > div[data-testid="stVerticalBlock"]
        > div[data-testid="element-container"]:has([data-testid="stImage"]),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) > div > div[data-testid="stVerticalBlock"]
        > div[data-testid="element-container"]:has([data-testid="stVideo"]),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) > div > div[data-testid="stVerticalBlock"]
        > div[data-testid="element-container"]:has(.aqv-root) {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            margin: 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) [data-testid="stImage"] { position: absolute; inset: 0; z-index: 0; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) [data-testid="stVideo"] { position: absolute; inset: 0; z-index: 1; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) div[data-testid="element-container"]:has(.aqv-root) { z-index: 3; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) [data-testid="stImage"] img,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) [data-testid="stVideo"] video {
            width: 100% !important;
            height: 320px !important;
            object-fit: cover;
            object-position: 50% 52%;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > div[data-testid="stVerticalBlock"]
            > div[data-testid="element-container"] .aqv-root
        ) video::-webkit-media-controls {
            display: none !important;
        }
        @media (max-width: 700px) {
            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                > div > div[data-testid="stVerticalBlock"]
                > div[data-testid="element-container"] .aqv-root
            ),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                > div > div[data-testid="stVerticalBlock"]
                > div[data-testid="element-container"] .aqv-root
            ) [data-testid="stImage"] img,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                > div > div[data-testid="stVerticalBlock"]
                > div[data-testid="element-container"] .aqv-root
            ) [data-testid="stVideo"] video {
                height: 360px !important;
                min-height: 360px !important;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            div[data-testid="stVerticalBlockBorderWrapper"]:has(
                > div > div[data-testid="stVerticalBlock"]
                > div[data-testid="element-container"] .aqv-root
            ) [data-testid="stVideo"] { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_atmospheric_hero(
    city: str, motion_enabled: bool = True,
    pollutant: str = "PM2.5", threshold: float = 60.0,
) -> bool:
    """Render the optional hero and return whether it replaced the fallback."""
    if not feature_enabled() or not assets_available():
        return False

    _inject_styles()
    safe_city = html.escape(city.strip() or "Selected city")
    with st.container(height=320, border=False):
        st.image(str(POSTER_PATH), use_column_width=True)
        if motion_enabled and VIDEO_PATH.is_file():
            st.video(str(VIDEO_PATH), autoplay=True, muted=True, loop=True)
        st.markdown(
            f"""
            <div class="aqv-root" data-motion="{'on' if motion_enabled else 'off'}">
                <span class="aqv-flow f1" aria-hidden="true"></span>
                <span class="aqv-flow f2" aria-hidden="true"></span>
                <div class="aqv-copy">
                    <div class="aqv-kicker">Live atmospheric intelligence</div>
                    <h1>Urban Air Quality<br><span>Early Warning System</span></h1>
                    <p>{html.escape(pollutant)} risk, before tomorrow arrives.</p>
                </div>
                <div class="aqv-readouts">
                    <div class="aqv-readout"><small>Location</small><strong>{safe_city}</strong></div>
                    <div class="aqv-readout"><small>Training cap</small><strong>5 years</strong></div>
                    <div class="aqv-readout"><small>Alert level</small><strong>{threshold:.0f} ug/m3</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return True
