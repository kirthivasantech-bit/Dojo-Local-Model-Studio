import time
from datetime import datetime, timedelta

import streamlit as st

from core.downloader import DOWNLOAD_MANAGER
from core.eval import (
    complete_once,
    load_llama,
    stream_chat,
    unload_model,
)
from core.registry import (
    PRESET_MODELS,
    build_download_url,
    delete_model,
    disk_usage,
    list_local_models,
    list_repo_ggufs,
    search_gguf_models,
)
from core.sessions import (
    DEFAULT_SETTINGS,
    export_session_markdown,
    load_sessions,
    new_session,
    save_sessions,
    title_from_message,
)
from core.voice import (
    AVAILABLE_VOICES,
    audio_autoplay_html,
    download_voice,
    get_engine,
    is_voice_downloaded,
    list_downloaded_voices,
)


# ==================================================================
# PAGE CONFIG
# ==================================================================

st.set_page_config(
    page_title="Dojo-Local-Model-Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================================
# PRODUCTION CSS (UI UPGRADE)
# ==================================================================

def inject_css():
    st.markdown(
        """
        <style>
        /* ============================================================
           DESIGN TOKENS (Premium Dark Theme)
           ============================================================ */
        :root {
            --bg-main:         #09090b; /* Deep Zinc / Pure dark */
            --bg-sidebar:      #121214;
            --bg-elevated:     #18181b;
            --bg-hover:        #27272a;
            --bg-active:       #3f3f46;
            --bg-input:        #18181b;
            --bg-user-bubble:  #27272a;

            --border-subtle:   rgba(255, 255, 255, 0.06);
            --border-default:  rgba(255, 255, 255, 0.12);
            --border-strong:   rgba(255, 255, 255, 0.20);

            --text-primary:    #fafafa;
            --text-secondary:  #a1a1aa;
            --text-tertiary:   #71717a;
            --text-muted:      #52525b;

            --accent:          #ffffff;
            --accent-blue:     #3b82f6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --accent-glow:     rgba(99, 102, 241, 0.4);

            --radius-xs: 6px;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            --radius-full: 9999px;

            --shadow-sm: 0 2px 4px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 16px rgba(0,0,0,0.4);
            --shadow-lg: 0 16px 32px rgba(0,0,0,0.5);
            --shadow-glow: 0 0 15px var(--accent-glow);

            --ease: cubic-bezier(0.23, 1, 0.32, 1);
            --t-fast: 150ms var(--ease);
            --t-base: 250ms var(--ease);
        }

        /* ============================================================
           GLOBAL RESET & TYPOGRAPHY
           ============================================================ */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: var(--bg-main) !important;
            color: var(--text-primary);
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 15px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        /* Hide Streamlit Chrome */
        [data-testid="stHeader"], [data-testid="stToolbar"], 
        [data-testid="stDecoration"], .stAppDeployButton, footer {
            display: none !important;
        }

        /* Fluid Main Container */
        .block-container, [data-testid="stMainBlockContainer"] {
            max-width: 900px !important;
            padding: 3rem 2rem 10rem 2rem !important;
            margin: 0 auto;
        }

        /* Scrollbars */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: var(--radius-full); }
        ::-webkit-scrollbar-thumb:hover { background: var(--border-strong); }

        /* Typography */
        .page-title {
            font-size: 28px !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
            color: var(--text-primary) !important;
            margin-bottom: 0.25rem !important;
        }
        .page-sub {
            font-size: 14px !important;
            color: var(--text-tertiary) !important;
            margin-bottom: 2rem !important;
            font-weight: 400 !important;
        }

        /* ============================================================
           SIDEBAR STYLING
           ============================================================ */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-subtle) !important;
            min-width: 280px !important;
            max-width: 280px !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 1.5rem 1rem !important;
        }
        .side-title {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }
        .side-sub {
            font-size: 12px;
            color: var(--text-tertiary);
            margin-bottom: 1.5rem;
        }
        .side-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin: 1.5rem 0 0.5rem 0.5rem;
        }

        /* Sidebar Buttons / Session List */
        [data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 0.5rem 0.75rem !important;
            justify-content: flex-start !important;
            text-align: left !important;
            width: 100% !important;
            height: auto !important;
            min-height: 40px !important;
            font-size: 14px !important;
            transition: all var(--t-fast) !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--bg-hover) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] .stButton > button:active {
            transform: scale(0.98) !important;
        }
        /* Primary Button in Sidebar (New Chat) */
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: var(--text-primary) !important;
            color: var(--bg-main) !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            justify-content: center !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            opacity: 0.9 !important;
            transform: translateY(-1px) !important;
        }

        /* Mode Radio (Chat, Voice, Models) */
        [data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: transparent;
            padding: 0.5rem 1rem !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-secondary) !important;
            transition: all var(--t-fast) !important;
            font-size: 14px !important;
            border: 1px solid transparent !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: var(--bg-hover) !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-default) !important;
            font-weight: 500 !important;
            box-shadow: var(--shadow-sm);
        }

        /* Sidebar Popovers (Rename/Export/Delete Menu) */
        [data-testid="stPopoverBody"], [role="dialog"][data-testid*="Popover"] {
            background: var(--bg-elevated) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow-lg) !important;
            padding: 0.5rem !important;
        }
        [data-testid="stPopoverBody"] .stButton > button {
            background: transparent !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.5rem 0.75rem !important;
            color: var(--text-primary) !important;
        }

        /* ============================================================
           CHAT MESSAGES & ANIMATIONS
           ============================================================ */
        @keyframes messageFadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            padding: 1.5rem 0 !important;
            animation: messageFadeIn var(--t-base) forwards;
        }
        /* Hide avatars */
        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] { display: none !important; }
        [data-testid="stChatMessage"] > div { gap: 0 !important; }

        /* User Message */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            flex-direction: row;
            justify-content: flex-end;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
            background: var(--bg-user-bubble) !important;
            border: 1px solid var(--border-subtle);
            border-radius: 20px 20px 4px 20px !important;
            padding: 0.8rem 1.2rem !important;
            max-width: 80% !important;
            margin-left: auto;
            color: var(--text-primary) !important;
            box-shadow: var(--shadow-sm);
        }

        /* Assistant Message */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
            padding: 0 1rem 0 0 !important;
            max-width: 100% !important;
            color: var(--text-primary) !important;
        }

        /* Message Content Typography */
        [data-testid="stChatMessage"] p { margin-bottom: 1rem !important; }
        [data-testid="stChatMessage"] p:last-child { margin-bottom: 0 !important; }
        [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 {
            color: var(--text-primary) !important;
            margin: 2rem 0 1rem 0 !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }
        [data-testid="stChatMessage"] h1 { font-size: 24px !important; }
        [data-testid="stChatMessage"] h2 { font-size: 20px !important; }
        [data-testid="stChatMessage"] h3 { font-size: 18px !important; }
        
        [data-testid="stChatMessage"] pre {
            background: #000000 !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem !important;
            margin: 1.5rem 0 !important;
        }
        [data-testid="stChatMessage"] code { color: #e2e8f0 !important; font-family: "JetBrains Mono", monospace !important; font-size: 13px !important;}
        [data-testid="stChatMessage"] :not(pre) > code {
            background: var(--bg-hover) !important;
            padding: 0.2rem 0.4rem !important;
            border-radius: var(--radius-xs) !important;
            color: #fb7185 !important;
        }

        /* Message Action Buttons (Copy, Retry, Delete) */
        [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] {
            gap: 8px !important;
            margin-top: 12px !important;
            opacity: 0;
            transition: opacity var(--t-fast);
        }
        [data-testid="stChatMessage"]:hover [data-testid="stHorizontalBlock"] {
            opacity: 1;
        }
        [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] .stButton > button {
            background: transparent !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-sm) !important;
            width: 32px !important; height: 32px !important;
            color: var(--text-tertiary) !important;
        }
        [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
            background: var(--bg-hover) !important;
            border-color: var(--border-default) !important;
            color: var(--text-primary) !important;
        }

        /* ============================================================
           PREMIUM CHAT INPUT
           ============================================================ */
        [data-testid="stChatInput"] {
            background: transparent !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stChatInput"] > div, [data-testid="stChatInput"] > div > div {
            background-color: var(--bg-input) !important;
            border-radius: var(--radius-xl) !important;
            border: 1px solid var(--border-default) !important;
            box-shadow: var(--shadow-md) !important;
            transition: all var(--t-base) !important;
        }
        [data-testid="stChatInput"] > div {
            padding: 0.5rem 0.5rem 0.5rem 1.5rem !important;
            min-height: 60px !important;
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 1px var(--accent-blue), var(--shadow-glow) !important;
        }
        [data-testid="stChatInput"] textarea {
            color: var(--text-primary) !important;
            font-size: 15px !important;
            padding: 10px 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-tertiary) !important;
        }
        /* Send Button */
        [data-testid="stChatInput"] button {
            background: var(--text-primary) !important;
            color: var(--bg-main) !important;
            border-radius: 50% !important;
            width: 44px !important; height: 44px !important;
            min-width: 44px !important; max-width: 44px !important;
            border: none !important;
            transition: all var(--t-fast) !important;
        }
        [data-testid="stChatInput"] button:hover {
            transform: scale(1.05) !important;
            opacity: 0.9 !important;
        }
        [data-testid="stChatInput"] button:disabled {
            background: var(--bg-hover) !important;
            color: var(--text-muted) !important;
            transform: scale(1) !important;
        }

        /* ============================================================
           COMPONENTS (Empty States, Cards, Selects)
           ============================================================ */
        .empty-state {
            text-align: center;
            padding: 5rem 1rem;
            animation: messageFadeIn var(--t-base) forwards;
        }
        .empty-state h2 {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
            letter-spacing: -0.03em;
        }
        .empty-state p {
            font-size: 16px;
            color: var(--text-tertiary);
            margin-bottom: 3rem;
        }

        /* Suggestion Cards */
        .sugg-card {
            background: rgba(255,255,255, 0.02);
            border: 1px solid var(--border-default);
            border-radius: var(--radius-lg);
            padding: 2px;
            transition: all var(--t-base);
        }
        .sugg-card:hover {
            background: rgba(255,255,255, 0.04);
            border-color: var(--border-strong);
            transform: translateY(-2px);
            box-shadow: var(--shadow-sm);
        }
        .sugg-card .stButton > button {
            background: transparent !important;
            border: none !important;
            color: var(--text-secondary) !important;
            padding: 1rem 1.25rem !important;
            min-height: 80px !important;
            height: 100% !important;
            justify-content: flex-start !important;
            text-align: left !important;
            font-size: 14px !important;
            white-space: normal !important;
        }
        .sugg-card .stButton > button:hover {
            color: var(--text-primary) !important;
        }

        /* Generic Inputs & Selects */
        [data-baseweb="select"] > div, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
            background: var(--bg-elevated) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            box-shadow: none !important;
            transition: border var(--t-fast) !important;
        }
        [data-baseweb="select"] > div:hover, [data-testid="stTextInput"] input:focus {
            border-color: var(--border-strong) !important;
        }
        
        /* Tabs */
        [data-testid="stTabs"] [role="tablist"] {
            border-bottom: 1px solid var(--border-subtle) !important;
            gap: 1.5rem !important;
        }
        [data-testid="stTabs"] button[role="tab"] {
            color: var(--text-tertiary) !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            padding: 0.75rem 0 !important;
            border-bottom: 2px solid transparent !important;
            background: transparent !important;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: var(--text-primary) !important;
            border-bottom-color: var(--text-primary) !important;
        }

        /* Cards & Badges */
        .card-title { font-size: 16px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }
        .card-sub { font-size: 13px; color: var(--text-tertiary); font-family: "JetBrains Mono", monospace; }
        .badge {
            display: inline-flex; align-items: center; padding: 2px 8px; border-radius: var(--radius-full);
            font-size: 11px; font-weight: 600; text-transform: uppercase; margin-left: 8px;
        }
        .badge-gray { background: rgba(255,255,255,0.08); color: var(--text-secondary); border: 1px solid var(--border-default); }

        /* General Buttons (Download / Regular action) */
        .block-container .stButton > button {
            background: var(--bg-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-md) !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 500 !important;
            transition: all var(--t-fast) !important;
        }
        .block-container .stButton > button:hover {
            background: var(--bg-hover) !important;
            border-color: var(--border-strong) !important;
        }
        .block-container .stButton > button[kind="primary"] {
            background: var(--accent-gradient) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        }
        .block-container .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
            filter: brightness(1.1);
        }

        /* Status Pills (Voice Tab) */
        .status-pill {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 16px; border-radius: var(--radius-full);
            font-size: 13px; font-weight: 600; margin-bottom: 1.5rem;
        }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; }

        /* Expander */
        [data-testid="stExpander"] {
            background: transparent !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-lg) !important;
            overflow: hidden !important;
        }
        [data-testid="stExpander"] summary { padding: 1rem !important; color: var(--text-primary) !important; }
        [data-testid="stExpander"] summary:hover { background: rgba(255,255,255,0.02) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================
# HELPERS
# ==================================================================

def _render_progress(task):
    if task.status in ("running", "paused"):
        pct = task.progress()
        mb_done = task.downloaded // (1024 * 1024)
        mb_total = task.total // (1024 * 1024) if task.total else 0
        st.progress(
            pct,
            text=f"{mb_done} MB / {mb_total} MB ({int(pct * 100)}%)",
        )
        if task.status == "paused":
            st.caption("Paused")
    elif task.status == "failed":
        st.error(task.error or "Download failed")
    elif task.status == "done":
        st.caption("Ready to use.")


def _download_button(key_suffix, task):
    if task.status == "pending":
        if st.button(
            "Download",
            key=f"dl_btn_{key_suffix}",
            use_container_width=True,
        ):
            task.start()
            st.rerun()
    elif task.status == "running":
        if st.button(
            "Pause",
            key=f"p_btn_{key_suffix}",
            use_container_width=True,
        ):
            task.pause()
            st.rerun()
    elif task.status == "paused":
        if st.button(
            "Resume",
            key=f"r_btn_{key_suffix}",
            use_container_width=True,
        ):
            task.start()
            st.rerun()
    elif task.status == "done":
        st.button(
            "Done",
            key=f"d_btn_{key_suffix}",
            disabled=True,
            use_container_width=True,
        )
    elif task.status == "failed":
        if st.button(
            "Retry",
            key=f"rt_btn_{key_suffix}",
            use_container_width=True,
        ):
            task.status = "pending"
            task.error = None
            task.start()
            st.rerun()


def _render_download_row(preset, task):
    col1, col2 = st.columns([5, 1], gap="medium")

    with col1:
        st.markdown(
            f'<div class="card-title">'
            f'{preset["name"]}'
            f'<span class="badge badge-gray">{preset["size"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-sub">{preset["description"]}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        _download_button(preset["repo"], task)

    _render_progress(task)
    st.markdown("<hr style='opacity: 0.2; margin: 1.5rem 0;'>", unsafe_allow_html=True)


def _render_status_pill(status: str):
    colors = {
        "Ready": "#a1a1aa",
        "Listening": "#ef4444",
        "Transcribing": "#f59e0b",
        "Thinking": "#3b82f6",
        "Speaking": "#10b981",
    }
    color = colors.get(status, "#a1a1aa")
    st.markdown(
        f"""
        <div class="status-pill" style="
            border: 1px solid {color}40;
            background: {color}10;
        ">
            <div class="status-dot" style="
                background: {color};
                box-shadow: 0 0 12px {color};
            "></div>
            <div style="color: {color}; line-height: 1;">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _group_sessions_by_date(sessions: dict) -> dict:
    now = datetime.now().date()
    yesterday = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    groups = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Previous 30 Days": [],
        "Older": [],
    }

    ordered = sorted(
        sessions.values(),
        key=lambda s: s.get("updated_at", 0),
        reverse=True,
    )

    for s in ordered:
        ts = s.get("updated_at", 0)
        d = datetime.fromtimestamp(ts).date()
        if d == now:
            groups["Today"].append(s)
        elif d == yesterday:
            groups["Yesterday"].append(s)
        elif d >= week_ago:
            groups["Previous 7 Days"].append(s)
        elif d >= month_ago:
            groups["Previous 30 Days"].append(s)
        else:
            groups["Older"].append(s)

    return {k: v for k, v in groups.items() if v}


# ==================================================================
# BOOT
# ==================================================================

inject_css()


if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()

if "current_sid" not in st.session_state:
    if st.session_state.sessions:
        ordered = sorted(
            st.session_state.sessions.values(),
            key=lambda s: s.get("updated_at", 0),
            reverse=True,
        )
        st.session_state.current_sid = ordered[0]["id"]
    else:
        sid, sess = new_session()
        st.session_state.sessions[sid] = sess
        st.session_state.current_sid = sid
        save_sessions(st.session_state.sessions)


def current_session():
    return st.session_state.sessions.get(st.session_state.current_sid)


def persist():
    save_sessions(st.session_state.sessions)


# ==================================================================
# SIDEBAR
# ==================================================================

with st.sidebar:
    st.markdown(
        '<div class="side-title">Dojo-Local-Model-Studio</div>'
        '<div class="side-sub">Local model studio</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "",
        ["Chat", "Voice", "Models"],
        index=0,
        label_visibility="collapsed",
        key="mode_radio",
    )

    # ==============================================================
    # CHAT SIDEBAR
    # ==============================================================
    if mode == "Chat":
        if st.button(
            "＋ New chat",
            type="primary",
            key="new_chat",
            use_container_width=True,
        ):
            sid, sess = new_session()
            st.session_state.sessions[sid] = sess
            st.session_state.current_sid = sid
            persist()
            st.rerun()

        groups = _group_sessions_by_date(st.session_state.sessions)

        for group_name, sessions in groups.items():
            st.markdown(
                f'<div class="side-label">{group_name}</div>',
                unsafe_allow_html=True,
            )

            for sess in sessions:
                sid = sess["id"]
                active = sid == st.session_state.current_sid

                row = st.columns([8, 1], gap="small")

                with row[0]:
                    label = sess["title"]
                    if active:
                        label = "▸ " + label
                    if st.button(
                        label,
                        key=f"open_{sid}",
                        use_container_width=True,
                    ):
                        st.session_state.current_sid = sid
                        st.rerun()

                with row[1]:
                    with st.popover("⋯", use_container_width=False):
                        if st.button("Rename", key=f"ren_{sid}"):
                            st.session_state["renaming"] = sid
                            st.rerun()
                        if st.button(
                            "Export .md", key=f"exp_{sid}"
                        ):
                            st.session_state["exporting"] = sid
                            st.rerun()
                        if st.button("Delete", key=f"del_{sid}"):
                            st.session_state.sessions.pop(sid, None)
                            if st.session_state.current_sid == sid:
                                remaining = list(
                                    st.session_state.sessions.keys()
                                )
                                if remaining:
                                    st.session_state.current_sid = (
                                        remaining[0]
                                    )
                                else:
                                    nsid, nsess = new_session()
                                    st.session_state.sessions[nsid] = nsess
                                    st.session_state.current_sid = nsid
                            persist()
                            st.rerun()

        if st.session_state.get("renaming"):
            sid = st.session_state["renaming"]
            if sid in st.session_state.sessions:
                st.markdown("---")
                new_title = st.text_input(
                    "Rename chat",
                    value=st.session_state.sessions[sid]["title"],
                    key="rename_input",
                    label_visibility="collapsed",
                )
                c1, c2 = st.columns(2, gap="small")
                if c1.button("Save", key="save_rename"):
                    st.session_state.sessions[sid]["title"] = (
                        new_title.strip() or "Untitled"
                    )
                    persist()
                    st.session_state.pop("renaming", None)
                    st.rerun()
                if c2.button("Cancel", key="cancel_rename"):
                    st.session_state.pop("renaming", None)
                    st.rerun()

        st.markdown("---")

        st.markdown(
            '<div class="side-label" style="margin-left: 0;">Model Settings</div>',
            unsafe_allow_html=True,
        )

        models = list_local_models()

        if not models:
            selected = None
            st.caption("No models. Go to Models tab.")
        else:
            names = [m["name"] for m in models]
            idx = st.session_state.get("chat_model_idx", 0)
            if idx >= len(names):
                idx = 0

            choice = st.selectbox(
                "model_sel",
                names,
                index=idx,
                label_visibility="collapsed",
            )
            st.session_state["chat_model_idx"] = names.index(choice)
            selected = next(m for m in models if m["name"] == choice)

        sess = current_session()

        with st.expander("Advanced Generation", expanded=False):
            sess["temperature"] = st.slider(
                "Temperature",
                0.0, 2.0,
                float(sess.get("temperature", 0.7)), 0.05,
            )
            sess["top_p"] = st.slider(
                "Top-p",
                0.0, 1.0,
                float(sess.get("top_p", 0.9)), 0.05,
            )
            sess["top_k"] = st.slider(
                "Top-k",
                0, 100,
                int(sess.get("top_k", 40)), 1,
            )
            sess["repeat_penalty"] = st.slider(
                "Repeat penalty",
                0.8, 1.5,
                float(sess.get("repeat_penalty", 1.1)), 0.05,
            )
            sess["max_tokens"] = st.slider(
                "Max tokens",
                64, 4096,
                int(sess.get("max_tokens", 512)), 64,
            )
            sess["seed"] = st.number_input(
                "Seed (-1 = random)",
                value=int(sess.get("seed", -1)),
                step=1,
            )
        persist()

    # ==============================================================
    # VOICE SIDEBAR
    # ==============================================================
    elif mode == "Voice":
        st.markdown(
            '<div class="side-label">Model</div>',
            unsafe_allow_html=True,
        )

        models = list_local_models()

        if not models:
            voice_selected = None
            st.caption("No models. Go to Models tab.")
        else:
            names = [m["name"] for m in models]
            idx = st.session_state.get("voice_model_idx", 0)
            if idx >= len(names):
                idx = 0

            choice = st.selectbox(
                "voice_model_sel",
                names,
                index=idx,
                label_visibility="collapsed",
                key="voice_model_selectbox",
            )
            st.session_state["voice_model_idx"] = names.index(choice)
            voice_selected = next(
                m for m in models if m["name"] == choice
            )

        st.markdown(
            '<div class="side-label">Voice</div>',
            unsafe_allow_html=True,
        )

        downloaded_voices = list_downloaded_voices()

        if not downloaded_voices:
            voice_choice_id = None
            st.caption("No voices. Download one in Voice tab.")
        else:
            voice_labels = [v["id"] for v in downloaded_voices]
            v_idx = st.session_state.get("voice_idx", 0)
            if v_idx >= len(voice_labels):
                v_idx = 0

            v_choice = st.selectbox(
                "voice_sel",
                voice_labels,
                index=v_idx,
                label_visibility="collapsed",
                key="voice_voice_selectbox",
            )
            st.session_state["voice_idx"] = voice_labels.index(v_choice)
            voice_choice_id = v_choice

        st.markdown("---")

        if st.button(
            "New voice session",
            key="new_voice",
            use_container_width=True,
        ):
            st.session_state["voice_messages"] = []
            st.session_state["voice_last_audio"] = None
            st.session_state["voice_last_hash"] = None
            st.session_state["voice_status"] = "Ready"
            st.rerun()

    # ==============================================================
    # MODELS SIDEBAR
    # ==============================================================
    elif mode == "Models":
        usage = disk_usage()

        st.markdown(
            '<div class="side-label" style="margin-left: 0;">Storage</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{usage['count']} models · {usage['total_gb']} GB"
        )

        voices = list_downloaded_voices()

        st.markdown(
            '<div class="side-label" style="margin-left: 0;">Voices</div>',
            unsafe_allow_html=True,
        )

        if voices:
            total_mb = sum(v["size_mb"] for v in voices)
            st.caption(
                f"{len(voices)} voices · {round(total_mb, 1)} MB"
            )
        else:
            st.caption("No voices yet.")


# ==================================================================
# EXPORT MODAL
# ==================================================================

if st.session_state.get("exporting"):
    sid = st.session_state["exporting"]
    sess = st.session_state.sessions.get(sid)

    if sess:
        md = export_session_markdown(sess)
        st.download_button(
            "Download chat",
            data=md.encode("utf-8"),
            file_name=f"{sess['title'][:40]}.md",
            mime="text/markdown",
            key="dl_export",
        )
        if st.button("Close"):
            st.session_state.pop("exporting", None)
            st.rerun()


# ==================================================================
# CHAT
# ==================================================================

if mode == "Chat":
    sess = current_session()

    top = st.columns([5, 1], gap="medium")

    with top[0]:
        st.markdown(
            f'<div class="page-title">{sess["title"]}</div>',
            unsafe_allow_html=True,
        )
        sub = "Streaming chat with your local model"
        if selected:
            sub = f"{selected['name']}  ·  {sub}"
        st.markdown(
            f'<div class="page-sub">{sub}</div>',
            unsafe_allow_html=True,
        )

    with top[1]:
        with st.popover("Context", use_container_width=True):
            sess["system_prompt"] = st.text_area(
                "System prompt",
                value=sess.get(
                    "system_prompt", DEFAULT_SETTINGS["system_prompt"]
                ),
                height=150,
                key=f"sys_{sess['id']}",
            )
            if st.button("Save system prompt", key="save_sys"):
                persist()
                st.success("Saved")
            if st.button("Clear messages", key="clear_msgs"):
                sess["messages"] = []
                persist()
                st.rerun()

    if selected is None:
        st.markdown(
            '<div class="empty-state">'
            '<h2>No models available</h2>'
            '<p>Go to the Models tab and download one to begin.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    if sess.get("model_path") != selected["path"]:
        sess["model_path"] = selected["path"]
        sess["model_name"] = selected["name"]
        sess["messages"] = []
        persist()

    try:
        model = load_llama(
            selected["path"],
            n_ctx=4096,
            n_gpu_layers=-1,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    # ---- Empty state ----
    if not sess["messages"]:
        st.markdown(
            """
            <div class="empty-state">
                <h2>What can I help with?</h2>
                <p>Ask anything, or pick a suggestion below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        suggestions = [
            "Explain closures in Python with a simple example",
            "Write a haiku about rain",
            "Give me 3 ideas for a weekend project",
            "Summarize the water cycle in 3 sentences",
        ]

        for row_start in (0, 2):
            cols = st.columns(2, gap="small")
            for col_idx, text in enumerate(
                suggestions[row_start:row_start + 2]
            ):
                with cols[col_idx]:
                    st.markdown(
                        '<div class="sugg-card">',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        text,
                        key=f"sugg_{sess['id']}_{row_start + col_idx}",
                        use_container_width=True,
                    ):
                        st.session_state["pending_prompt"] = text
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Message history ----
    else:
        last_assistant_idx = None
        for i, m in enumerate(sess["messages"]):
            if m["role"] == "assistant":
                last_assistant_idx = i

        for i, msg in enumerate(sess["messages"]):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    is_last = i == last_assistant_idx
                    ac = st.columns([1, 1, 1, 12], gap="small")

                    with ac[0]:
                        if st.button(
                            "⎘",
                            key=f"copy_{sess['id']}_{i}",
                            help="Copy",
                        ):
                            st.session_state["clipboard"] = msg["content"]
                            st.toast("Copied to clipboard")

                    with ac[1]:
                        if is_last and st.button(
                            "↻",
                            key=f"regen_{sess['id']}_{i}",
                            help="Regenerate",
                        ):
                            sess["messages"].pop()
                            persist()
                            st.rerun()

                    with ac[2]:
                        if st.button(
                            "✕",
                            key=f"delmsg_{sess['id']}_{i}",
                            help="Delete",
                        ):
                            sess["messages"].pop(i)
                            persist()
                            st.rerun()

    # ---- Clipboard panel ----
    if st.session_state.get("clipboard"):
        with st.expander("Copied content", expanded=False):
            st.code(st.session_state["clipboard"], language=None)
            if st.button("Dismiss", key="dismiss_clip"):
                st.session_state.pop("clipboard", None)
                st.rerun()

    # ---- Input ----
    pending = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("Message Dojo-Local-Model-Studio")
    prompt = pending or typed

    if prompt:
        sess["messages"].append({"role": "user", "content": prompt})
        if len(sess["messages"]) == 1:
            sess["title"] = title_from_message(prompt)
        sess["updated_at"] = time.time()
        persist()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            msgs = []
            if sess.get("system_prompt"):
                msgs.append(
                    {"role": "system", "content": sess["system_prompt"]}
                )
            msgs.extend(sess["messages"])

            stream = stream_chat(
                model,
                msgs,
                temperature=float(sess.get("temperature", 0.7)),
                top_p=float(sess.get("top_p", 0.9)),
                top_k=int(sess.get("top_k", 40)),
                repeat_penalty=float(sess.get("repeat_penalty", 1.1)),
                max_tokens=int(sess.get("max_tokens", 512)),
                seed=int(sess.get("seed", -1)),
            )
            response = st.write_stream(stream)

        sess["messages"].append(
            {"role": "assistant", "content": response}
        )
        sess["updated_at"] = time.time()
        persist()
        st.rerun()


# ==================================================================
# VOICE
# ==================================================================

elif mode == "Voice":
    models = list_local_models()

    if not models:
        st.markdown(
            '<div class="empty-state">'
            '<h2>No models available</h2>'
            '<p>Go to the Models tab and download one.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    if voice_selected is None:
        st.info("Pick a model in the sidebar.")
        st.stop()

    downloaded_voices = list_downloaded_voices()

    if not downloaded_voices:
        st.markdown(
            '<div class="page-title">Set up a voice</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="page-sub">'
            "Download a voice pack to enable speech. "
            "Each is about 63 MB and stored on your disk."
            "</div>",
            unsafe_allow_html=True,
        )

        for v in AVAILABLE_VOICES:
            col1, col2 = st.columns([5, 1], gap="medium")

            with col1:
                st.markdown(f"**{v['name']}**")
                st.caption(f"ID: {v['id']}")

            with col2:
                if st.button(
                    "Download",
                    key=f"dl_voice_{v['id']}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Downloading {v['name']}..."):
                        try:
                            download_voice(v)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            st.markdown("<hr style='opacity: 0.2; margin: 1.5rem 0;'>", unsafe_allow_html=True)

        st.stop()

    if voice_choice_id is None:
        st.info("Pick a voice in the sidebar.")
        st.stop()

    if "voice_messages" not in st.session_state:
        st.session_state["voice_messages"] = []
    if "voice_status" not in st.session_state:
        st.session_state["voice_status"] = "Ready"
    if "voice_last_audio" not in st.session_state:
        st.session_state["voice_last_audio"] = None
    if "voice_last_hash" not in st.session_state:
        st.session_state["voice_last_hash"] = None

    st.markdown(
        '<div class="page-title">Voice</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-sub">'
        "Speak. The model listens and replies. Fully offline."
        "</div>",
        unsafe_allow_html=True,
    )

    _render_status_pill(st.session_state["voice_status"])

    for msg in st.session_state["voice_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    last_audio = st.session_state.get("voice_last_audio")

    if last_audio:
        st.markdown(
            '<div class="side-label" style="margin-left:0;">Last reply</div>',
            unsafe_allow_html=True,
        )
        st.components.v1.html(
            audio_autoplay_html(last_audio),
            height=80,
        )

    st.markdown("<hr style='opacity: 0.2; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    audio_value = st.audio_input(
        "Speak your message",
        key=f"voice_audio_{len(st.session_state['voice_messages'])}",
    )

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        current_hash = hash(audio_bytes)

        if current_hash != st.session_state["voice_last_hash"]:
            st.session_state["voice_last_hash"] = current_hash
            engine = get_engine()

            st.session_state["voice_status"] = "Transcribing"

            with st.spinner("Transcribing..."):
                try:
                    user_text = engine.transcribe_bytes(audio_bytes)
                except Exception as e:
                    st.session_state["voice_status"] = "Ready"
                    st.error(f"Transcription failed: {e}")
                    st.stop()

            if not user_text:
                st.session_state["voice_status"] = "Ready"
                st.warning("No speech detected. Try again.")
                st.stop()

            st.session_state["voice_messages"].append(
                {"role": "user", "content": user_text}
            )

            st.session_state["voice_status"] = "Thinking"

            try:
                model = load_llama(
                    voice_selected["path"],
                    n_ctx=2048,
                    n_gpu_layers=-1,
                )
            except Exception as e:
                st.session_state["voice_status"] = "Ready"
                st.error(str(e))
                st.stop()

            msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful voice assistant. "
                        "Answer in 1 to 3 short sentences. "
                        "Be direct. No bullet points. No markdown."
                    ),
                }
            ]
            msgs.extend(st.session_state["voice_messages"])

            with st.spinner("Thinking..."):
                try:
                    response_text = complete_once(
                        model,
                        msgs,
                        temperature=0.7,
                        max_tokens=180,
                    )
                except Exception as e:
                    st.session_state["voice_status"] = "Ready"
                    st.error(f"Generation failed: {e}")
                    st.stop()

            st.session_state["voice_messages"].append(
                {"role": "assistant", "content": response_text}
            )

            st.session_state["voice_status"] = "Speaking"

            with st.spinner("Generating speech..."):
                try:
                    audio_out = engine.synthesize_to_bytes(
                        response_text, voice_choice_id
                    )
                    st.session_state["voice_last_audio"] = audio_out
                except Exception as e:
                    st.session_state["voice_status"] = "Ready"
                    st.error(f"Speech synthesis failed: {e}")
                    st.stop()

            st.session_state["voice_status"] = "Ready"
            st.rerun()

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        if st.button(
            "New voice session",
            use_container_width=True,
            key="reset_voice",
        ):
            st.session_state["voice_messages"] = []
            st.session_state["voice_last_audio"] = None
            st.session_state["voice_last_hash"] = None
            st.session_state["voice_status"] = "Ready"
            st.rerun()

    with c2:
        if st.session_state.get("voice_last_audio"):
            if st.button(
                "Replay last reply",
                use_container_width=True,
                key="replay_voice",
            ):
                st.components.v1.html(
                    audio_autoplay_html(
                        st.session_state["voice_last_audio"]
                    ),
                    height=80,
                )


# ==================================================================
# MODELS
# ==================================================================

elif mode == "Models":
    st.markdown(
        '<div class="page-title">Model Management</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-sub">'
        "Download and manage localized GGUF inference models."
        "</div>",
        unsafe_allow_html=True,
    )

    tab_browse, tab_search, tab_installed, tab_voices = st.tabs(
        ["Recommended", "Search Hugging Face", "Installed", "Voices"]
    )

    with tab_browse:
        for preset in PRESET_MODELS:
            url = build_download_url(preset["repo"], preset["file"])
            task = DOWNLOAD_MANAGER.get_or_create(
                preset["repo"],
                preset["file"],
                url,
                name=preset["name"],
            )
            _render_download_row(preset, task)

    with tab_search:
        c1, c2 = st.columns([4, 1], gap="small")

        with c1:
            query = st.text_input(
                "Search",
                placeholder="Search models (e.g. qwen, llama, mistral)",
                label_visibility="collapsed",
            )

        with c2:
            hf_token = st.text_input(
                "HF token (optional)",
                type="password",
                label_visibility="collapsed",
                placeholder="Token...",
            )

        if query and st.button("Search Registry", type="primary"):
            with st.spinner("Searching..."):
                try:
                    results = search_gguf_models(query, limit=20)
                    st.session_state["search_results"] = results
                except Exception as e:
                    st.error(str(e))

        results = st.session_state.get("search_results", [])

        if results:
            for r in results:
                with st.expander(
                    f"{r['repo_id']}  ·  {r['downloads']} downloads"
                ):
                    if st.button(
                        "List files",
                        key=f"ls_{r['repo_id']}",
                    ):
                        try:
                            files = list_repo_ggufs(
                                r["repo_id"], hf_token or None
                            )
                            st.session_state[
                                f"files_{r['repo_id']}"
                            ] = files
                        except Exception as e:
                            st.error(str(e))

                    files = st.session_state.get(
                        f"files_{r['repo_id']}", []
                    )

                    if files:
                        options = [
                            f"{f['filename']}  ({f['size_gb']} GB)"
                            for f in files
                        ]
                        sel = st.selectbox(
                            "Choose a file",
                            options,
                            key=f"sel_{r['repo_id']}",
                        )
                        idx = options.index(sel)
                        fname = files[idx]["filename"]

                        if st.button(
                            "Download Model",
                            key=f"dl_{r['repo_id']}",
                            type="primary",
                        ):
                            url = build_download_url(
                                r["repo_id"], fname
                            )
                            task = DOWNLOAD_MANAGER.get_or_create(
                                r["repo_id"],
                                fname,
                                url,
                                hf_token=hf_token or None,
                                name=fname.replace(".gguf", ""),
                            )
                            task.start()
                            st.rerun()

            for key, task in DOWNLOAD_MANAGER.tasks.items():
                if task.status in ("running", "paused"):
                    _render_progress(task)

    with tab_installed:
        installed = list_local_models()

        if not installed:
            st.markdown(
                '<div class="empty-state">'
                '<h2>No models yet</h2>'
                '<p>Download one from the Recommended tab to begin inference.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for m in installed:
                col1, col2 = st.columns([5, 1], gap="medium")

                with col1:
                    st.markdown(
                        f'<div class="card-title">'
                        f'{m["name"]}'
                        f'<span class="badge badge-gray">'
                        f'{m["size_gb"]} GB'
                        f'</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="card-sub">{m["path"]}</div>',
                        unsafe_allow_html=True,
                    )

                with col2:
                    with st.popover(
                        "Remove", use_container_width=True
                    ):
                        st.write(f"Delete **{m['name']}**?")
                        if st.button(
                            "Confirm",
                            key=f"cdel_{m['path']}",
                        ):
                            unload_model(m["path"])
                            delete_model(m["path"])
                            st.rerun()

                st.markdown("<hr style='opacity: 0.2; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    with tab_voices:
        st.markdown(
            '<div style="font-size: 18px; font-weight:600; color:var(--text-primary); margin-top: 1rem;">'
            "Voice packs"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="page-sub">'
            "Small speech models (60-80 MB) for text-to-speech. "
            "Everything runs offline."
            "</div>",
            unsafe_allow_html=True,
        )

        for v in AVAILABLE_VOICES:
            exists = is_voice_downloaded(v["id"])

            col1, col2 = st.columns([5, 1], gap="medium")

            with col1:
                badge = (
                    '<span class="badge">Installed</span>'
                    if exists
                    else ""
                )
                st.markdown(
                    f'<div class="card-title">'
                    f'{v["name"]}{badge}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="card-sub">{v["id"]}</div>',
                    unsafe_allow_html=True,
                )

            with col2:
                if not exists:
                    if st.button(
                        "Download",
                        key=f"vd_{v['id']}",
                        use_container_width=True,
                    ):
                        with st.spinner("Downloading..."):
                            try:
                                download_voice(v)
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                else:
                    st.button(
                        "Installed",
                        key=f"vi_{v['id']}",
                        disabled=True,
                        use_container_width=True,
                    )

            st.markdown("<hr style='opacity: 0.2; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    if DOWNLOAD_MANAGER.has_active():
        time.sleep(0.5)
        st.rerun()
