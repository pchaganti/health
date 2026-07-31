"""Route chat input to the configured AI model via litellm."""
from __future__ import annotations
from datetime import timedelta
import os
import re
import sys
import time
import threading

import pandas as pd

from healthai.models import MODELS
from healthai.preferences import load_preferences

_MODEL_LABELS: dict[str, str] = {m[1]: m[2] for m in MODELS if m[1] != "__custom__"}

ALL_DATA_FILES = [
    ("steps_data.csv",      "Steps"),
    ("distance_data.csv",   "Distance"),
    ("heart_rate_data.csv", "Heart Rate"),
    ("weight_data.csv",     "Weight"),
    ("sleep_daily.csv",     "Sleep"),
    ("workout_data.csv",    "Workout"),
]

_LEGACY_PROVIDER_MAP = {
    "openai":     "gpt-4o",
    "claude":     "claude-sonnet-4-20250514",
    "gemini":     "gemini/gemini-2.5-flash",
    "grok":       "xai/grok-3",
    "openrouter": "openrouter/openai/gpt-4o",
    "ollama":     "ollama_chat/llama3",
    "lmstudio":   "lmstudio/local",
    "litellm":    "gpt-4o",
}


def _load_prefs() -> dict:
    return load_preferences()


def get_configured_model() -> str:
    """Return the full litellm model string from prefs, defaulting to gpt-4o."""
    prefs = _load_prefs()
    model = prefs.get("default_model") or prefs.get("default_provider", "gpt-4o")
    return _LEGACY_PROVIDER_MAP.get(model, model)


# Backwards-compat alias — cli.py calls this name
get_configured_provider = get_configured_model


def get_model_label(model_str: str) -> str:
    return _MODEL_LABELS.get(model_str, model_str)


# Backwards-compat alias
get_provider_label = get_model_label


def _read_health_frames(output_dir: str) -> list[tuple[str, str, pd.DataFrame]]:
    frames = []
    for filename, label in ALL_DATA_FILES:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            try:
                frame = pd.read_csv(path)
                if "date" in frame.columns and not frame.empty:
                    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                    frame = frame.dropna(subset=["date"]).sort_values("date")
                if not frame.empty:
                    frames.append((filename, label, frame))
            except Exception:
                pass
    return frames


def _date_window(
    question: str,
    anchor: pd.Timestamp,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    lowered = question.casefold()
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", lowered)]
    if years:
        year = years[-1]
        return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(
            year=year,
            month=12,
            day=31,
        )
    if "last year" in lowered:
        year = anchor.year - 1
        return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(
            year=year,
            month=12,
            day=31,
        )
    if "this year" in lowered:
        return pd.Timestamp(year=anchor.year, month=1, day=1), anchor
    if "this month" in lowered:
        return anchor.replace(day=1), anchor
    if "this week" in lowered:
        return anchor - timedelta(days=anchor.weekday()), anchor
    if "today" in lowered:
        return anchor, anchor
    if "last week" in lowered:
        end = anchor - timedelta(days=anchor.weekday() + 1)
        return end - timedelta(days=6), end
    if "last month" in lowered:
        end = anchor.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end

    match = re.search(
        r"\b(?:past|last)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b",
        lowered,
    )
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("day"):
            start = anchor - timedelta(days=max(0, amount - 1))
        elif unit.startswith("week"):
            start = anchor - timedelta(days=max(0, amount * 7 - 1))
        elif unit.startswith("month"):
            start = anchor - pd.DateOffset(months=amount) + timedelta(days=1)
        else:
            start = anchor - pd.DateOffset(years=amount) + timedelta(days=1)
        return pd.Timestamp(start), anchor

    if any(term in lowered for term in ("recent", "latest")):
        return anchor - timedelta(days=89), anchor
    return None, None


def _complete_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text.rstrip() + "\n"
    lines = text.splitlines()
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        line_size = len(line) + 1
        if kept and size + line_size > limit:
            break
        kept.append(line)
        size += line_size
    return "[Earlier rows omitted; newest complete rows retained]\n" + "\n".join(
        reversed(kept)
    ) + "\n"


def _daily_metric_context(
    label: str,
    frame: pd.DataFrame,
    question: str,
) -> str:
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    value_column = "value" if "value" in frame else (
        numeric_columns[0] if numeric_columns else None
    )
    date_min = frame["date"].min().date()
    date_max = frame["date"].max().date()
    lines = [
        f"=== {label} ===",
        f"Available date range: {date_min} to {date_max}",
        f"Rows in selected period: {len(frame)}",
    ]
    if value_column:
        values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
        if not values.empty:
            lines.append(
                f"{value_column}: mean={values.mean():.2f}, "
                f"median={values.median():.2f}, min={values.min():.2f}, "
                f"max={values.max():.2f}, total={values.sum():.2f}"
            )

    lowered = question.casefold()
    if label == "Steps" and value_column:
        weekly = (
            frame.set_index("date")[value_column]
            .resample("W-SUN")
            .sum()
            .dropna()
            .sort_index()
        )
        lines.append("\nWeekly step totals (top 20 in selected period):")
        top_weeks = weekly.sort_values(ascending=False).head(20).sort_index()
        lines.append(top_weeks.rename("steps").to_csv())
        if "week" not in lowered:
            lines.append("Most recent weekly step totals:")
            lines.append(weekly.tail(12).rename("steps").to_csv())
    if value_column and (len(frame) > 60 or "month" in lowered):
        monthly_values = frame.set_index("date")[value_column].resample("MS")
        if label in {"Steps", "Distance"}:
            monthly = monthly_values.sum().rename("total").to_frame()
        else:
            monthly = monthly_values.agg(["mean", "min", "max"]).dropna(how="all")
        lines.append("\nMonthly statistics:")
        lines.append(monthly.tail(60).to_csv())

    lines.append("\nComplete daily rows (newest 180 maximum):")
    lines.append(frame.tail(180).to_csv(index=False, date_format="%Y-%m-%d"))
    return _complete_text("\n".join(lines))


def _workout_context(frame: pd.DataFrame) -> str:
    lines = [
        "=== Workout ===",
        f"Available date range: {frame['date'].min().date()} to {frame['date'].max().date()}",
        f"Workouts in selected period: {len(frame)}",
    ]
    if "duration_minutes" in frame:
        lines.append(
            f"Total workout minutes: {pd.to_numeric(frame['duration_minutes'], errors='coerce').sum():.1f}"
        )
    if "activity_type" in frame and "duration_minutes" in frame:
        activity = (
            frame.groupby("activity_type")
            .agg(
                workouts=("activity_type", "size"),
                total_minutes=("duration_minutes", "sum"),
                average_minutes=("duration_minutes", "mean"),
            )
            .sort_values("total_minutes", ascending=False)
        )
        lines.append("\nWorkout types:")
        lines.append(activity.to_csv())
    lines.append("\nComplete workout rows (newest 180 maximum):")
    lines.append(frame.tail(180).to_csv(index=False, date_format="%Y-%m-%d"))
    return _complete_text("\n".join(lines))


def _build_context(output_dir: str, user_message: str = "") -> str:
    """Build complete-row, query-aware summaries from processed health data."""
    frames = _read_health_frames(output_dir)
    dated = [frame for _, _, frame in frames if "date" in frame and not frame.empty]
    if not dated:
        return ""
    anchor = max(frame["date"].max() for frame in dated)
    start, end = _date_window(user_message, anchor)

    parts = []
    for _, label, original in frames:
        frame = original
        if "date" in frame and start is not None:
            frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        if frame.empty:
            continue
        if label == "Workout":
            parts.append(_workout_context(frame))
        else:
            parts.append(_daily_metric_context(label, frame, user_message))
    return "\n".join(parts).rstrip() + "\n" if parts else ""


def _metric_exports_stale(output_dir: str, export_path: str) -> bool:
    try:
        export_mtime = os.path.getmtime(export_path)
    except OSError:
        return False
    for filename, _ in ALL_DATA_FILES:
        path = os.path.join(output_dir, filename)
        try:
            if os.path.getmtime(path) < export_mtime:
                return True
        except OSError:
            return True
    return False


_THINKING_PHRASES = [
    "Thinking…",
    "Analyzing your data…",
    "Crunching the numbers…",
    "Reading your health records…",
    "Almost there…",
    "Connecting the dots…",
]
_SPINNER_FRAMES = ["⠋", "⠙", "⠚", "⠞", "⠖", "⠦", "⠴", "⠲", "⠳", "⠓"]
_CONVERSATION_HISTORY: list[dict[str, str]] = []


def _thinking_spinner(stop_event: threading.Event) -> None:
    """Animated thinking indicator — runs in a background thread."""
    from healthai.ui import _D, _X
    start = time.time()
    i = 0
    phrase_idx = 0
    phrase_change = 3.0  # seconds between phrase rotations
    try:
        while not stop_event.is_set():
            elapsed = time.time() - start
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            phrase = _THINKING_PHRASES[phrase_idx % len(_THINKING_PHRASES)]
            line = f"\r  {_D}{frame}  {phrase}  {elapsed:.1f}s{_X}   "
            sys.stdout.write(line)
            sys.stdout.flush()
            if elapsed >= phrase_change * (phrase_idx + 1):
                phrase_idx += 1
            i += 1
            time.sleep(0.08)
    except Exception:
        pass
    # clear the line
    sys.stdout.write("\r" + " " * 72 + "\r")
    sys.stdout.flush()


def chat(user_message: str) -> dict:
    """Send user_message to the configured litellm model with health CSV context.

    Returns a stats dict: {"elapsed": float, "tokens_in": int, "tokens_out": int}.
    """
    try:
        import litellm
    except ImportError:
        print("  litellm not installed. Run: pip install litellm")
        return {"elapsed": 0.0, "tokens_in": 0, "tokens_out": 0}

    from healthai.ui import _C, _D, _G, _X, _Y

    model = get_configured_model()
    prefs = _load_prefs()
    output_dir = prefs.get("output_dir", os.path.join(os.path.expanduser("~"), "healthai_output"))
    export_path = prefs.get("export_xml") or prefs.get("export_xml_path")

    # Inject any saved API key for this model into the environment
    for _grp, _mstr, _disp, _needs_key, _key_env in MODELS:
        if _mstr == model and _key_env and not os.environ.get(_key_env):
            saved = prefs.get(_key_env, "")
            if saved:
                os.environ[_key_env] = saved
            break

    if export_path and _metric_exports_stale(output_dir, export_path):
        try:
            from healthai.cli import prepare_metric_exports

            prepare_metric_exports(
                export_path=export_path,
                output_dir=output_dir,
                quiet=True,
            )
        except Exception as error:
            print(f"  Could not refresh processed health data: {error}")

    context = _build_context(output_dir, user_message)
    system_prompt = (
        "You are a personal health analyst with access to the user's Apple Health data. "
        "Answer questions using only the data provided. Be specific and cite numbers and dates. "
        "The context contains locally calculated summaries and complete CSV rows. "
        "If data is insufficient, say so clearly.\n\n"
        + (
            f"Health data:\n{context}"
            if context
            else "No processed health data was found. Ask the user to verify the export path or run /csv."
        )
    )

    print()
    stop = threading.Event()
    spinner_thread = threading.Thread(target=_thinking_spinner, args=(stop,), daemon=True)
    spinner_thread.start()

    t0 = time.time()
    try:
        extra: dict = {}
        if model.startswith("ollama"):
            extra["api_base"] = prefs.get("ollama_host", "http://localhost:11434")
        if prefs.get("openai_compat_base") and model.startswith("openai/"):
            extra["api_base"] = prefs["openai_compat_base"]

        messages = [
            {"role": "system", "content": system_prompt},
            *_CONVERSATION_HISTORY[-8:],
            {"role": "user", "content": user_message},
        ]
        response = litellm.completion(
            model=model,
            messages=messages,
            **extra,
        )
        elapsed = time.time() - t0
        stop.set()
        spinner_thread.join()

        answer = response.choices[0].message.content or ""
        _CONVERSATION_HISTORY.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer},
            ]
        )
        usage = getattr(response, "usage", None)
        tokens_in  = getattr(usage, "prompt_tokens",     0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

        print(f"  {_G}●{_X} {answer.strip()}\n")
        return {"elapsed": elapsed, "tokens_in": tokens_in, "tokens_out": tokens_out}

    except Exception as e:
        elapsed = time.time() - t0
        stop.set()
        spinner_thread.join()
        print(f"  {_Y}Error:{_X} {e}\n  Run /settings to check your model and API key.\n")
        return {"elapsed": elapsed, "tokens_in": 0, "tokens_out": 0}
