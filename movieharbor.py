import os
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from config import get_settings
from prompts import build_recommendation_prompt

try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError:  # Optional when running local-only.
    genai = None
    types = None


def demo_log(level: str, message: str) -> None:
    """Render live agent logs when demo logging is enabled."""
    if not st.session_state.get("show_logs", False):
        return

    if level == "info":
        st.info(message)
    elif level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.caption(message)


def apply_modern_styles() -> None:
    """Apply a lightweight modern 2026-style UI theme."""
    css_path = Path(__file__).with_name("styles.css")
    if not css_path.exists():
        st.warning("styles.css не знайдено. Використовується стандартний стиль Streamlit.")
        return

    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def normalize_genres(value: Any) -> list[str]:
    """Normalize a genre field into a clean list of strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        cleaned = value.strip().strip("[]")
        if not cleaned:
            return []
        parts = [part.strip().strip("'\"") for part in cleaned.split(",")]
        return [part for part in parts if part]

    return []


def get_google_api_key() -> str:
    """Return the first configured API key from supported env names."""
    key_names = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "AISTUDIO_API")
    for key_name in key_names:
        value = os.getenv(key_name)
        if value and value.strip():
            return value.strip().strip('"').strip("'")
    return ""


def trigger_cinematic_video(movie_title: str, visual_justification: str) -> str:
    """Fetch a trailer when visual evidence is needed for recommendation confidence."""
    demo_log("info", f"🎬 Виклик інструмента: `trigger_cinematic_video('{movie_title}')`")
    demo_log("caption", f"Причина: {visual_justification}")
    st.session_state["video_justification"] = visual_justification
    tmdb_api_key = os.getenv("TMDB_API_KEY", "").strip().strip('"').strip("'")
    if not tmdb_api_key:
        message = "Відсутній TMDB_API_KEY."
        demo_log("error", f"❌ {message}")
        return message

    try:
        demo_log("caption", f"🔍 Пошук у TMDB: `{movie_title}`")
        search_res = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": tmdb_api_key, "query": movie_title},
            timeout=15,
        )
        search_res.raise_for_status()
        search_json = search_res.json()
        if not search_json.get("results"):
            message = f"Фільм '{movie_title}' не знайдено."
            demo_log("warning", f"⚠️ {message}")
            return message

        movie_id = search_json["results"][0]["id"]
        demo_log("caption", f"✅ Знайдено TMDB ID `{movie_id}`. Отримую посилання на трейлер.")
        video_res = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
            params={"api_key": tmdb_api_key},
            timeout=15,
        )
        video_res.raise_for_status()
        videos = video_res.json().get("results", [])

        for video in videos:
            is_youtube = video.get("site") == "YouTube"
            is_trailer = video.get("type") == "Trailer"
            if is_youtube and is_trailer and video.get("key"):
                url = f"https://www.youtube.com/watch?v={video['key']}"
                demo_log("success", f"🎯 Знайдено URL трейлера: {url}")
                st.session_state["fetched_trailer"] = url
                return url

        message = f"Офіційний трейлер YouTube для '{movie_title}' не знайдено."
        demo_log("warning", f"⚠️ {message}")
        return message
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        message = f"Помилка запиту до TMDB: {exc}"
        demo_log("error", f"❌ {message}")
        return message


@st.cache_data(show_spinner=False)
def load_movies(path: str) -> pd.DataFrame:
    """Load and normalize the movie dataset."""
    movies = pd.read_pickle(path).copy()
    if "genres" not in movies.columns:
        movies["genres"] = [[] for _ in range(len(movies))]
    movies["genres"] = movies["genres"].map(normalize_genres)
    return movies


def genre_options(movies: pd.DataFrame) -> list[str]:
    """Extract sorted unique genres."""
    return sorted({genre for genres in movies["genres"] for genre in genres})


def filter_by_genre(movies: pd.DataFrame, genre: str) -> pd.DataFrame:
    """Return movies that include the selected genre."""
    genre_mask = movies["genres"].explode().eq(genre).groupby(level=0).any()
    return movies[genre_mask.reindex(movies.index, fill_value=False)].copy()


def _embedding_array(value: Any) -> np.ndarray | None:
    """Convert a stored embedding vector to a numeric numpy array."""
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        return None
    return array


def semantic_search(client: Any, movies: pd.DataFrame, query: str, top_k: int) -> pd.DataFrame:
    """Run semantic ranking with cosine similarity inside a filtered movie pool."""
    settings = get_settings()
    google_api_key = get_google_api_key()
    can_embed = (
        settings.enable_semantic_search
        and settings.llm_provider == "google"
        and client is not None
        and google_api_key
    )
    if not can_embed or movies.empty or "vector" not in movies.columns:
        return movies.head(top_k).copy()

    query_response = client.models.embed_content(model=settings.google_embedding_model, contents=query)
    query_vector = np.asarray(query_response.embeddings[0].values, dtype=float)
    if query_vector.ndim != 1 or query_vector.size == 0:
        return movies.head(top_k).copy()

    vectors = movies["vector"].map(_embedding_array)
    valid_mask = vectors.map(lambda vector: vector is not None and vector.shape == query_vector.shape)
    if not valid_mask.any():
        st.info("No valid vectors matched the embedding shape. Falling back to top candidates.")
        return movies.head(top_k).copy()

    ranked = movies[valid_mask].copy()
    matrix = np.vstack(vectors[valid_mask].to_list())
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    denominator = np.where(denominator == 0.0, 1e-12, denominator)
    ranked["similarity"] = np.dot(matrix, query_vector) / denominator
    return ranked.sort_values(by="similarity", ascending=False).head(top_k)


def build_context(movies: pd.DataFrame) -> str:
    """Create grounded context for Gemini from top candidate movies."""
    rows: list[str] = []
    for row in movies.itertuples(index=False):
        title = getattr(row, "title", "Unknown title")
        overview = getattr(row, "overview", "No overview available.")
        genres = ", ".join(getattr(row, "genres", []))
        rows.append(f"- {title} | Genres: {genres}\n  Overview: {overview}")
    return "\n\n".join(rows)


def create_local_client(base_url: str, api_key: str) -> Any:
    """Create an OpenAI-compatible client for LM Studio."""
    try:
        import importlib

        openai_module = importlib.import_module("openai")
    except ModuleNotFoundError:
        st.error("Відсутня залежність openai. Встановіть `python -m pip install openai`.")
        st.stop()
    return openai_module.OpenAI(base_url=base_url, api_key=api_key)


def generate_with_fallback(client: Any, prompt: str, preferred_model: str) -> str:
    """Generate content with trailer tool support and model fallback."""
    settings = get_settings()
    generation_config = types.GenerateContentConfig(
        temperature=0.7,
        tools=[trigger_cinematic_video],
    )
    attempted: list[str] = []
    for model_name in (preferred_model, *settings.google_model_fallbacks):
        if model_name in attempted:
            continue
        attempted.append(model_name)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )
            text = response.text if response and getattr(response, "text", None) else ""
            if text:
                if model_name != preferred_model:
                    st.info(f"Using fallback model: {model_name}")
                return text
        except Exception as exc:
            # Retry with fallback only for model-availability issues.
            error_text = str(exc).upper()
            if "NOT_FOUND" not in error_text and "404" not in error_text:
                raise
    raise RuntimeError(f"No available generation model from: {attempted}")


def generate_local_response(client: Any, prompt: str, model_name: str) -> str:
    """Generate recommendations from LM Studio with function-calling support."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "trigger_cinematic_video",
                "description": (
                    "Fetch a video ONLY when text is insufficient to convey a movie's specific "
                    "visual aesthetic, directorial style, pacing, or comedic tone."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_title": {
                            "type": "string",
                            "description": "The exact title of the movie to fetch a trailer for.",
                        },
                        "visual_justification": {
                            "type": "string",
                            "description": (
                                "A short sentence explaining exactly what visual or auditory "
                                "element the video demonstrates."
                            ),
                        },
                    },
                    "required": ["movie_title", "visual_justification"],
                },
            },
        }
    ]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "Ти — корисний AI-помічник з рекомендації фільмів. Відповідай виключно українською мовою та уважно дотримуйся інструкцій щодо використання інструментів.",
        },
        {"role": "user", "content": prompt},
    ]

    first = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        tools=tools,
        tool_choice="auto",
    )
    if not first.choices:
        return ""

    message = first.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        return message.content or ""

    assistant_tool_calls = []
    tool_messages = []
    for call in tool_calls:
        if call.function.name != "trigger_cinematic_video":
            continue

        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        movie_title = str(args.get("movie_title", "")).strip()
        justification = str(args.get("visual_justification", "")).strip()
        if movie_title:
            tool_result = trigger_cinematic_video(movie_title, justification)
        else:
            tool_result = "Відсутній обов'язковий аргумент: movie_title."

        assistant_tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": tool_result,
            }
        )

    if not assistant_tool_calls:
        return message.content or ""

    messages.append(
        {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": assistant_tool_calls,
        }
    )
    messages.extend(tool_messages)

    second = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        tools=tools,
        tool_choice="auto",
    )
    if not second.choices:
        return message.content or ""
    return second.choices[0].message.content or message.content or ""


def main() -> None:
    """Run the Streamlit application."""
    settings = get_settings()
    st.set_page_config(page_title=settings.app_title, page_icon="🎬", layout="wide")
    apply_modern_styles()

    load_dotenv()
    google_api_key = get_google_api_key()
    google_client: Any = None
    local_client: Any = None

    if settings.llm_provider == "google":
        if genai is None or types is None:
            st.error("Відсутня залежність google-genai. Встановіть її для провайдера Google.")
            st.stop()
        if not google_api_key:
            st.error("Ключ Google API не знайдено. Вкажіть GOOGLE_API_KEY (або GEMINI_API_KEY / AISTUDIO_API).")
            st.stop()
        google_client = genai.Client(api_key=google_api_key)
    else:
        local_client = create_local_client(base_url=settings.lmstudio_base_url, api_key=settings.lmstudio_api_key)

    movies = load_movies(settings.data_path)
    genres = genre_options(movies)
    if not genres:
        st.error("У датасеті немає доступних жанрів.")
        st.stop()

    st.markdown(
        f"""
<section class="mh-hero">
  <h1 class="mh-title">🎬 {settings.app_title}</h1>
  <p class="mh-subtitle">
    Легкий діалоговий сервіс рекомендацій фільмів на базі обраної моделі.
  </p>
  <div class="mh-chip-row">
    <span class="mh-chip">Провайдер: {settings.provider_label}</span>
    <span class="mh-chip">Модель: {settings.active_model}</span>
    <span class="mh-chip">Фільмів: {len(movies):,}</span>
    <span class="mh-chip">Жанрів: {len(genres)}</span>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.title("Система рекомендацій фільмів")
    st.sidebar.markdown("Вкажіть ваші дані та вподобання:")
    if not os.getenv("TMDB_API_KEY"):
        st.sidebar.warning("TMDB_API_KEY не знайдено. Пошук трейлерів вимкнено.")
    st.session_state["show_logs"] = st.sidebar.toggle("Показувати логи роботи агента (демо)", value=True)
    age = st.sidebar.slider("Ваш вік", 1, 100, 25)
    gender = st.sidebar.radio("Ваша стать", ("Чоловік", "Жінка", "Інше"))
    genre = st.sidebar.selectbox("Улюблений жанр", genres)

    genre_filtered = filter_by_genre(movies, genre)
    if genre_filtered.empty:
        st.warning(f"Фільмів у жанрі '{genre}' не знайдено. Будь ласка, оберіть інший жанр.")
        st.stop()

    query = st.text_input(
        "Ваш запит:",
        placeholder="Наприклад: порадь щось атмосферне на кшталт Того, хто біжить по лезу",
    )
    if not query:
        return

    with st.spinner("Виконую семантичний пошук..."):
        candidates = semantic_search(google_client, genre_filtered, query, top_k=settings.max_context_rows)
        context = build_context(candidates)

    tool_enabled = settings.llm_provider in {"google", "lmstudio"}
    prompt = build_recommendation_prompt(
        user_query=query,
        age=age,
        gender=gender,
        genre=genre,
        context=context,
        enable_cinematic_tool=tool_enabled,
    )

    st.session_state.pop("fetched_trailer", None)
    st.session_state.pop("video_justification", None)
    max_retries = 3
    retry_delay_seconds = 12.0
    response_text = ""

    st.subheader("Рекомендації")

    for attempt in range(1, max_retries + 1):
        try:
            with st.status(
                "🤖 Агент аналізує запит та використовує інструменти...",
                expanded=st.session_state.get("show_logs", False),
            ) as status:
                demo_log(
                    "caption",
                    f"🧠 Контекст підготовлено. Звертаюсь до моделі `{settings.active_model}` ({settings.provider_label}).",
                )
                if settings.llm_provider == "google":
                    response_text = generate_with_fallback(
                        client=google_client,
                        prompt=prompt,
                        preferred_model=settings.active_model,
                    )
                else:
                    response_text = generate_local_response(
                        client=local_client,
                        prompt=prompt,
                        model_name=settings.active_model,
                    )
                status.update(label="✅ Агент завершив роботу!", state="complete", expanded=False)
            break
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            error_text = str(exc).upper()
            is_rate_limit = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text
            is_last_attempt = attempt == max_retries
            if is_rate_limit and not is_last_attempt:
                st.warning(
                    f"⏳ Досягнуто ліміт API. Повтор через {retry_delay_seconds:.0f} с "
                    f"(спроба {attempt}/{max_retries})."
                )
                time.sleep(retry_delay_seconds)
                retry_delay_seconds *= 1.5
                continue

            if is_rate_limit and is_last_attempt:
                st.error("❌ API тимчасово перевантажене. Спробуйте ще раз за кілька хвилин.")
            else:
                st.error(f"Помилка запиту до моделі: {exc}")
            return

    st.markdown(response_text if response_text else "Відповідь не згенерована.")
    trailer_url = st.session_state.get("fetched_trailer")
    if trailer_url:
        st.markdown("### 🎥 Оцініть атмосферу фільму")
        justification = st.session_state.get("video_justification")
        if justification:
            st.info(f"**Чому ми показуємо це відео:** {justification}")
        st.video(trailer_url)


if __name__ == "__main__":
    main()

