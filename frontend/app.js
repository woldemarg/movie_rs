// MovieHarbor web UI: a chat over the recommender API. Spec: docs/04-interface.md.
// Every string from the API is rendered as text (never as HTML).

const GENRES_UK = {
  Action: "Бойовик", Adventure: "Пригоди", Animation: "Анімація", Comedy: "Комедія",
  Crime: "Кримінал", Documentary: "Документальний", Drama: "Драма", Family: "Сімейний",
  Fantasy: "Фентезі", History: "Історичний", Horror: "Жахи", Music: "Музика",
  Mystery: "Детектив", Romance: "Мелодрама", "Science Fiction": "Фантастика",
  "TV Movie": "Телефільм", Thriller: "Трилер", War: "Воєнний", Western: "Вестерн",
};

// Verified demo queries (README, "Демо-запити").
const SUGGESTIONS = [
  "Щось атмосферне на кшталт «Того, хто біжить по лезу»",
  "A mind-bending thriller where nothing is what it seems",
  "Порадь страшний фільм без крові, щоб лякав атмосферою",
  "Show me a film with breathtaking visuals of outer space",
];

const STORAGE_KEY = "movieharbor.preferences";
const $ = (id) => document.getElementById(id);
const state = { providers: [], provider: null, messages: [], busy: false };

// Build a DOM element; string children become text nodes, so data is never parsed as HTML.
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  el.append(...children.flat().filter((child) => child != null && child !== false));
  return el;
}

const genreName = (genre) => GENRES_UK[genre] ?? genre;

function plural(n, one, few, many) {
  const tens = n % 100, units = n % 10;
  if (units === 1 && tens !== 11) return one;
  if (units >= 2 && units <= 4 && (tens < 12 || tens > 14)) return few;
  return many;
}

function loadPreferences() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? {};
  } catch {
    return {};
  }
}

function savePreferences() {
  const preferences = {
    provider: state.provider, genre: $("genre").value, age: $("age").value, gender: $("gender").value,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Storage may be unavailable (private mode); preferences then last for the session only.
  }
}

async function init() {
  const preferences = loadPreferences();
  let config;
  try {
    config = await (await fetch("api/config")).json();
  } catch {
    showError("Не вдалося з'єднатися з сервером MovieHarbor. Перевірте, що бекенд запущено.");
    return;
  }
  state.providers = config.providers;
  const configured = config.providers.filter((p) => p.configured).map((p) => p.id);
  state.provider = configured.includes(preferences.provider)
    ? preferences.provider
    : (configured.includes("openrouter") ? "openrouter" : configured[0]);
  renderSourceSwitch();

  const genres = [...config.genres].sort((a, b) => genreName(a).localeCompare(genreName(b), "uk"));
  $("genre").append(...genres.map((g) => h("option", { value: g }, genreName(g))));
  $("genre").value = config.genres.includes(preferences.genre) ? preferences.genre : "";
  $("age").value = preferences.age ?? "";
  $("gender").value = preferences.gender ?? "";
  $("catalog-info").textContent =
    `${config.movies.toLocaleString("uk")} фільмів у каталозі` + (config.trailers ? " · трейлери TMDB" : "");

  $("suggestions").append(
    ...SUGGESTIONS.map((text) => h("button", { type: "button", class: "chip", onclick: () => ask(text) }, text)),
  );
}

function renderSourceSwitch() {
  const buttons = state.providers.map((p) =>
    h("button", {
      type: "button",
      role: "radio",
      class: "source-option",
      "aria-checked": String(p.id === state.provider),
      "data-provider": p.id,
      disabled: !p.configured,
      title: p.configured ? `Модель: ${p.model}` : "Не налаштовано: додайте LLM_API_KEY у файл .env",
      onclick: () => {
        state.provider = p.id;
        savePreferences();
        renderSourceSwitch();
      },
    }, h("span", { class: "source-label" }, p.label), h("span", { class: "source-model" }, p.model)),
  );
  $("source-switch").replaceChildren(...buttons);
}

function viewer() {
  const age = Number.parseInt($("age").value, 10);
  return {
    age: age >= 1 && age <= 120 ? age : null,
    gender: $("gender").value || null,
    genre: $("genre").value || null,
  };
}

function append(node) {
  $("welcome").hidden = true;
  $("thread").append(node);
  node.scrollIntoView({ block: "end", behavior: "smooth" });
  return node;
}

function setBusy(busy) {
  state.busy = busy;
  $("send").disabled = busy;
  $("new-chat").disabled = busy;
  $("question").readOnly = busy;
}

async function ask(text, { echo = true } = {}) {
  const question = text.trim();
  if (!question || state.busy || !state.provider) return;
  if (echo) append(h("article", { class: "msg user" }, h("p", {}, question)));
  state.messages.push({ role: "user", content: question });
  setBusy(true);
  const pending = append(
    h("article", { class: "msg assistant pending", "aria-busy": "true" },
      h("span", { class: "dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")),
      "Шукаю фільми й думаю над добіркою…"),
  );
  try {
    const response = await fetch("api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: state.provider, messages: state.messages, viewer: viewer() }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Некоректний запит.");
    state.messages.push(data.message);
    const answer = renderAnswer(data);
    pending.replaceWith(answer);
    answer.scrollIntoView({ block: "start", behavior: "smooth" }); // read from the reply down
  } catch (error) {
    state.messages.pop(); // keep user/assistant turns alternating; the retry sends it again
    const message = error instanceof TypeError ? "Немає зв'язку з сервером MovieHarbor." : error.message;
    pending.replaceWith(renderError(message, () => ask(question, { echo: false })));
  } finally {
    setBusy(false);
  }
}

function renderAnswer(data) {
  return h("article", { class: "msg assistant" },
    data.reply.split(/\n{2,}/).map((paragraph) => h("p", { class: "reply" }, paragraph)),
    data.picks.length > 0 && h("div", { class: "cards" }, data.picks.map(renderCard)),
    data.trailer && renderTrailer(data.trailer),
    renderTrace(data.trace),
  );
}

function renderCard(movie) {
  const poster = movie.poster_url
    ? h("img", {
      src: movie.poster_url, alt: `Постер «${movie.title}»`, loading: "lazy",
      onerror: (event) => event.target.replaceWith(h("span", { class: "poster-fallback" }, "🎬")),
    })
    : h("span", { class: "poster-fallback" }, "🎬");
  return h("article", { class: "card", "data-movie-id": movie.id },
    h("a", {
      class: "poster", href: `https://www.themoviedb.org/movie/${movie.id}`, target: "_blank",
      rel: "noopener", "aria-label": `«${movie.title}» на TMDB`,
    }, poster),
    h("div", { class: "card-body" },
      h("h3", {}, movie.title, movie.year && h("span", { class: "year" }, ` ${movie.year}`)),
      h("div", { class: "meta" },
        h("span", { class: "rating", title: "Зважений рейтинг TMDB" }, `★ ${movie.rating.toFixed(1)}`),
        movie.genres.map((g) => h("span", { class: "tag" }, genreName(g)))),
      h("p", { class: "why" }, movie.why),
      h("details", { class: "overview" }, h("summary", {}, "Сюжет"), h("p", {}, movie.overview)),
    ),
  );
}

function renderTrailer(trailer) {
  const play = h("button", {
    type: "button", class: "video-thumb", "aria-label": `Відтворити трейлер «${trailer.title}»`,
    style: `background-image: url("https://i.ytimg.com/vi/${encodeURIComponent(trailer.youtube_key)}/hqdefault.jpg")`,
    onclick: () => play.replaceWith(h("iframe", {
      class: "video",
      src: `https://www.youtube-nocookie.com/embed/${encodeURIComponent(trailer.youtube_key)}?autoplay=1`,
      title: `Трейлер «${trailer.title}»`,
      allow: "autoplay; encrypted-media; picture-in-picture",
      allowfullscreen: true,
    })),
  }, h("span", { class: "play", "aria-hidden": "true" }, "▶"));
  return h("section", { class: "trailer" },
    h("h3", {}, `Оцініть атмосферу: ${trailer.title}`),
    h("p", { class: "reason" }, trailer.reason),
    play,
  );
}

function renderTrace(trace) {
  const label = state.providers.find((p) => p.id === trace.provider)?.label ?? trace.provider;
  const calls = `${trace.llm_calls} ${plural(trace.llm_calls, "звернення", "звернення", "звернень")} до LLM`;
  return h("details", { class: "trace" },
    h("summary", {}, `Як це працює · ${calls} · ${(trace.latency_ms / 1000).toFixed(1)} с`),
    h("p", {}, `Модель: ${label} · ${trace.model}`),
    h("ol", {}, trace.steps.map((step) =>
      h("li", {},
        h("code", {}, step.tool),
        step.arguments.query && h("span", { class: "arg" }, ` «${step.arguments.query}»`),
        h("span", { class: "outcome" }, ` → ${step.outcome}`)))),
  );
}

function renderError(message, retry) {
  return h("article", { class: "msg assistant error", role: "alert" },
    h("p", {}, message),
    retry && h("button", {
      type: "button", class: "ghost",
      onclick: (event) => { event.target.closest(".msg").remove(); retry(); },
    }, "Спробувати ще раз"),
  );
}

function showError(message) {
  append(renderError(message, null));
}

function newChat() {
  state.messages = [];
  $("thread").replaceChildren($("welcome"));
  $("welcome").hidden = false;
  $("question").focus();
}

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) return;
  const text = $("question").value;
  $("question").value = "";
  $("question").style.height = "";
  ask(text);
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("ask-form").requestSubmit();
  }
});
$("question").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`;
});
for (const id of ["genre", "age", "gender"]) $(id).addEventListener("change", savePreferences);
$("new-chat").addEventListener("click", newChat);

init();
