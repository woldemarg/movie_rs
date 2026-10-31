// MovieHarbor web UI: a chat over the recommender API. Spec: docs/04-interface.md.
// Every string from the API is rendered as text (never as HTML).

const GENRES_UK = {
  Action: "Бойовик", Adventure: "Пригоди", Animation: "Анімація", Comedy: "Комедія",
  Crime: "Кримінал", Documentary: "Документальний", Drama: "Драма", Family: "Сімейний",
  Fantasy: "Фентезі", History: "Історичний", Horror: "Жахи", Music: "Музика",
  Mystery: "Детектив", Romance: "Мелодрама", "Science Fiction": "Фантастика",
  "TV Movie": "Телефільм", Thriller: "Трилер", War: "Воєнний", Western: "Вестерн",
};

// Demo queries validated live on the local model (README, "Демо-запити").
const SUGGESTIONS = [
  "Комедійна наукова фантастика з яскравими візуальними ефектами",
  "Пригодницький фільм про пустелю з епічними краєвидами",
  "Щось атмосферне на кшталт «Blade Runner»",
  "Порадь страшний фільм без крові, щоб лякав атмосферою",
  "Космічна пригода з краєвидами, від яких перехоплює подих",
  "Мрійлива романтична історія в Нью-Йорку, як «Амелі»",
  "Історія про подорожі в часі, що змушує замислитися",
];

const STORAGE_KEY = "movieharbor.preferences";
const POLL_MS = 3000; // how often a page waiting for the backend or the model asks again
const SLOW_START_POLLS = 200; // ≈ 10 min of waiting: then point to the logs
const MAX_HISTORY = 20; // messages sent per request; the agent reads the last 8, the API accepts 50
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
  const config = await waitForConfig();
  state.providers = config.providers;
  const ready = config.providers.filter((p) => p.ready).map((p) => p.id);
  state.provider = ready.includes(preferences.provider) ? preferences.provider : ready[0]; // local first
  renderSourceSwitch();
  watchSources();

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

async function fetchConfig() {
  try {
    const response = await fetch("api/config");
    const config = await response.json();
    return response.ok && Array.isArray(config.providers) && Array.isArray(config.genres) ? config : null;
  } catch {
    return null; // not reachable yet
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The page may be up before the backend (first start builds the catalogue), and the backend before
// the local model (first start downloads it): wait until one model source can answer.
async function waitForConfig() {
  let notice = null, shown = "";
  for (let poll = 1; ; poll++) {
    const config = await fetchConfig();
    if (config?.providers.some((p) => p.ready)) {
      if (notice) {
        notice.remove();
        $("welcome").hidden = false;
        $("send").disabled = false;
        $("question").readOnly = false;
      }
      return config;
    }
    const message = waitingMessage(config, poll >= SLOW_START_POLLS);
    if (message !== shown) {
      notice?.remove();
      notice = disableAsking(message);
      shown = message;
    }
    await sleep(POLL_MS);
  }
}

function waitingMessage(config, slow) {
  if (!config) {
    return "Сервер MovieHarbor ще запускається (перший запуск готує каталог). Сторінка підключиться сама."
      + (slow ? " Це триває довше, ніж зазвичай: перегляньте журнал (docker compose logs backend)." : "");
  }
  return "Локальна модель ще завантажується (перший запуск: ≈ 5,5 ГБ) або її сервер недоступний; OpenRouter вимагає LLM_API_KEY у файлі .env. Сторінка підключиться сама."
    + (slow ? " Це триває довше, ніж зазвичай: перегляньте журнал (docker compose logs ollama-init)." : "");
}

// A source that is configured but not ready yet (the local model still downloading) is enabled
// in the switch once it is; the page stops asking after SLOW_START_POLLS.
async function watchSources() {
  for (let poll = 0; poll < SLOW_START_POLLS; poll++) {
    if (!state.providers.some((p) => p.configured && !p.ready)) return;
    await sleep(POLL_MS);
    const config = await fetchConfig();
    if (config) {
      state.providers = config.providers;
      renderSourceSwitch();
    }
  }
}

function sourceTitle(p) {
  if (p.ready) return `Модель: ${p.model}`;
  return p.configured
    ? "Модель ще завантажується або її сервер недоступний"
    : "Не налаштовано: додайте LLM_API_KEY у файл .env";
}

function renderSourceSwitch() {
  const buttons = state.providers.map((p) =>
    h("button", {
      type: "button",
      role: "radio",
      class: "source-option",
      "aria-checked": String(p.id === state.provider),
      "data-provider": p.id,
      disabled: !p.ready,
      title: sourceTitle(p),
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

// The trace panel: the current turn's steps and facts — what the model searched for, the shortlist
// with its similarity, the trailer, the choice. Reference for the study; the chat stays uncluttered.
const failed = (step) => step.outcome.startsWith("помилка");

function stage(number, title, tool, ...body) {
  return h("div", { class: "stage" },
    h("div", { class: "stage-title" }, h("span", { class: "stage-no" }, number), title,
      tool && h("code", {}, tool)),
    ...body);
}

function candidateList(candidates, chosen) {
  const top = Math.max(...candidates.map((c) => c.score), 0.01);
  return h("ol", { class: "candidates" }, candidates.map((c, i) => {
    const pick = chosen.indexOf(c.id);
    return h("li", { class: pick >= 0 ? "chosen" : null },
      h("span", { class: "rank" }, `${i + 1}`),
      h("span", { class: "name" }, `${c.title}${c.year ? ` (${c.year})` : ""}`),
      h("span", { class: "bar" }, h("i", { style: `width: ${Math.round((c.score / top) * 100)}%` })),
      h("span", { class: "score" }, c.score.toFixed(2)),
      h("span", { class: "mark" }, pick >= 0 ? `✓ ${pick + 1}` : ""));
  }));
}

function choiceFacts(recommend, candidates) {
  const ranks = recommend.data.accepted.map((id) => candidates.findIndex((c) => c.id === id) + 1);
  return ranks.length
    ? `Обрано ${ranks.length} з ${candidates.length} · місця за схожістю: ${ranks.join(", ")}`
    : "Фільмів не обрано (запит не про кіно)";
}

function traceStages(steps) {
  const searches = steps.filter((s) => s.tool === "search_movies" && !failed(s));
  const trailer = steps.find((s) => s.tool === "show_trailer" && !failed(s));
  const recommend = steps.findLast((s) => s.tool === "recommend" && !failed(s));
  const candidates = searches.at(-1)?.data.candidates ?? [];
  const chosen = recommend?.data.accepted ?? [];
  const corrections = steps.filter(failed);
  const stages = [];
  if (searches.length) {
    stages.push(stage("1", "Запит до пошуку", "search_movies",
      searches.map((s) => h("p", { class: "decision" }, `«${s.arguments.query}»`))));
  }
  if (candidates.length) {
    stages.push(stage("2", `Кандидати (${candidates.length})`, null, candidateList(candidates, chosen)));
  }
  if (trailer) {
    stages.push(stage("3", "Трейлер", "show_trailer",
      h("p", { class: "decision" }, `${trailer.data.title} · ${trailer.data.status === "shown" ? "показано" : "не знайдено"}`),
      h("p", { class: "outcome" }, `«${trailer.arguments.reason}»`)));
  }
  if (recommend) {
    stages.push(stage(trailer ? "4" : "3", "Вибір", "recommend",
      h("p", { class: "decision" }, choiceFacts(recommend, candidates)),
      recommend.data.rejected.length > 0
        && h("p", { class: "outcome" }, `відхилено (не з пошуку): id ${recommend.data.rejected.join(", ")}`)));
  }
  if (corrections.length) {
    stages.push(stage("!", "Корекції", null,
      corrections.map((c) => h("p", { class: "outcome" }, `${c.tool}: ${c.outcome}`))));
  }
  return stages;
}

function traceHead(trace) {
  const label = state.providers.find((p) => p.id === trace.provider)?.label ?? trace.provider;
  const calls = `${trace.llm_calls} ${plural(trace.llm_calls, "звернення", "звернення", "звернень")} до LLM`;
  return `${label} · ${trace.model} · ${calls} · ${(trace.latency_ms / 1000).toFixed(1)} с`;
}

function fillTrace(entry, question, head, steps) {
  entry.replaceChildren(
    h("h3", {}, `«${question.length > 90 ? `${question.slice(0, 90)}…` : question}»`),
    h("p", { class: "trace-head" }, head),
    ...traceStages(steps));
}

function showInPanel(entry) {
  $("trace-empty").hidden = true;
  $("trace-log").replaceChildren(entry); // the panel shows the current turn
  $("trace-panel").scrollTop = 0;
}

// Parse an NDJSON response body into events as the lines arrive.
async function* streamEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

async function ask(text, { echo = true } = {}) {
  const question = text.trim();
  if (!question || state.busy || !state.provider) return;
  if (echo) append(h("article", { class: "msg user" }, h("p", {}, question)));
  state.messages.push({ role: "user", content: question });
  setBusy(true);
  const steps = [];
  const trace = h("section", { class: "trace" });
  fillTrace(trace, question, "у процесі…", steps);
  showInPanel(trace);
  const status = h("span", {}, "Думаю над запитом…");
  const pending = append(
    h("article", { class: "msg assistant pending", "aria-busy": "true" },
      h("span", { class: "dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")), status),
  );
  const fail = (message) => {
    state.messages.pop(); // keep user/assistant turns alternating; the retry sends it again
    fillTrace(trace, question, `не завершено: ${message}`, steps);
    pending.replaceWith(renderError(message, () => ask(question, { echo: false })));
  };
  try {
    let response;
    try {
      response = await fetch("api/recommend/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: state.provider, messages: state.messages.slice(-MAX_HISTORY), viewer: viewer(),
        }),
      });
    } catch {
      return fail("Немає зв'язку з сервером MovieHarbor.");
    }
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      return fail(typeof data.detail === "string" ? data.detail : `Сервер повернув помилку ${response.status}.`);
    }
    let final = null;
    try {
      for await (const event of streamEvents(response)) {
        if (event.type === "step") {
          steps.push(event);
          fillTrace(trace, question, "у процесі…", steps);
          status.textContent = event.tool === "search_movies" ? "Обираю найкращі фільми…" : "Майже готово…";
        } else {
          final = event;
        }
      }
    } catch (error) {
      return fail(error instanceof SyntaxError
        ? "Сервер повернув відповідь у неочікуваному форматі."
        : "Зв'язок із сервером обірвався. Спробуйте ще раз.");
    }
    if (final?.type === "error") return fail(final.detail);
    let answer;
    try {
      answer = renderAnswer(final);
      fillTrace(trace, question, traceHead(final.trace), final.trace.steps);
    } catch {
      return fail("Сервер повернув відповідь у неочікуваному форматі.");
    }
    state.messages.push(final.message);
    pending.replaceWith(answer);
    answer.scrollIntoView({ block: "start", behavior: "smooth" }); // read from the reply down
  } finally {
    setBusy(false);
  }
}

function renderAnswer(data) {
  return h("article", { class: "msg assistant" },
    data.reply.split(/\n{2,}/).map((paragraph) => h("p", { class: "reply" }, paragraph)),
    data.picks.length > 0 && h("div", { class: "cards" }, data.picks.map(renderCard)),
    data.trailer && renderTrailer(data.trailer),
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

function renderError(message, retry) {
  return h("article", { class: "msg assistant error", role: "alert" },
    h("p", {}, message),
    retry && h("button", {
      type: "button", class: "ghost",
      onclick: (event) => { event.target.closest(".msg").remove(); retry(); },
    }, "Спробувати ще раз"),
  );
}

function disableAsking(message) {
  const notice = append(renderError(message, null));
  $("send").disabled = true;
  $("question").readOnly = true;
  return notice;
}

function newChat() {
  state.messages = [];
  $("trace-log").replaceChildren();
  $("trace-empty").hidden = false;
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
$("panel-toggle").addEventListener("click", (event) => {
  const open = document.body.classList.toggle("panel-open");
  event.currentTarget.setAttribute("aria-expanded", String(open));
});

init();
