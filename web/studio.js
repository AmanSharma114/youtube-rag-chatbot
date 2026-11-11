const $ = (selector, root = document) => root.querySelector(selector);
const main = $("#main");
let library = {videos: [], moments: []};
let examples = [];
let current = null;
let routeVersion = 0;
let toastTimer;
let pending = false;
let preferences;
try { preferences = JSON.parse(localStorage.getItem("cueframe-settings") || "{}"); } catch { preferences = {}; }
$("#mode").value = preferences.mode || "ai";
$("#lang").value = preferences.lang || "en";

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function stamp(seconds) {
  const s = Math.floor(seconds);
  return s >= 3600 ? `${Math.floor(s/3600)}:${String(Math.floor(s/60)%60).padStart(2,"0")}:${String(s%60).padStart(2,"0")}` :
    `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`;
}
function notify(message) {
  const notice = $("#notice");
  notice.textContent = message;
  notice.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => notice.hidden = true, 6000);
}
async function api(action, data = {}) {
  const response = await fetch("/api/" + action, {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request could not finish.");
  return result;
}
function settings() { return {mode:$("#mode").value, lang:$("#lang").value.trim() || "en"}; }
$("#settings").onclick = () => $("#prefs").showModal();
$("#prefs").addEventListener("close", () => {
  try { localStorage.setItem("cueframe-settings", JSON.stringify(settings())); } catch {}
  if ($("#answer-mode")) $("#answer-mode").textContent = settings().mode === "ai" ? "Local AI · answers with transcript evidence" : "Transcript excerpts · exact words from the video";
});
function counters() {
  $("#vcount").textContent = library.videos.length;
  $("#mcount").textContent = library.moments.length;
}
async function refresh() {
  library = await api("workspace");
  counters();
}
function videoRoute(id, lang = settings().lang) {
  location.hash = "video/" + encodeURIComponent(id) + "/" + encodeURIComponent(lang);
}
function videoItem(video) {
  const button = element("button", undefined, "item");
  const text = element("div");
  text.append(element("strong", video.title), element("small", video.author));
  button.append(text, element("span", "Open →"));
  button.onclick = () => videoRoute(video.id, video.lang);
  return button;
}
function home() {
  main.replaceChildren($("#home").content.cloneNode(true));
  $("#import").onsubmit = event => {
    event.preventDefault();
    openInput($("#url").value);
  };
  for (const example of examples) {
    const button = element("button", undefined, "card");
    const image = element("img", undefined, "thumb");
    image.src = example.image;
    image.alt = "";
    const info = element("div", undefined, "details");
    info.append(element("div", example.category, "category"), element("h3", example.title),
      element("small", example.author + " · " + stamp(example.duration)));
    button.append(image, info);
    button.onclick = () => videoRoute(example.id, "en");
    $("#featured").append(button);
  }
  if (library.videos.length) {
    const head = element("div", undefined, "sectionhead");
    head.append(element("h2", "Pick up where you left off"));
    const all = element("a", "View library →");
    all.href = "#library";
    head.append(all);
    const list = element("div", undefined, "list");
    library.videos.slice(0,3).forEach(v => list.append(videoItem(v)));
    $("#recent").append(head, list);
  }
}
function openInput(value) {
  let id = value.trim();
  if (!/^[\w-]{11}$/.test(id)) {
    try {
      const url = new URL(id.includes("://") ? id : "https://" + id);
      if (url.hostname === "youtu.be") id = url.pathname.slice(1);
      else if (["youtube.com","www.youtube.com","m.youtube.com"].includes(url.hostname)) {
        id = url.searchParams.get("v") || url.pathname.split("/")[2];
      } else throw Error();
    } catch { notify("Paste a valid YouTube video link."); return; }
  }
  if (!/^[\w-]{11}$/.test(id || "")) { notify("Paste a valid YouTube video link."); return; }
  videoRoute(id);
}
function showLibrary() {
  main.replaceChildren(element("h1", "Your video library", "pagetitle"), element("p", "Transcripts and conversations saved on this computer.", "sub"));
  const list = element("div", undefined, "list");
  list.style.marginTop = "24px";
  library.videos.forEach(v => list.append(videoItem(v)));
  if (!library.videos.length) list.append(element("p", "Your first video will appear here. Open one from the Studio to begin.", "empty"));
  main.append(list);
}
function showMoments() {
  main.replaceChildren(element("h1", "Saved moments", "pagetitle"), element("p", "The passages you want to come back to.", "sub"));
  const list = element("div", undefined, "list");
  list.style.marginTop = "24px";
  for (const moment of library.moments) {
    const card = element("article", undefined, "item");
    const text = element("div");
    text.append(element("small", moment.title), element("p", moment.quote));
    const controls = element("div", undefined, "tools");
    const seek = element("button", "▶ " + stamp(moment.start), "cue");
    seek.onclick = () => {
      sessionStorage.setItem("cueframe-seek", JSON.stringify({video:moment.video, time:moment.start}));
      videoRoute(moment.video, moment.lang);
    };
    const remove = element("button", "Remove");
    remove.onclick = async () => {
      remove.disabled = true;
      try { library = await api("unsave", {id:moment.id}); counters(); showMoments(); }
      catch (error) { notify(error.message); remove.disabled = false; }
    };
    controls.append(seek, remove);
    card.append(text, controls);
    list.append(card);
  }
  if (!library.moments.length) list.append(element("p", "Save a passage from an answer to keep it here.", "empty"));
  main.append(list);
}
function seek(time) {
  const player = $("#player");
  if (!player || !current) return;
  player.src = `https://www.youtube.com/embed/${current.id}?start=${Math.floor(time)}&autoplay=1&rel=0`;
  $("#yt").href = `https://youtu.be/${current.id}?t=${Math.floor(time)}`;
  document.querySelectorAll(".caption").forEach(c => c.classList.toggle("active", Number(c.dataset.start) === Math.floor(time)));
  player.scrollIntoView({behavior:"smooth", block:"nearest"});
  notify("Selected " + stamp(time));
}
async function save(quote, meta, button) {
  button.disabled = true;
  const video = current;
  try {
    library = await api("save", {video:video.id, lang:video.lang, quote, ...meta});
    counters();
    button.textContent = "✓ Saved";
    notify("Added to saved moments");
  } catch (error) { button.disabled = false; notify(error.message); }
}
function controls(quote, meta) {
  const box = element("div", undefined, "tools");
  const cue = element("button", "▶ " + stamp(meta.start) + "–" + stamp(meta.end), "cue");
  cue.setAttribute("aria-label", "Play passage " + stamp(meta.start));
  cue.onclick = () => seek(meta.start);
  const button = element("button", "⚑ Save");
  const exists = library.moments.some(m => m.video === current.id && m.lang === current.lang && m.start === meta.start && m.quote === quote);
  if (exists) { button.textContent = "✓ Saved"; button.disabled = true; }
  button.onclick = () => save(quote, meta, button);
  box.append(cue, button);
  return box;
}
function captions(filter = "") {
  const area = $("#captions");
  area.replaceChildren();
  const rows = current.captions.filter(c => c.text.toLowerCase().includes(filter.toLowerCase()));
  for (const row of rows) {
    const line = element("button", undefined, "caption");
    line.dataset.start = Math.floor(row.start);
    line.append(element("span", stamp(row.start), "time"), element("span", row.text));
    line.onclick = () => seek(row.start);
    area.append(line);
  }
  if (!rows.length) area.append(element("p", "No captions match your search.", "empty"));
}
function turn(result) {
  const box = element("article", undefined, "turn");
  box.append(element("h3", result.question));
  if (!result.claims.length) {
    box.append(element("p", "I couldn’t find support for that in this transcript. Try asking about a specific topic the speaker mentions."));
  }
  for (const [statement, quote, meta] of result.claims) {
    box.append(element("p", statement));
    if (quote !== statement) box.append(element("blockquote", quote));
    box.append(controls(quote, meta));
  }
  $("#history").append(box);
}
async function ask(question) {
  if (pending) { notify("An answer is still being prepared. Please wait a moment."); return; }
  if (!question.trim()) return;
  pending = true;
  const owner = current;
  const version = routeVersion;
  const button = $("#ask button");
  button.disabled = true;
  button.textContent = settings().mode === "ai" ? "Reading transcript and conversation…" : "Finding passages…";
  try {
    const result = await api("ask", {video:owner.id, lang:owner.lang, question, mode:settings().mode});
    owner.turns.push(result);
    if (version !== routeVersion) return;
    $("#empty-chat")?.remove();
    turn(result);
    $("#question").value = "";
    $("#history").scrollTop = $("#history").scrollHeight;
  } catch (error) { notify(error.message); }
  finally {
    pending = false;
    if (button.isConnected) { button.disabled = false; button.textContent = "Ask question →"; }
  }
}
function exportNotes() {
  let text = `# ${current.title}\n\nhttps://youtu.be/${current.id}\n\n`;
  for (const t of current.turns) {
    text += `## ${t.question}\n\n`;
    for (const [statement, quote, meta] of t.claims) text += `${statement}\n\n> ${quote}\n\n[${stamp(meta.start)}](https://youtu.be/${current.id}?t=${meta.start})\n\n`;
    if (!t.claims.length) text += "No supporting passage found.\n\n";
  }
  const saved = library.moments.filter(m => m.video === current.id && m.lang === current.lang);
  if (saved.length) text += "## Saved moments\n\n";
  saved.forEach(m => text += `> ${m.quote}\n\n[${stamp(m.start)}](https://youtu.be/${current.id}?t=${m.start})\n\n`);
  const url = URL.createObjectURL(new Blob([text], {type:"text/markdown"}));
  const link = element("a");
  link.href = url;
  link.download = "cueframe-" + current.id + ".md";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify("Notes exported");
}
async function loadVideo(id, lang, version) {
  main.replaceChildren(element("h1", "Opening your video…", "pagetitle"), element("p", "Extracting the transcript and preparing your workspace.", "sub"));
  try {
    const video = await api("load", {video:id, lang});
    await refresh();
    if (version !== routeVersion) return;
    current = video;
    main.replaceChildren($("#video").content.cloneNode(true));
    $("#vtitle").textContent = video.title;
    $("#byline").textContent = video.author + " · " + video.captions.length + " captions · " + lang.toUpperCase();
    $("#answer-mode").textContent = settings().mode === "ai" ? "Local AI · answers with transcript evidence" : "Transcript excerpts · exact words from the video";
    $("#player").src = "https://www.youtube.com/embed/" + video.id + "?rel=0";
    $("#yt").href = "https://youtu.be/" + video.id;
    $("#export").onclick = exportNotes;
    $("#filter").oninput = e => captions(e.target.value);
    captions();
    video.turns.forEach(turn);
    if (!video.turns.length) {
      const empty = element("div", undefined, "empty");
      empty.id = "empty-chat";
      empty.append(element("p", "Your video is ready. What would you like to find?"));
      const example = examples.find(e => e.id === video.id);
      if (example) {
        const prompt = element("button", example.question + " →", "suggest");
        prompt.onclick = () => ask(example.question);
        empty.append(prompt);
      }
      $("#history").append(empty);
    }
    $("#ask").onsubmit = e => { e.preventDefault(); ask($("#question").value); };
    $("#question").onkeydown = e => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(e.target.value); }
    };
    const saved = sessionStorage.getItem("cueframe-seek");
    if (saved) {
      sessionStorage.removeItem("cueframe-seek");
      const point = JSON.parse(saved);
      if (point.video === video.id) seek(point.time);
    }
  } catch (error) {
    if (version !== routeVersion) return;
    main.replaceChildren(element("h1", "We couldn’t open this video", "pagetitle"), element("p", error.message, "empty"));
    const retry = element("button", "Try again", "primary");
    retry.onclick = () => loadVideo(id, lang, ++routeVersion);
    const back = element("a", " ← Back to studio", "back");
    back.href = "#studio";
    main.append(retry, back);
  }
}
function route() {
  const parts = location.hash.slice(1).split("/");
  const page = parts[0] || "studio";
  const version = ++routeVersion;
  current = null;
  document.querySelectorAll("[data-nav]").forEach(a => a.classList.toggle("active", a.dataset.nav === (page === "video" ? "studio" : page)));
  $("#location").textContent = {studio:"Studio",library:"Video library",moments:"Saved moments",video:"Video workspace"}[page] || "Studio";
  if (page === "video") loadVideo(decodeURIComponent(parts[1] || ""), decodeURIComponent(parts[2] || "en"), version);
  else if (page === "library") showLibrary();
  else if (page === "moments") showMoments();
  else home();
}
window.addEventListener("hashchange", route);
try {
  [examples] = await Promise.all([fetch("/examples.json").then(r => r.json()), refresh()]);
  route();
} catch (error) {
  main.replaceChildren(element("h1", "Workspace unavailable", "pagetitle"), element("p", error.message, "empty"));
}
