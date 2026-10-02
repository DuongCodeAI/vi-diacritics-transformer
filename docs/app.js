// Cùng logic với src/vi_diacritics/labels.py: mỗi ký tự = chữ gốc + shape + tone.
const SHAPES = ["", "̂", "̆", "̛", "stroke"];
const TONES = ["", "̀", "́", "̉", "̃", "̣"];
const AMBIG = new Set("aeiouyd");
const LOW_CONF = 0.8;

let session = null;
let stoi = null;

function decompose(ch) {
  if (ch === "đ") return ["d", 4, 0];
  if (ch === "Đ") return ["D", 4, 0];
  const nfd = ch.normalize("NFD");
  let shape = 0, tone = 0, other = false;
  for (const m of nfd.slice(1)) {
    const s = SHAPES.indexOf(m), t = TONES.indexOf(m);
    if (s > 0 && s < 4) shape = s;
    else if (t > 0) tone = t;
    else other = true;
  }
  if (other) return [ch, 0, 0];
  return [nfd[0], shape, tone];
}

function compose(base, label) {
  const shape = Math.floor(label / TONES.length), tone = label % TONES.length;
  if (shape === 4) return base === "d" ? "đ" : base === "D" ? "Đ" : base;
  return (base + (shape ? SHAPES[shape] : "") + (tone ? TONES[tone] : "")).normalize("NFC");
}

async function load() {
  const vocab = await (await fetch("model/vocab.json")).json();
  stoi = new Map(vocab.itos.map((c, i) => [c, i]));
  ort.env.wasm.numThreads = 1;
  try {
    session = await ort.InferenceSession.create("model/tagger.int8.onnx");
  } catch (e) {
    console.warn("int8 lỗi, dùng fp32", e);
    session = await ort.InferenceSession.create("model/tagger.onnx");
  }
}

async function restore(text) {
  const chars = [...text.normalize("NFC")];
  const parts = chars.map(decompose);
  const ids = parts.map(([b]) => stoi.get(b) ?? 1);
  const t0 = performance.now();
  const res = await session.run({ ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]) });
  const ms = performance.now() - t0;
  const probs = res.probs.data, L = res.probs.dims[2];
  const out = [];
  for (let i = 0; i < chars.length; i++) {
    const [base, s, t] = parts[i];
    if (s || t) { out.push({ ch: chars[i], conf: 1 }); continue; } // giữ dấu người dùng đã gõ
    let best = 0, p = -1;
    for (let k = 0; k < L; k++) if (probs[i * L + k] > p) { p = probs[i * L + k]; best = k; }
    out.push({ ch: compose(base, best), conf: AMBIG.has(base.toLowerCase()) ? p : 1 });
  }
  return { out, ms };
}

function render({ out, ms }) {
  const el = document.getElementById("out");
  el.innerHTML = "";
  // gom theo từ để tô màu cả từ kém tự tin
  let word = [], minConf = 1;
  const flush = () => {
    if (!word.length) return;
    const span = document.createElement("span");
    span.textContent = word.join("");
    if (minConf < LOW_CONF) { span.className = "low"; span.title = `tự tin ${(minConf * 100).toFixed(0)}%`; }
    el.appendChild(span);
    word = []; minConf = 1;
  };
  for (const { ch, conf } of out) {
    if (/[\p{L}\p{N}]/u.test(ch)) { word.push(ch); minConf = Math.min(minConf, conf); }
    else { flush(); el.appendChild(document.createTextNode(ch)); }
  }
  flush();
  document.getElementById("meta").textContent = `${out.length} ký tự · ${ms.toFixed(1)} ms`;
}

let timer = null;
const input = document.getElementById("in");
const run = () => { if (session && input.value.trim()) restore(input.value).then(render); };
input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 120); });
document.querySelectorAll(".ex button").forEach(b => b.onclick = () => { input.value = b.textContent; run(); });

load().then(run).catch(e => { document.getElementById("out").textContent = "Không tải được model: " + e; });
