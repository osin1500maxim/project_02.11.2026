// ===================== Страница «Оценка» =====================
// 1) пользователь выбирает фото и вводит год и пробег
// 2) отправляем на сервер POST /api/estimate (FormData)
// 3) показываем ответ: цена, диапазон, графики, ссылка на PDF

const $ = (id) => document.getElementById(id);
const fmt = (x) => Math.round(x).toLocaleString("ru-RU");
const rub = (x) => fmt(Math.round(x / 10000) * 10000) + " ₽";
const pct = (x) => (x * 100).toFixed(0) + " %";
const ACCENT = "#2451c7", GREY = "#b8c0cf";

let file = null;          // выбранное фото
let charts = [];          // созданные графики (удаляем при новой оценке)

// ---------- выбор фото: клик или перетаскивание ----------
const drop = $("drop"), input = $("file");
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
input.addEventListener("change", () => input.files[0] && setFile(input.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

function setFile(f) {
  if (!/^image\//.test(f.type)) return showError("Нужно изображение: JPG или PNG.");
  if (f.size > 10 * 1024 * 1024) return showError("Фото больше 10 МБ — выберите поменьше.");
  file = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").hidden = false; $("change").hidden = false; $("hint").hidden = true;
  showError(null);
}

// пробег: "85000" -> "85 000" прямо при вводе
$("km").addEventListener("input", (e) => {
  const digits = e.target.value.replace(/\D/g, "").slice(0, 7);
  e.target.value = digits ? Number(digits).toLocaleString("ru-RU") : "";
});
$("year").max = new Date().getFullYear();

function showError(msg) {
  $("formError").hidden = !msg;
  $("formError").textContent = msg || "";
}

// ---------- отправка ----------
$("form").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("recalc").addEventListener("click", () => send($("manualModel").value.trim()));
$("again").addEventListener("click", () => {
  $("result").classList.remove("show");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

async function send(manualModel = "") {
  const year = parseInt($("year").value, 10);
  const km = parseInt($("km").value.replace(/\D/g, ""), 10);
  const nowYear = new Date().getFullYear();
  if (!file) return showError("Загрузите фото автомобиля.");
  if (!year || year < 1990 || year > nowYear) return showError(`Укажите год выпуска от 1990 до ${nowYear}.`);
  if (isNaN(km) || km < 0) return showError("Укажите пробег в километрах.");
  showError(null);

  const fd = new FormData();
  fd.append("photo", file);
  fd.append("year", year);
  fd.append("km", km);
  if (manualModel) fd.append("model", manualModel);

  setLoading(true);
  try {
    const res = await fetch("/api/estimate", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Ошибка сервера");
    render(data);
  } catch (err) {
    showError(err.message || "Не удалось связаться с сервером.");
    window.scrollTo({ top: 0, behavior: "smooth" });
  } finally {
    setLoading(false);
  }
}

let loadingTimer = null;
function setLoading(on) {
  $("submit").disabled = on; $("recalc").disabled = on;
  $("loading").classList.toggle("show", on);
  clearInterval(loadingTimer);
  if (on) {
    $("result").classList.remove("show");
    const steps = ["Убираем фон с фото…", "Распознаём марку и модель…", "Считаем цену…", "Готовим отчёт…"];
    let i = 0; $("loadingText").textContent = steps[0];
    loadingTimer = setInterval(() => { i = Math.min(i + 1, steps.length - 1); $("loadingText").textContent = steps[i]; }, 1500);
    $("loading").scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

// ---------- вывод результата ----------
function render(d) {
  charts.forEach((c) => c.destroy()); charts = [];

  $("rPhoto").src = d.photo;
  $("rModel").textContent = d.model;
  $("rModelNote").textContent = d.manual_model ? "модель указана вручную" : "распознано по фото";
  $("rChips").innerHTML = d.candidates
    .map((c, i) => `<span class="chip ${i === 0 && !d.manual_model ? "main" : ""}">${c.name} · ${pct(c.prob)}</span>`)
    .join("");
  $("rSpecs").innerHTML = `
    <dt>Год выпуска</dt><dd>${d.year}</dd>
    <dt>Пробег</dt><dd>${fmt(d.km)} км</dd>
    <dt>Цвет</dt><dd>${d.color}</dd>
    <dt>Кузов</dt><dd>${d.body}</dd>`;

  $("rPrice").textContent = rub(d.price_rub);
  $("rLow").textContent = rub(d.low_rub);
  $("rHigh").textContent = rub(d.high_rub);
  const pos = (d.price_rub - d.low_rub) / Math.max(d.high_rub - d.low_rub, 1);
  $("rDot").style.left = Math.min(Math.max(pos, 0.05), 0.95) * 100 + "%";
  $("rRate").textContent = `≈ £${fmt(d.price_gbp)} · 1 £ = ${d.rate.value.toFixed(2)} ₽ (${d.rate.source}, ${d.rate.date})`;
  $("rWarn").innerHTML = d.warnings.map((w) => `<div class="warn">⚠ ${w}</div>`).join("");
  $("rPdf").href = d.report_url;
  $("manualModel").value = "";

  $("sYear").textContent = "−" + pct(d.year_loss);
  $("s10k").textContent = rub(d.per_10k_km).replace(" ₽", "") + " ₽";
  $("sKm").textContent = fmt(d.km_per_year.yours / 1000) + " тыс. км";
  $("sKmCap").textContent = `ваш пробег в год (типичный — ${fmt(d.km_per_year.typical / 1000)} тыс.)`;
  if (d.new_price) {
    $("sNew").textContent = pct(d.new_price.share);
    $("sNewCap").textContent = `от цены новой в ${d.year} г. (£${fmt(d.new_price.gbp)})`;
  } else {
    $("sNew").textContent = "—"; $("sNewCap").textContent = "цена новой неизвестна";
  }

  $("rMarket").innerHTML = d.market
    ? `Найдено <b>${d.market.n}</b> объявлений о продаже этой модели того же возраста и с похожим пробегом.
       Половина из них — от <b>${rub(d.market.p25)}</b> до <b>${rub(d.market.p75)}</b> (в пересчёте на сегодня).`
    : "Похожих объявлений в датасете слишком мало для сравнения.";

  drawCharts(d);
  $("result").classList.add("show");
  $("result").scrollIntoView({ behavior: "smooth" });
}

// ---------- графики (Chart.js) ----------
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.color = "#5b6475";
const mln = (v) => (v / 1e6).toFixed(1) + " млн";

function drawCharts(d) {
  const yourIdx = d.curve.years.indexOf(d.year);
  charts.push(new Chart($("cCurve"), {
    type: "line",
    data: {
      labels: d.curve.years,
      datasets: [{
        data: d.curve.prices, borderColor: GREY, tension: .3,
        pointRadius: d.curve.years.map((_, i) => (i === yourIdx ? 7 : 3)),
        pointBackgroundColor: d.curve.years.map((_, i) => (i === yourIdx ? ACCENT : GREY)),
      }],
    },
    options: baseOpts({ y: { ticks: { callback: mln } } }, (c) => rub(c.parsed.y)),
  }));

  charts.push(new Chart($("cForecast"), {
    type: "bar",
    data: {
      labels: ["сейчас", "через 1 год", "через 2 года", "через 3 года"],
      datasets: [{ data: d.forecast, backgroundColor: [ACCENT, GREY, GREY, GREY], borderRadius: 8 }],
    },
    options: baseOpts({ y: { ticks: { callback: mln } } }, (c) => rub(c.parsed.y)),
  }));

  charts.push(new Chart($("cColors"), {
    type: "bar",
    data: {
      labels: d.colors.map((c) => c.color),
      datasets: [{ data: d.colors.map((c) => c.share * 100), borderRadius: 8,
                   backgroundColor: d.colors.map((c) => (c.is_yours ? ACCENT : GREY)) }],
    },
    options: baseOpts({ y: { ticks: { callback: (v) => v + " %" } } }, (c) => {
      const col = d.colors[c.dataIndex];
      const eff = (col.effect >= 0 ? "+" : "−") + Math.abs(col.effect * 100).toFixed(1) + " % к цене";
      return `${c.parsed.y.toFixed(0)} % объявлений · ${eff}`;
    }),
  }));

  const sales = d.history.sales;
  charts.push(new Chart($("cSales"), {
    type: "bar",
    data: {
      labels: Object.keys(sales),
      datasets: [{ data: Object.values(sales), borderRadius: 4,
                   backgroundColor: Object.keys(sales).map((y) => (+y === d.year ? ACCENT : GREY)) }],
    },
    options: baseOpts({}, (c) => fmt(c.parsed.y) + " шт."),
  }));
}

function baseOpts(scales, label) {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { callbacks: { label } } },
    scales: { x: { grid: { display: false } }, y: { grid: { color: "#eef0f4" }, ...(scales.y || {}) } },
  };
}

// ---------- список моделей для ручного выбора ----------
fetch("/api/info").then((r) => r.json()).then((info) => {
  $("modelsList").innerHTML = info.models.map((m) => `<option value="${m}">`).join("");
}).catch(() => {});
