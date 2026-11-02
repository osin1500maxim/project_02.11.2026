"""
ЯДРО САЙТА: всё, что связано с моделями.

Модели загружаются ОДИН раз при старте сервера (это долго — несколько секунд),
а потом каждый запрос только считает: фото -> распознавание -> цена -> аналитика -> PDF.

Файлы моделей лежат в папке models/ (или в папке из переменной окружения MODEL_DIR):
    car_model.pt         нейросеть (из train.py)
    price_model.joblib   модель цены (из train.py)
    train.py             код обучения — в нём описаны классы моделей, без него их не загрузить
    metrics.txt, training_curves.png   — необязательно, показываются на странице «О проекте»
"""
import datetime as dt
import io
import os
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET

import joblib
import matplotlib

matplotlib.use("Agg")                     # рисуем графики без экрана (на сервере его нет)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torchvision import transforms as T  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
sys.path.insert(0, MODEL_DIR)             # чтобы импортировать train.py из папки models
import train  # noqa: E402

# price_model.joblib сохранялся при запуске `python train.py`, поэтому внутри файла класс записан
# как __main__.PriceModel. На сервере __main__ — это uvicorn, поэтому подсказываем, где искать класс.
import __main__  # noqa: E402

__main__.PriceModel = train.PriceModel

KM_PER_MILE = 1.609344
FALLBACK_RATE, FALLBACK_DATE = 107.6574, "29.10.2026"
COLORS_RU = {"Black": "чёрный", "White": "белый", "Silver": "серебристый", "Grey": "серый", "Blue": "синий",
             "Red": "красный", "Green": "зелёный", "Yellow": "жёлтый", "Orange": "оранжевый", "Brown": "коричневый",
             "Beige": "бежевый", "Gold": "золотистый", "Purple": "фиолетовый", "Bronze": "бронзовый",
             "Maroon": "бордовый", "Turquoise": "бирюзовый", "Pink": "розовый", "Multicolour": "разноцветный"}
BODY_RU = {"Saloon": "седан", "Hatchback": "хэтчбек", "Estate": "универсал", "SUV": "внедорожник", "MPV": "минивэн",
           "Coupe": "купе", "Convertible": "кабриолет", "Pickup": "пикап", "Limousine": "лимузин"}

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


def n(x):
    return f"{int(round(x)):,}".replace(",", " ")


def rub(x):
    return n(round(x, -4)) + " ₽"


# ------------------------------------------------------------------ курс ЦБ (кэш на 6 часов)
_rate_cache = {"t": 0, "v": None}


def gbp_rate():
    if _rate_cache["v"] and time.time() - _rate_cache["t"] < 6 * 3600:
        return _rate_cache["v"]
    try:
        req = urllib.request.Request("https://www.cbr.ru/scripts/XML_daily.asp", headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(urllib.request.urlopen(req, timeout=6).read())
        for v in root.findall("Valute"):
            if v.findtext("CharCode") == "GBP":
                val = (float(v.findtext("Value").replace(",", ".")) / float(v.findtext("Nominal")),
                       root.attrib["Date"], "ЦБ РФ")
                _rate_cache.update(t=time.time(), v=val)
                return val
    except Exception:
        pass
    val = (FALLBACK_RATE, FALLBACK_DATE, "запасной курс ЦБ (сайт ЦБ недоступен)")
    _rate_cache.update(t=time.time() - 6 * 3600 + 600, v=val)    # повторим попытку через 10 минут
    return val


# ------------------------------------------------------------------ модели
class Models:
    """Держит модели в памяти. Создаётся один раз при старте сайта."""

    def __init__(self):
        t0 = time.time()
        ck = torch.load(os.path.join(MODEL_DIR, "car_model.pt"), map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        self.net = train.CarNet(len(self.classes["model"]), len(self.classes["color"]), len(self.classes["body"]),
                                pretrained=False)
        self.net.load_state_dict(ck["weights"])
        self.net.eval()
        s = ck["img_size"]
        self.tf = T.Compose([T.Resize(s), T.CenterCrop(s), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        d = joblib.load(os.path.join(MODEL_DIR, "price_model.joblib"))
        self.price_model, self.table, self.market, self.sales, self.quality = (
            d["model"], d["new_prices"], d["market"], d["sales"], d["quality"])
        self.models_list = sorted(set(self.classes["model"]) & set(self.market.model))
        self.lock = threading.Lock()      # нейросеть считает по одному запросу за раз — бережём память
        self.bg = None
        try:                               # модель удаления фона (u2net, ~170 МБ)
            from rembg import new_session
            self.bg = new_session("u2net")
        except Exception as e:
            print(f"[!] удаление фона недоступно: {e}")
        print(f"Модели загружены за {time.time() - t0:.1f} c: {len(self.models_list)} моделей авто")

    # --- фото
    def prepare_photo(self, data: bytes) -> Image.Image:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        img.thumbnail((1600, 1600))                       # большие фото уменьшаем — быстрее
        if self.bg is not None:
            from rembg import remove
            cut = remove(img.convert("RGB"), session=self.bg)
            box = cut.split()[-1].point(lambda a: 255 if a > 20 else 0).getbbox()
            img = train.to_rgb(cut.crop(box)) if box else img.convert("RGB")
        else:
            img = img.convert("RGB")
        s = max(img.size)
        sq = Image.new("RGB", (s, s), (255, 255, 255))
        sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
        return sq.resize((300, 300))

    def recognize(self, img):
        with self.lock, torch.no_grad():
            outs = [torch.softmax(o, 1)[0].numpy() for o in self.net(self.tf(img)[None])]
        res = {}
        for k, p in zip(("model", "color", "body"), outs):
            order = [i for i in np.argsort(-p) if k != "model" or self.classes[k][i] in self.models_list]
            res[k] = [(self.classes[k][i], float(p[i])) for i in order[:3]]
        return res

    # --- цена
    def info(self, model):
        mk = self.market[self.market.model == model]
        return mk, mk.genmodel_id.mode()[0], mk.maker.iloc[0]

    def price(self, model, year, miles, color, body, now=None):
        now = now or dt.date.today()
        mk, gid, maker = self.info(model)
        year, miles = np.atleast_1d(year).astype(float), np.atleast_1d(miles).astype(float)
        df = pd.DataFrame({"model": model, "maker": maker, "body": body, "color": color, "genmodel_id": gid,
                           "reg_year": year, "mileage": miles,
                           "age": np.clip(now.year + now.month / 12 - year - 0.5, 0.1, None)})
        return self.price_model.predict(train.price_features(df, self.table))

    # --- главная функция
    def estimate(self, photo: bytes, year: int, km: float, model_override: str | None = None) -> dict:
        today = dt.date.today()
        img = self.prepare_photo(photo)
        rec = self.recognize(img)
        model = model_override or rec["model"][0][0]
        color, body = rec["color"][0][0], rec["body"][0][0]
        miles = km / KM_PER_MILE
        rate, rate_date, rate_src = gbp_rate()
        p, lo, hi = (float(v[0]) for v in self.price(model, year, miles, color, body))
        mk, gid, _ = self.info(model)

        warnings = []
        if not model_override and rec["model"][0][1] < 0.4:
            warnings.append(f"Модель распознана неуверенно ({rec['model'][0][1]:.0%}). Если это другая машина — "
                            "выберите модель вручную.")
        if year > 2021:
            warnings.append("Машина новее данных датасета (до 2021 г.) — точность ниже.")
        if today.year - year > 14:
            warnings.append("Машина старше, чем машины в обучающих данных, — точность ниже.")

        # аналитика
        mpy = float((mk.mileage / mk.age).median())
        years = np.arange(max(year - 6, int(mk.reg_year.min())), min(today.year, max(int(mk.reg_year.max()), year)) + 1)
        curve = self.price(model, years, mpy * (today.year - years + 0.5), color, body)[0]
        user_mpy = miles / max(today.year - year + 0.5, 0.5)
        future = [float(self.price(model, year, miles + user_mpy * k, color, body,
                                   dt.date(today.year + k, today.month, 1))[0][0]) for k in (1, 2, 3)]
        d10 = 10000 / KM_PER_MILE / 2
        per10k = float(self.price(model, year, max(miles - d10, 0), color, body)[0][0]
                       - self.price(model, year, miles + d10, color, body)[0][0])
        top_colors = mk.color.value_counts(normalize=True).head(7)
        colors = [{"color": COLORS_RU.get(c, c), "share": float(s), "is_yours": c == color,
                   "effect": float(self.price(model, year, miles, c, body)[0][0] / p - 1)}
                  for c, s in top_colors.items()]
        comp = mk[(mk.age.between(today.year - year - 1, today.year - year + 2)) &
                  (mk.mileage.between(miles * 0.7, miles * 1.3 + 3000))]
        newp = self.table.get(str(gid), {})
        sales = self.sales[self.sales.gid == str(gid)].sort_values("year") if self.sales is not None else None

        R = lambda gbp: float(gbp * rate)  # noqa: E731
        result = {
            "model": model, "manual_model": bool(model_override),
            "candidates": [{"name": m, "prob": pr} for m, pr in rec["model"]],
            "color": COLORS_RU.get(color, color), "color_raw": color,
            "body": BODY_RU.get(body, body), "body_raw": body,
            "year": year, "km": km,
            "price_rub": R(p), "low_rub": R(lo), "high_rub": R(hi), "price_gbp": p,
            "rate": {"value": rate, "date": rate_date, "source": rate_src},
            "warnings": warnings,
            "curve": {"years": [int(y) for y in years], "prices": [R(v) for v in curve]},
            "forecast": [R(p)] + [R(v) for v in future],
            "per_10k_km": R(per10k),
            "km_per_year": {"yours": user_mpy * KM_PER_MILE, "typical": mpy * KM_PER_MILE},
            "year_loss": 1 - future[0] / p,
            "colors": colors,
            "market": ({"n": int(len(comp)), "p25": R(comp.price_today.quantile(.25)),
                        "p75": R(comp.price_today.quantile(.75))} if len(comp) >= 5 else None),
            "new_price": ({"gbp": float(train.new_price(self.table, gid, year)),
                           "share": p / float(train.new_price(self.table, gid, year))} if newp else None),
            "history": {"new_price": {str(k): float(v) for k, v in sorted(newp.items())},
                        "sales": ({str(int(r.year)): float(r.sales) for r in sales.itertuples()}
                                  if sales is not None and len(sales) else {})},
        }
        result["_img"] = img                       # для PDF (в ответ сайту не уходит)
        result["_curve_years"], result["_curve"] = years, curve * rate
        return result


# ------------------------------------------------------------------ PDF
def build_pdf(r: dict, quality: dict) -> bytes:
    today = dt.date.today()
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .955, "Оценка автомобиля по фото", fontsize=20, weight="bold", color="#2451c7")
        fig.text(.07, .935, f"Отчёт от {today:%d.%m.%Y}", fontsize=9, color="gray")
        ax = fig.add_axes([.07, .68, .38, .24]); ax.imshow(r["_img"]); ax.axis("off")
        fig.text(.5, .9, r["model"], fontsize=15, weight="bold", va="top")
        lines = ["Распознано по фото:"] + [f"  {c['name']} — {c['prob']:.0%}" for c in r["candidates"]] + [
            "", f"Год выпуска: {r['year']}", f"Пробег: {n(r['km'])} км", f"Цвет: {r['color']}", f"Кузов: {r['body']}"]
        fig.text(.5, .865, "\n".join(lines), fontsize=10, va="top", linespacing=1.6)
        fig.patches.append(plt.Rectangle((.06, .55), .88, .115, transform=fig.transFigure, color="#eef2fb"))
        fig.text(.08, .645, "Оценка рыночной стоимости", fontsize=10)
        fig.text(.08, .6, rub(r["price_rub"]), fontsize=30, weight="bold", color="#2451c7")
        rt = r["rate"]
        fig.text(.08, .57, f"Диапазон: {rub(r['low_rub'])} — {rub(r['high_rub'])}   ·   £{n(r['price_gbp'])}   ·   "
                 f"1 £ = {rt['value']:.2f} ₽ ({rt['source']}, {rt['date']})", fontsize=8.5)
        if r["warnings"]:
            fig.text(.07, .53, "\n".join("⚠ " + w for w in r["warnings"]), fontsize=9, color="#b35c00", va="top")
        ax = fig.add_axes([.1, .27, .82, .22])
        ax.plot(r["_curve_years"], r["_curve"] / 1e6, "o-", color="#9aa3b2", label="эта модель при типичном пробеге")
        ax.scatter([r["year"]], [r["price_rub"] / 1e6], s=90, color="#2451c7", zorder=5, label="ваша машина")
        ax.set_ylabel("млн ₽"); ax.set_title("Цена в зависимости от года выпуска", loc="left", fontsize=11)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax = fig.add_axes([.1, .06, .82, .15])
        ax.bar(["сейчас", "через 1 год", "через 2 года", "через 3 года"], np.array(r["forecast"]) / 1e6,
               color=["#2451c7"] + ["#9aa3b2"] * 3)
        for i, v in enumerate(r["forecast"]):
            ax.text(i, v / 1e6, rub(v), ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Прогноз при вашем темпе пробега ({n(r['km_per_year']['yours'])} км/год)", loc="left", fontsize=11)
        ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top", "right", "left")]
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .955, f"Аналитика: {r['model']}", fontsize=16, weight="bold")
        txt = [f"• Каждые 10 000 км пробега меняют цену примерно на {rub(r['per_10k_km'])}.",
               f"• Типичный пробег этой модели — {n(r['km_per_year']['typical'])} км/год, ваш — "
               f"{n(r['km_per_year']['yours'])} км/год.",
               f"• За год машина подешевеет примерно на {r['year_loss']:.0%}."]
        if r["market"]:
            m = r["market"]
            txt.append(f"• Похожие объявления ({m['n']} шт.): половина цен от {rub(m['p25'])} до {rub(m['p75'])}.")
        if r["new_price"]:
            txt.append(f"• Новая в {r['year']} г. стоила £{n(r['new_price']['gbp'])}; сейчас — "
                       f"{r['new_price']['share']:.0%} от этой цены (без поправки на инфляцию).")
        fig.text(.07, .93, "\n".join(txt), fontsize=10, va="top", linespacing=1.8)
        if r["colors"]:
            ax = fig.add_axes([.1, .58, .82, .17])
            cs = r["colors"]
            ax.bar([c["color"] for c in cs], [c["share"] * 100 for c in cs],
                   color=["#2451c7" if c["is_yours"] else "#9aa3b2" for c in cs])
            for i, c in enumerate(cs):
                ax.text(i, c["share"] * 100, f"{c['share']:.0%}\n{c['effect']:+.1%} к цене", ha="center",
                        va="bottom", fontsize=7.5)
            ax.set_ylim(0, max(c["share"] for c in cs) * 140); ax.set_ylabel("% объявлений")
            ax.set_title("Цвета этой модели: популярность и влияние на цену", loc="left", fontsize=11)
        h = r["history"]
        if h["new_price"]:
            ax = fig.add_axes([.1, .36, .36, .15])
            ax.plot([int(k) for k in h["new_price"]], list(h["new_price"].values()), "o-", ms=3)
            ax.axvline(r["year"], color="red", ls="--", lw=1)
            ax.set_title("Цена новой, £", loc="left", fontsize=10); ax.grid(alpha=.3)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
        if h["sales"]:
            ax = fig.add_axes([.56, .36, .36, .15])
            ax.bar([int(k) for k in h["sales"]], list(h["sales"].values()), color="#9aa3b2")
            ax.axvline(r["year"], color="red", ls="--", lw=1)
            ax.set_title("Продажи новых в Великобритании", loc="left", fontsize=10)
        q = quality
        method = (
            "Как получена оценка\n"
            "1. С фото удаляется фон; нейросеть ConvNeXt-Tiny, дообученная на фото DVM-CAR, определяет модель, цвет, кузов.\n"
            "2. Градиентный бустинг считает цену по модели, году, пробегу, цвету, кузову и цене новой машины.\n"
            f"3. Обучено на {n(q['n_ads'])} объявлениях и {n(q['n_images'])} фото (Великобритания).\n"
            "4. Цены приведены к 2026 г. по индексу ONS (подержанные авто) и переведены по курсу ЦБ РФ.\n\n"
            "Точность на тестовой выборке\n"
            f"• ошибка цены (MAPE), модель известна: {q['price_test_mape']:.1%}\n"
            f"• ошибка цены по фото (распознавание + год + пробег): {q['e2e_mape']:.1%}\n"
            f"• точность распознавания модели: {q['recognition']:.0%}\n"
            f"• доля реальных цен внутри диапазона: {q['coverage']:.0%}\n\n"
            "Ограничения: это цены британского рынка в пересчёте на рубли, а не цена в РФ; по фото не видно\n"
            "состояние, ДТП и комплектацию. Отчёт не является оценочным заключением.")
        fig.text(.07, .3, method, fontsize=8.8, va="top", linespacing=1.6)
        pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()
