"""
ОЦЕНКА МАШИНЫ ПО ФОТО -> PDF-отчёт

Запуск:   python predict.py фото.jpg 2017 85000
          (фото, год выпуска, пробег в км — все три обязательны)
или просто python predict.py — программа спросит сама.

Нужны файлы после обучения (в той же папке): car_model.pt, price_model.joblib
Результат: otchet_<модель>_<год>.pdf
"""
import datetime as dt
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from train import CarNet, PriceModel, new_price, price_features, to_rgb  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
KM_PER_MILE = 1.609344
FALLBACK_RATE, FALLBACK_DATE = 107.6574, "29.10.2026"      # курс ЦБ, если нет интернета
COLORS_RU = {"Black": "чёрный", "White": "белый", "Silver": "серебристый", "Grey": "серый", "Blue": "синий",
             "Red": "красный", "Green": "зелёный", "Yellow": "жёлтый", "Orange": "оранжевый", "Brown": "коричневый",
             "Beige": "бежевый", "Gold": "золотистый", "Purple": "фиолетовый", "Bronze": "бронзовый"}


def n(x):
    """12345 -> '12 345'"""
    return f"{int(round(x)):,}".replace(",", " ")


def rub(x):
    return n(round(x, -4)) + " ₽"


# ------------------------------------------------------------------ курс ЦБ
def gbp_rate():
    try:
        req = urllib.request.Request("https://www.cbr.ru/scripts/XML_daily.asp", headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(urllib.request.urlopen(req, timeout=8).read())
        for v in root.findall("Valute"):
            if v.findtext("CharCode") == "GBP":
                return float(v.findtext("Value").replace(",", ".")) / float(v.findtext("Nominal")), root.attrib["Date"], "ЦБ РФ"
    except Exception:
        pass
    return FALLBACK_RATE, FALLBACK_DATE, "запасной курс ЦБ (нет интернета)"


# ------------------------------------------------------------------ фото
def prepare_photo(file):
    img = ImageOps.exif_transpose(Image.open(file))
    try:                                   # удаляем фон, как на фото датасета
        from rembg import new_session, remove
        cut = remove(img.convert("RGB"), session=new_session("u2net"))
        box = cut.split()[-1].point(lambda a: 255 if a > 20 else 0).getbbox()
        img = to_rgb(cut.crop(box)) if box else img.convert("RGB")
    except Exception:
        img = img.convert("RGB")
    s = max(img.size)                      # квадрат на белом фоне
    sq = Image.new("RGB", (s, s), (255, 255, 255))
    sq.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return sq.resize((300, 300))


def recognize(img):
    ck = torch.load(os.path.join(HERE, "car_model.pt"), map_location="cpu", weights_only=False)
    cls = ck["classes"]
    net = CarNet(len(cls["model"]), len(cls["color"]), len(cls["body"]), pretrained=False)
    net.load_state_dict(ck["weights"])
    net.eval()
    s = ck["img_size"]
    x = T.Compose([T.Resize(s), T.CenterCrop(s), T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img)[None]
    with torch.no_grad():
        outs = [torch.softmax(o, 1)[0].numpy() for o in net(x)]
    return {k: [(cls[k][i], float(p[i])) for i in np.argsort(-p)[:3]] for k, p in zip(("model", "color", "body"), outs)}


# ------------------------------------------------------------------ цена
class Pricer:
    def __init__(self):
        d = joblib.load(os.path.join(HERE, "price_model.joblib"))
        self.m, self.table, self.market, self.sales, self.quality = (d["model"], d["new_prices"], d["market"],
                                                                     d["sales"], d["quality"])

    def info(self, model):
        mk = self.market[self.market.model == model]
        return mk, mk.genmodel_id.mode()[0], mk.maker.iloc[0]

    def price(self, model, year, miles, color, body, now=None):
        """Цена (£, сегодняшние цены) и интервал P10–P90. year/miles могут быть массивами."""
        now = now or dt.date.today()
        mk, gid, maker = self.info(model)
        year, miles = np.atleast_1d(year).astype(float), np.atleast_1d(miles).astype(float)
        df = pd.DataFrame({"model": model, "maker": maker, "body": body, "color": color, "genmodel_id": gid,
                           "reg_year": year, "mileage": miles,
                           "age": np.clip(now.year + now.month / 12 - year - 0.5, 0.1, None)})
        return self.m.predict(price_features(df, self.table))


# ------------------------------------------------------------------ отчёт
def main():
    a = sys.argv[1:]
    photo = a[0] if a else input("Путь к фото: ").strip('" ')
    year = int(a[1] if len(a) > 1 else input("Год выпуска: "))
    km = float((a[2] if len(a) > 2 else input("Пробег, км: ")).replace(" ", ""))
    today = dt.date.today()
    if not 1990 <= year <= today.year or km < 0:
        sys.exit("Проверьте год выпуска и пробег.")

    img = prepare_photo(photo)
    rec = recognize(img)
    model, color, body = rec["model"][0][0], rec["color"][0][0], rec["body"][0][0]
    P = Pricer()
    miles = km / KM_PER_MILE
    rate, rate_date, rate_src = gbp_rate()
    p, lo, hi = (v[0] for v in P.price(model, year, miles, color, body))
    mk, gid, _ = P.info(model)

    warnings = []
    if rec["model"][0][1] < 0.4:
        warnings.append(f"Модель распознана неуверенно ({rec['model'][0][1]:.0%}) — проверьте варианты.")
    if year > 2021:
        warnings.append("Машина новее данных датасета (до 2021 г.) — точность ниже.")
    if today.year - year > 14:
        warnings.append("Машина старше, чем в обучающих данных, — точность ниже.")

    # --- аналитика
    mpy = float((mk.mileage / mk.age).median())                          # типичный пробег в год, мили
    years = np.arange(max(year - 6, int(mk.reg_year.min())), min(today.year, max(int(mk.reg_year.max()), year)) + 1)
    curve = P.price(model, years, mpy * (today.year - years + 0.5), color, body)[0]
    user_mpy = miles / max(today.year - year + 0.5, 0.5)
    future = [P.price(model, year, miles + user_mpy * k, color, body, dt.date(today.year + k, today.month, 1))[0][0]
              for k in (1, 2, 3)]
    per10k = (P.price(model, year, max(miles - 3107, 0), color, body)[0][0] -
              P.price(model, year, miles + 3107, color, body)[0][0]) / 2
    top_colors = mk.color.value_counts(normalize=True).head(7)
    color_eff = [P.price(model, year, miles, c, body)[0][0] / p - 1 for c in top_colors.index]
    comp = mk[(mk.age.between(today.year - year - 1, today.year - year + 2)) &
              (mk.mileage.between(miles * 0.7, miles * 1.3 + 3000))]
    newp = P.table.get(str(gid), {})
    sales = P.sales[P.sales.gid == str(gid)].sort_values("year") if P.sales is not None else pd.DataFrame()
    q = P.quality

    out = os.path.join(HERE, f"otchet_{''.join(c if c.isalnum() else '_' for c in model)}_{year}.pdf")
    with PdfPages(out) as pdf:
        # ===== страница 1
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .955, "Оценка автомобиля по фото", fontsize=20, weight="bold", color="#1f4fb4")
        fig.text(.07, .935, f"Отчёт от {today:%d.%m.%Y}", fontsize=9, color="gray")
        ax = fig.add_axes([.07, .68, .38, .24]); ax.imshow(img); ax.axis("off")
        lines = [f"{model}", "", "Распознано по фото:"] + [f"  {m} — {pr:.0%}" for m, pr in rec["model"]] + [
            "", f"Год выпуска: {year}", f"Пробег: {n(km)} км",
            f"Цвет: {COLORS_RU.get(color, color)}", f"Кузов: {body}"]
        fig.text(.5, .9, lines[0], fontsize=15, weight="bold", va="top")
        fig.text(.5, .865, "\n".join(lines[2:]), fontsize=10, va="top", linespacing=1.6)
        fig.patches.append(plt.Rectangle((.06, .55), .88, .115, transform=fig.transFigure, color="#eef2fb"))
        fig.text(.08, .645, "Оценка рыночной стоимости", fontsize=10)
        fig.text(.08, .6, rub(p * rate), fontsize=30, weight="bold", color="#1f4fb4")
        fig.text(.08, .57, f"Диапазон: {rub(lo * rate)} — {rub(hi * rate)}   ·   £{n(p)}   ·   "
                 f"1 £ = {rate:.2f} ₽ ({rate_src}, {rate_date})", fontsize=8.5)
        if warnings:
            fig.text(.07, .53, "\n".join("⚠ " + w for w in warnings), fontsize=9, color="#b35c00", va="top")
        ax = fig.add_axes([.1, .27, .82, .22])
        ax.plot(years, curve * rate / 1e6, "o-", color="#9aa3b2", label="эта модель при типичном пробеге")
        ax.scatter([year], [p * rate / 1e6], s=90, color="#1f4fb4", zorder=5, label="ваша машина")
        ax.set_ylabel("млн ₽"); ax.set_title("Цена в зависимости от года выпуска", loc="left", fontsize=11)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax = fig.add_axes([.1, .06, .82, .15])
        vals = [p] + future
        ax.bar(["сейчас", "через 1 год", "через 2 года", "через 3 года"], np.array(vals) * rate / 1e6,
               color=["#1f4fb4"] + ["#9aa3b2"] * 3)
        for i, v in enumerate(vals):
            ax.text(i, v * rate / 1e6, rub(v * rate), ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Прогноз при вашем темпе пробега ({n(user_mpy * KM_PER_MILE)} км/год)",
                     loc="left", fontsize=11)
        ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top", "right", "left")]
        pdf.savefig(fig); plt.close(fig)

        # ===== страница 2
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .955, f"Аналитика: {model}", fontsize=16, weight="bold")
        txt = [f"• Каждые 10 000 км пробега меняют цену примерно на {rub(per10k * rate)}.",
               f"• Типичный пробег этой модели — {n(mpy * KM_PER_MILE)} км/год, ваш — "
               f"{n(user_mpy * KM_PER_MILE)} км/год.",
               f"• За год машина подешевеет примерно на {1 - future[0] / p:.0%}."]
        if len(comp) >= 5:
            txt.append(f"• Похожие объявления (тот же возраст и пробег, {len(comp)} шт.): половина цен от "
                       f"{rub(comp.price_today.quantile(.25) * rate)} до {rub(comp.price_today.quantile(.75) * rate)}.")
        if newp:
            np_y = new_price(P.table, gid, year)
            txt.append(f"• Новая в {year} г. стоила £{n(np_y)}; сейчас — {p / np_y:.0%} от этой цены "
                       f"(без поправки на инфляцию).")
        fig.text(.07, .93, "\n".join(txt), fontsize=10, va="top", linespacing=1.8)
        ax = fig.add_axes([.1, .58, .82, .17])
        ax.bar([COLORS_RU.get(c, c) for c in top_colors.index], top_colors.values * 100,
               color=["#1f4fb4" if c == color else "#9aa3b2" for c in top_colors.index])
        for i, (s, e) in enumerate(zip(top_colors.values, color_eff)):
            ax.text(i, s * 100, f"{s:.0%}\n{e:+.1%} к цене", ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0, top_colors.max() * 140); ax.set_ylabel("% объявлений")
        ax.set_title("Цвета этой модели: популярность и влияние на цену", loc="left", fontsize=11)
        if newp:
            ax = fig.add_axes([.1, .36, .36, .15])
            ax.plot(list(newp), list(newp.values()), "o-", ms=3); ax.axvline(year, color="red", ls="--", lw=1)
            ax.set_title("Цена новой, £", loc="left", fontsize=10); ax.grid(alpha=.3)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
        if len(sales):
            ax = fig.add_axes([.56, .36, .36, .15])
            ax.bar(sales.year, sales.sales, color="#9aa3b2"); ax.axvline(year, color="red", ls="--", lw=1)
            ax.set_title("Продажи новых в Великобритании", loc="left", fontsize=10)
        method = (
            "Как получена оценка\n"
            "1. С фото удаляется фон; нейросеть ConvNeXt-Tiny, дообученная на фото DVM-CAR, определяет модель, цвет, кузов.\n"
            "2. Градиентный бустинг считает цену по модели, году, пробегу, цвету, кузову и цене новой машины.\n"
            f"3. Обучено на {n(q['n_ads'])} объявлениях и {n(q['n_images'])} фото (Великобритания, 2018–2021 гг.).\n"
            + "4. Цены приведены к 2026 г. по индексу ONS (подержанные авто) и переведены по курсу ЦБ РФ.\n\n"
            "Точность на тестовой выборке\n"
            f"• ошибка цены (MAPE), модель известна: {q['price_test_mape']:.1%}\n"
            f"• ошибка цены по фото (распознавание + год + пробег): {q['e2e_mape']:.1%}\n"
            f"• точность распознавания модели: {q['recognition']:.0%}\n"
            f"• доля реальных цен внутри диапазона: {q['coverage']:.0%}\n\n"
            "Ограничения: это цены британского рынка в пересчёте на рубли, а не цена в РФ; по фото не видно\n"
            "состояние, ДТП и комплектацию. Отчёт не является оценочным заключением.")
        fig.text(.07, .3, method, fontsize=8.8, va="top", linespacing=1.6)
        pdf.savefig(fig); plt.close(fig)

    print(f"\n{model} ({rec['model'][0][1]:.0%}), {year} г., {n(km)} км")
    print(f"Оценка: {rub(p * rate)}  (диапазон {rub(lo * rate)} — {rub(hi * rate)})")
    for w in warnings:
        print("  ! " + w)
    print(f"PDF: {out}")


if __name__ == "__main__":
    main()
