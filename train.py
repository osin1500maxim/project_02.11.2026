"""
ОБУЧЕНИЕ: оценка цены автомобиля по фото (датасет DVM-CAR)

Положите в ОДНУ папку с этим файлом:
    Ad_table.csv, Image_table.csv, Price_table.csv, Sales_table.csv   (таблицы DVM-CAR)
    архив с фото DVM-CAR (*.zip, ~13,6 ГБ) — распаковывать не нужно

Запуск:  python train.py
Результат (в той же папке): car_model.pt, price_model.joblib, metrics.txt, training_curves.png

Что делает:
  1. Отбирает до 50 000 фото машин 2013–2020 годов (вид спереди, 1 фото на объявление).
  2. Делит данные на train / val / test (70/15/15) по объявлениям — фото одной машины не попадают в разные части.
  3. Модель цены: градиентный бустинг, гиперпараметры подбираются Optuna по валидации.
  4. Нейросеть: ConvNeXt-Tiny (предобучена на ImageNet) с тремя выходами — модель, цвет, кузов.
     Гиперпараметры подбираются Optuna, обучение с ранней остановкой.
  5. Считает точность на тесте (тест не участвует ни в подборе, ни в обучении).
"""
import glob
import hashlib
import os
import sys
import time
import zipfile

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

# =============================== НАСТРОЙКИ ===============================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # папка с данными = папка с этим файлом
YEAR_MIN, YEAR_MAX = 2013, 2020            # годы выпуска машин на фото
PRICE_YEAR_MIN, PRICE_YEAR_MAX = 2008, 2021  # годы для модели цены (шире: нужны и старые машины)
MAX_IMAGES = 50_000                        # максимум фото
MIN_PER_MODEL, MAX_PER_MODEL = 150, 600    # фото на одну модель авто
VIEWPOINT = 0                              # 0 = вид спереди (колонка Predicted_viewpoint)

TUNE = True                                # подбирать гиперпараметры (Optuna)
PRICE_TRIALS = 30                          # попыток для модели цены
CNN_TRIALS = 8                             # попыток для нейросети (каждая — 3 эпохи на 30 % фото)

# Гиперпараметры по умолчанию (стартовая точка подбора)
PRICE_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
                    max_features=0.8)
CNN_PARAMS = dict(lr=1e-4, lr_head=1e-3, weight_decay=0.05, label_smoothing=0.1, dropout=0.2)
EPOCHS, PATIENCE, BATCH, IMG_SIZE = 15, 3, 96, 224   # для RTX 4070 (12 ГБ)
NUM_WORKERS = 4                            # если зависает на Windows — поставьте 0

# Индекс цен на подержанные авто в Великобритании (ONS, ряд D7E9, 2015=100) — перевод цен в сегодняшние
UK_INDEX = {2015: 100.0, 2016: 95.0, 2017: 93.6, 2018: 95.5, 2019: 93.3, 2020: 96.6, 2021: 108.8,
            2022: 121.5, 2023: 118.9, 2024: 111.2, 2025: 112.0, 2026: 109.9}
SEED = 42
QUICK = "--quick" in sys.argv              # быстрый режим для проверки кода
if QUICK:
    MIN_PER_MODEL, PRICE_TRIALS, CNN_TRIALS, EPOCHS, IMG_SIZE, BATCH, NUM_WORKERS = 30, 3, 2, 3, 96, 32, 0
PRETRAINED = "--no-pretrained" not in sys.argv
# ========================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.manual_seed(SEED)
np.random.seed(SEED)


def path(name):
    return os.path.join(DATA_DIR, name)


# ------------------------------------------------------------------ 1. ДАННЫЕ
def find_csv(word):
    files = [f for f in glob.glob(path("**/*.csv"), recursive=True) if word in os.path.basename(f).lower()]
    if not files:  # таблицы могли остаться в небольшом архиве — достаём
        for z in glob.glob(path("*.zip")):
            if os.path.getsize(z) < 500e6:
                zipfile.ZipFile(z).extractall(DATA_DIR)
        files = [f for f in glob.glob(path("**/*.csv"), recursive=True) if word in os.path.basename(f).lower()]
    files.sort(key=lambda f: ("extra" in f.lower(), len(f)))
    return files[0] if files else None


def col(df, *names):
    """Находит колонку по одному из возможных названий (без учёта регистра)."""
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return df[low[n.lower()]]
    raise KeyError(f"Нет колонки {names}. Колонки файла: {list(df.columns)}")


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")


def load_ads():
    raw = pd.read_csv(find_csv("ad_table"), low_memory=False)
    ads = pd.DataFrame({
        "maker": col(raw, "Maker").astype(str).str.strip(),
        "model": (col(raw, "Maker").astype(str).str.strip() + " " + col(raw, "Genmodel").astype(str).str.strip()),
        "genmodel_id": col(raw, "Genmodel_ID").astype(str),
        "adv_id": col(raw, "Adv_ID").astype(str),
        "adv_year": num(col(raw, "Adv_year")),
        "reg_year": num(col(raw, "Reg_year")),
        "mileage": num(col(raw, "Runned_Miles", "Mileage")),
        "price": num(col(raw, "Price")),
        "color": col(raw, "Color", "Colour").astype(str),
        "body": col(raw, "Bodytype", "Body_type").astype(str),
    })
    ads = ads.dropna(subset=["adv_year", "reg_year", "mileage", "price"])
    ads = ads[(ads.price.between(500, 250_000)) & (ads.mileage.between(0, 250_000)) &
              (ads.reg_year.between(PRICE_YEAR_MIN, PRICE_YEAR_MAX)) & (ads.adv_year >= ads.reg_year)]
    ads = ads.drop_duplicates("adv_id")
    ads = ads[ads.groupby("model")["model"].transform("size") >= 50]           # редкие модели убираем
    q = ads.groupby(["model", "reg_year"])["price"]                           # выбросы цены (опечатки)
    ads = ads[(ads.price >= q.transform(lambda s: s.quantile(.01))) & (ads.price <= q.transform(lambda s: s.quantile(.99)))]
    # цена в сегодняшних фунтах
    ads["price_today"] = ads.price * ads.adv_year.clip(2015, 2026).astype(int).map(lambda y: UK_INDEX[2026] / UK_INDEX[y])
    ads["age"] = (ads.adv_year - ads.reg_year + 0.5).clip(lower=0.1)
    ads["split"] = ads.adv_id.map(split_of)
    return ads.reset_index(drop=True)


def split_of(adv_id):
    h = int(hashlib.md5(adv_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


IMG_EXT = (".jpg", ".jpeg", ".png")


def scan_photos():
    """Все фото DVM-CAR, которые есть на диске: в архивах *.zip (без распаковки) и в распакованных папках.
    Возвращает {имя_файла: (архив или None, путь)}."""
    where = {}
    zips = sorted(glob.glob(path("**/*.zip"), recursive=True), key=os.path.getsize, reverse=True)
    print(f"  архивы: {[os.path.basename(z) + f' ({os.path.getsize(z) / 1e9:.1f} ГБ)' for z in zips] or 'не найдены'}")
    for z in zips:
        try:
            members = zipfile.ZipFile(z).namelist()
        except zipfile.BadZipFile:
            print(f"  [!] архив {os.path.basename(z)} повреждён или не докачан — скачайте заново")
            continue
        n0 = len(where)
        for m in members:
            if m.lower().endswith(IMG_EXT):
                where.setdefault(os.path.basename(m), (z, m))
        print(f"    {os.path.basename(z)}: фото внутри — {len(where) - n0}")
    n0 = len(where)
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(IMG_EXT) and "$$" in f:
                where.setdefault(f, (None, os.path.join(root, f)))
    if len(where) > n0:
        print(f"  в распакованных папках: {len(where) - n0} фото")
    print(f"  всего фото найдено: {len(where)}")
    if not where:
        print("\n[ОШИБКА] Не найдено ни одного фото. Положите архив с фото DVM-CAR (.zip) в папку с train.py:")
        print("  https://figshare.com/ndownloader/files/34792480  (фото спереди, 730 МБ)")
        print(f"  Папка, где ищутся фото: {DATA_DIR}")
        sys.exit(1)
    return where


def load_images(ads, where):
    """Таблица фото по именам файлов: Марка$$Модель$$Год$$Цвет$$GenmodelID$$N$$image_k.jpg"""
    rows = []
    for name in where:
        p = name.split("$$")
        if len(p) >= 7:
            rows.append((name, f"{p[0].strip()} {p[1].strip()}", p[2], p[3], f"{p[4]}$${p[5]}"))
    if not rows:
        print("\n[ОШИБКА] Имена фото не в формате DVM-CAR. Пример:", list(where)[:3])
        sys.exit(1)
    df = pd.DataFrame(rows, columns=["name", "model_from_name", "year", "color_from_name", "adv_id"])
    df["year"] = pd.to_numeric(df.year, errors="coerce")
    # ракурс из Image_table (если файл есть); фото, которых нет в таблице, оставляем
    f = find_csv("image_table")
    if f:
        it = pd.read_csv(f, low_memory=False)
        views = dict(zip(col(it, "Image_name").astype(str).map(os.path.basename), num(col(it, "Predicted_viewpoint"))))
        df["view"] = df.name.map(views)
        df = df[df.view.isna() | (df.view == VIEWPOINT)]
    df = df[df.year.between(YEAR_MIN, YEAR_MAX)]
    print(f"  фото {YEAR_MIN}–{YEAR_MAX} годов нужного ракурса: {len(df)}")
    # метки берём из объявления (если нашлось), иначе из имени файла
    df = df.merge(ads[["adv_id", "model", "color", "body", "split"]], on="adv_id", how="left")
    print(f"  из них сопоставлено с объявлениями: {df.model.notna().sum()}")
    df["model"] = df.model.fillna(df.model_from_name)
    df["color"] = df.color.fillna(df.color_from_name)
    df["body"] = df.body.fillna(df.model.map(ads.groupby("model").body.agg(lambda s: s.mode()[0])))
    df["split"] = df.split.fillna(df.adv_id.map(split_of))
    df = df[df.model.isin(set(ads.model)) & df.body.notna()]     # только модели, для которых есть цены
    df = df.sample(frac=1, random_state=SEED).drop_duplicates("adv_id")       # 1 фото на объявление
    return df.reset_index(drop=True)


def limit_images(df):
    """Модели с достаточным числом фото, не больше MAX_PER_MODEL на модель и MAX_IMAGES всего."""
    df = df[df.groupby("model")["model"].transform("size") >= MIN_PER_MODEL]
    df = df.groupby("model").head(MAX_PER_MODEL)
    if len(df) > MAX_IMAGES:
        df = df.groupby("model", group_keys=False).apply(
            lambda g: g.head(max(1, int(len(g) * MAX_IMAGES / len(df)))))
    return df.reset_index(drop=True)


class PhotoStore:
    """Читает фото из архива (без распаковки) или из папки."""

    def __init__(self, where):
        self.where, self._open = where, {}

    def __call__(self, name):
        z, m = self.where[name]
        if z is None:
            return to_rgb(Image.open(m))
        key = (os.getpid(), z)
        if key not in self._open:
            self._open[key] = zipfile.ZipFile(z)
        with self._open[key].open(m) as f:
            return to_rgb(Image.open(f))

    def __getstate__(self):
        return {"where": self.where, "_open": {}}


def to_rgb(img):
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def load_params(key):
    import json
    f = path("best_params.json")
    return json.load(open(f, encoding="utf-8")).get(key) if os.path.exists(f) and not QUICK else None


def save_params(key, value):
    import json
    f = path("best_params.json")
    d = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    d[key] = value
    json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ 2. МОДЕЛЬ ЦЕНЫ
CATS = ["maker", "body", "color"]


def new_price_table():
    f = find_csv("price_table")
    if not f:
        return {}
    t = pd.read_csv(f)
    t = pd.DataFrame({"gid": col(t, "Genmodel_ID").astype(str), "year": num(col(t, "Year")),
                      "price": num(col(t, "Entry_price"))}).dropna()
    return {g: dict(zip(d.year.astype(int), d.price)) for g, d in t.groupby("gid")}


def new_price(table, gid, year):
    d = table.get(str(gid))
    if not d:
        return np.nan
    y = min(d, key=lambda k: abs(k - year))
    return d[y]


def price_features(df, table):
    X = df[["model"] + CATS].copy()
    X["age"] = df["age"].values
    X["mileage"] = df["mileage"].values / 1000
    X["miles_per_year"] = df["mileage"].values / df["age"].values
    X["log_new_price"] = np.log([new_price(table, g, y) for g, y in zip(df.genmodel_id, df.reg_year)])
    return X


class PriceModel:
    def __init__(self, params):
        self.params = params
        self.te = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=SEED))
        self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)

    def _matrix(self, X, te):
        M = X.drop(columns=["model"]).copy()
        M[CATS] = self.oe.transform(X[CATS].astype(str))
        M.insert(0, "model_enc", te)
        return M

    def fit(self, X, y):
        te = self.te.fit_transform(X[["model"]], y).ravel()       # без утечки (перекрёстное обучение)
        self.oe.fit(X[CATS].astype(str))
        p = dict(self.params)
        self.gbm = HistGradientBoostingRegressor(
            max_iter=3000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
            categorical_features=[False, True, True, True] + [False] * 4, random_state=SEED,
            learning_rate=p.pop("learning_rate"), **p)
        self.gbm.fit(self._matrix(X, te), y)
        return self

    def predict_log(self, X):
        return self.gbm.predict(self._matrix(X, self.te.transform(X[["model"]]).ravel()))

    def calibrate(self, X, y):
        """Интервал P10–P90 по ошибкам на валидации (конформный метод), отдельно для 5 ценовых групп."""
        p = self.predict_log(X)
        r = y - p
        self.edges = np.quantile(p, [.2, .4, .6, .8])
        g = np.digitize(p, self.edges)
        self.q = [(np.quantile(r[g == i], .1), np.quantile(r[g == i], .9)) for i in range(5)]

    def predict(self, X):
        p = self.predict_log(X)
        g = np.digitize(p, self.edges)
        lo = np.array([self.q[i][0] for i in g])
        hi = np.array([self.q[i][1] for i in g])
        return np.exp(p), np.exp(p + lo), np.exp(p + hi)


def mape(y, p):
    return float(np.mean(np.abs(p - y) / y))


def train_price(ads, table):
    X, y = price_features(ads, table), np.log(ads.price_today.values)
    tr, va, te = [(ads.split == s).values for s in ("train", "val", "test")]
    params = dict(PRICE_PARAMS)
    saved = load_params("price")
    if saved:
        print(f"  подобранные ранее гиперпараметры взяты из best_params.json (удалите файл, чтобы подобрать заново)")
        params = saved
    elif TUNE:
        def objective(trial):
            p = dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                     max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
                     min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 400, log=True),
                     l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 30, log=True),
                     max_features=trial.suggest_float("max_features", 0.4, 1.0))
            m = PriceModel(p).fit(X[tr], y[tr])
            score = mape(np.exp(y[va]), np.exp(m.predict_log(X[va])))
            print(f"  попытка {trial.number + 1}/{PRICE_TRIALS}: MAPE val = {score:.4f}")
            return score
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED))
        study.enqueue_trial(PRICE_PARAMS)
        study.optimize(objective, n_trials=PRICE_TRIALS)
        params = study.best_params
        save_params("price", params)
    print(f"  гиперпараметры: {params}")
    model = PriceModel(params).fit(X[tr], y[tr])
    model.calibrate(X[va], y[va])
    res = {}
    for name, m in (("train", tr), ("val", va), ("test", te)):
        p, lo, hi = model.predict(X[m])
        yt = np.exp(y[m])
        res[name] = {"MAPE": mape(yt, p), "в пределах ±10%": float(np.mean(np.abs(p - yt) / yt <= .1)),
                     "попадание в интервал": float(np.mean((yt >= lo) & (yt <= hi)))}
    for k, v in res.items():
        print(f"  {k:<5}: MAPE {v['MAPE']:.1%}, в пределах ±10%: {v['в пределах ±10%']:.0%}, "
              f"в интервале: {v['попадание в интервал']:.0%}")
    return model, params, res


# ------------------------------------------------------------------ 3. НЕЙРОСЕТЬ
class CarNet(nn.Module):
    """ConvNeXt-Tiny + 3 выхода: модель авто, цвет, кузов."""

    def __init__(self, n_model, n_color, n_body, dropout=0.2, pretrained=True):
        super().__init__()
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(m.features, m.avgpool, m.classifier[0], m.classifier[1])
        self.drop = nn.Dropout(dropout)
        self.model_head, self.color_head, self.body_head = (nn.Linear(768, n) for n in (n_model, n_color, n_body))

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.model_head(f), self.color_head(f), self.body_head(f)


NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def tf(train):
    if train:   # оттенок не меняем: цвет машины — это метка
        return T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), NORM])
    return T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), NORM])


class Photos(Dataset):
    def __init__(self, df, store, train):
        self.df, self.store, self.tf = df.reset_index(drop=True), store, tf(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(self.store(r["name"])), int(r.y_model), int(r.y_color), int(r.y_body)


def run_epochs(df_tr, df_va, store, n, params, epochs, trial=None):
    loader = lambda d, t: DataLoader(Photos(d, store, t), batch_size=BATCH, shuffle=t, num_workers=NUM_WORKERS,  # noqa
                                     pin_memory=DEVICE == "cuda", persistent_workers=NUM_WORKERS > 0)
    dl_tr, dl_va = loader(df_tr, True), loader(df_va, False)
    net = CarNet(*n, params["dropout"], PRETRAINED).to(DEVICE)
    heads = [p for name, p in net.named_parameters() if "head" in name]
    body = [p for name, p in net.named_parameters() if "head" not in name]
    opt = torch.optim.AdamW([{"params": body, "lr": params["lr"]}, {"params": heads, "lr": params["lr_head"]}],
                            weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[params["lr"], params["lr_head"]],
                                                total_steps=epochs * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, epochs + 1):
        t0, total = time.time(), 0.0
        net.train()
        for x, ym, yc, yb in dl_tr:
            x, ym, yc, yb = x.to(DEVICE), ym.to(DEVICE), yc.to(DEVICE), yb.to(DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                om, oc, ob = net(x)
                ls = params["label_smoothing"]
                loss = (F.cross_entropy(om, ym, label_smoothing=ls) + 0.5 * F.cross_entropy(oc, yc, label_smoothing=ls)
                        + 0.5 * F.cross_entropy(ob, yb, label_smoothing=ls))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(x)
        acc = accuracy(net, dl_va)
        hist.append({"epoch": ep, "loss": total / len(df_tr), **acc})
        print(f"    эпоха {ep}/{epochs}: потери {total / len(df_tr):.3f} | точность val: модель {acc['model']:.3f}, "
              f"цвет {acc['color']:.3f}, кузов {acc['body']:.3f} | {time.time() - t0:.0f} c")
        if acc["model"] > best:
            best, bad = acc["model"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if trial is not None:
            trial.report(acc["model"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad >= PATIENCE:
            print("    ранняя остановка")
            break
    net.load_state_dict(best_state)
    return net, best, hist


@torch.no_grad()
def accuracy(net, loader):
    net.eval()
    hit, n = np.zeros(3), 0
    for x, ym, yc, yb in loader:
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            outs = net(x.to(DEVICE))
        for i, (o, y) in enumerate(zip(outs, (ym, yc, yb))):
            hit[i] += (o.argmax(1).cpu() == y).sum().item()
        n += len(x)
    return dict(zip(("model", "color", "body"), hit / max(n, 1)))


@torch.no_grad()
def predict_all(net, df, store):
    net.eval()
    dl = DataLoader(Photos(df, store, False), batch_size=BATCH, num_workers=NUM_WORKERS)
    out = [[], [], []]
    for x, *_ in dl:
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            for i, o in enumerate(net(x.to(DEVICE))):
                out[i].append(o.float().argmax(1).cpu().numpy())
    return [np.concatenate(o) for o in out]


def train_cnn(imgs, store):
    classes = {k: sorted(imgs[k].unique()) for k in ("model", "color", "body")}
    for k in classes:
        imgs["y_" + k] = imgs[k].map({c: i for i, c in enumerate(classes[k])})
    n = [len(classes[k]) for k in ("model", "color", "body")]
    tr, va = imgs[imgs.split == "train"], imgs[imgs.split == "val"]
    params = dict(CNN_PARAMS)
    saved = load_params("cnn")
    if saved:
        print(f"  подобранные ранее гиперпараметры взяты из best_params.json")
        params = saved
    elif TUNE:
        def objective(trial):
            p = dict(lr=trial.suggest_float("lr", 3e-5, 5e-4, log=True),
                     lr_head=trial.suggest_float("lr_head", 3e-4, 5e-3, log=True),
                     weight_decay=trial.suggest_float("weight_decay", 0.01, 0.2, log=True),
                     label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
                     dropout=trial.suggest_float("dropout", 0.0, 0.4))
            print(f"  попытка {trial.number + 1}/{CNN_TRIALS}: " + ", ".join(f"{k}={v:.3g}" for k, v in p.items()))
            _, best, _ = run_epochs(tr.sample(frac=0.3, random_state=trial.number), va.sample(frac=0.5, random_state=0),
                                    store, n, p, 3, trial)
            return best
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
        study.enqueue_trial(CNN_PARAMS)
        study.optimize(objective, n_trials=CNN_TRIALS)
        params = study.best_params
        save_params("cnn", params)
    print(f"  финальное обучение, гиперпараметры: {params}")
    net, _, hist = run_epochs(tr, va, store, n, params, EPOCHS)
    return net, classes, params, hist


# ------------------------------------------------------------------ 4. ЗАПУСК
def main():
    t0 = time.time()
    print(f"Устройство: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    if DEVICE == "cpu" and not QUICK and "--cpu" not in sys.argv:
        print(f"\n[ОШИБКА] PyTorch не видит видеокарту (установлен torch {torch.__version__}).")
        if "+cu" not in torch.__version__:
            print("  Установлена версия для процессора. Нужна версия с CUDA (в названии +cu126 / +cu128).")
        else:
            print("  Версия с CUDA установлена, но видеокарта не найдена: обновите драйвер NVIDIA и перезагрузите ПК.")
        print("  Обучение нейросети на процессоре займёт несколько суток.")
        print("  Если всё равно хотите на процессоре: python train.py --cpu")
        sys.exit(1)
    print("\n1) Данные")
    ads = load_ads()
    print(f"  объявлений: {len(ads)}, моделей: {ads.model.nunique()}")
    where = scan_photos()
    imgs = limit_images(load_images(ads, where))
    store = PhotoStore({n: where[n] for n in imgs.name})
    if len(imgs) == 0:
        sys.exit(f"[ОШИБКА] Фото найдены, но ни у одной модели нет {MIN_PER_MODEL}+ фото. "
                 f"Уменьшите MIN_PER_MODEL в начале train.py.")
    print(f"  отобрано фото: {len(imgs)} ({imgs.model.nunique()} моделей), train/val/test: "
          f"{(imgs.split == 'train').sum()}/{(imgs.split == 'val').sum()}/{(imgs.split == 'test').sum()}")

    print("\n2) Модель цены")
    table = new_price_table()
    price_model, price_params, price_res = train_price(ads, table)

    print("\n3) Нейросеть распознавания")
    net, classes, cnn_params, hist = train_cnn(imgs, store)

    print("\n4) Проверка на тесте")
    test = imgs[imgs.split == "test"].reset_index(drop=True)
    pm, pc, pb = predict_all(net, test, store)
    rec = {"модель": float(np.mean(pm == test.y_model)), "цвет": float(np.mean(pc == test.y_color)),
           "кузов": float(np.mean(pb == test.y_body))}
    trn = imgs[imgs.split == "train"].sample(min(3000, (imgs.split == "train").sum()), random_state=0)
    rec_train = float(np.mean(predict_all(net, trn, store)[0] == trn.y_model.values))
    print(f"  точность распознавания на тесте: " + ", ".join(f"{k} {v:.1%}" for k, v in rec.items())
          + f" (модель на train: {rec_train:.1%})")
    # сквозная проверка: модель/цвет/кузов из фото + настоящий год и пробег
    test["model"] = np.array(classes["model"])[pm]
    test["color"] = np.array(classes["color"])[pc]
    test["body"] = np.array(classes["body"])[pb]
    t = test[["adv_id", "model", "color", "body"]].merge(
        ads.drop(columns=["model", "color", "body", "split"]), on="adv_id")   # только фото с известной ценой
    t["genmodel_id"] = t.model.map(ads.groupby("model").genmodel_id.agg(lambda s: s.mode()[0]))
    t["maker"] = t.model.map(ads.groupby("model").maker.first())
    e2e = mape(t.price_today.values, price_model.predict(price_features(t, table))[0])
    print(f"  ошибка цены по фото (распознавание + год + пробег): MAPE {e2e:.1%}")

    # ---- сохранение
    torch.save({"weights": net.state_dict(), "classes": classes, "img_size": IMG_SIZE, "params": cnn_params},
               path("car_model.pt"))
    market = ads[["model", "maker", "genmodel_id", "reg_year", "adv_year", "age", "mileage", "price_today", "color",
                  "body"]]
    sales_f = find_csv("sales_table")
    sales = None
    if sales_f:
        s = pd.read_csv(sales_f)
        years = [c for c in s.columns if str(c).strip().isdigit()]
        sales = s.assign(gid=col(s, "Genmodel_ID").astype(str)).melt("gid", years, "year", "sales")
        sales["year"] = sales.year.astype(int)
    joblib.dump({"model": price_model, "new_prices": table, "market": market, "sales": sales,
                 "quality": {"price_test_mape": price_res["test"]["MAPE"], "e2e_mape": e2e,
                             "recognition": rec["модель"], "coverage": price_res["test"]["попадание в интервал"],
                             "n_ads": len(ads), "n_images": len(imgs)}}, path("price_model.joblib"))
    gap = price_res["test"]["MAPE"] - price_res["train"]["MAPE"]
    report = [f"Обучено {time.strftime('%Y-%m-%d %H:%M')} за {(time.time() - t0) / 60:.0f} мин",
              f"Объявлений: {len(ads)}, фото: {len(imgs)}, моделей на фото: {len(classes['model'])}",
              "", "МОДЕЛЬ ЦЕНЫ", f"гиперпараметры: {price_params}"]
    report += [f"  {k}: " + ", ".join(f"{a} {b:.1%}" for a, b in v.items()) for k, v in price_res.items()]
    report += [f"  разрыв MAPE train→test: {gap * 100:.1f} п.п. → "
               + ("переобучения нет" if gap < 0.03 else "небольшой" if gap < 0.06 else "есть переобучение"),
               "", "НЕЙРОСЕТЬ", f"гиперпараметры: {cnn_params}",
               "  точность на тесте: " + ", ".join(f"{k} {v:.1%}" for k, v in rec.items()),
               f"  точность модели на train: {rec_train:.1%} (разрыв с тестом {(rec_train - rec['модель']) * 100:.1f} п.п.)",
               "", f"ЦЕНА ПО ФОТО (тест): MAPE {e2e:.1%}"]
    open(path("metrics.txt"), "w", encoding="utf-8").write("\n".join(report))
    plot_curves(hist, price_res)
    print(f"\nГотово за {(time.time() - t0) / 60:.0f} мин. Файлы: car_model.pt, price_model.joblib, metrics.txt, "
          f"training_curves.png")


def plot_curves(hist, price_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h["epoch"] for h in hist]
    a.plot(ep, [h["loss"] for h in hist], "o-", label="потери (train)")
    a2 = a.twinx()
    a2.plot(ep, [h["model"] for h in hist], "s-", color="tab:green", label="точность модели (val)")
    a.set_xlabel("эпоха"); a.set_title("Обучение нейросети"); a.legend(loc="upper left"); a2.legend(loc="lower right")
    names = list(price_res)
    b.bar(names, [price_res[k]["MAPE"] * 100 for k in names], color=["#9aa3b2", "#9aa3b2", "#1f4fb4"])
    b.set_ylabel("MAPE, %"); b.set_title("Ошибка модели цены: train / val / test")
    fig.tight_layout()
    fig.savefig(path("training_curves.png"), dpi=130)


if __name__ == "__main__":
    main()
