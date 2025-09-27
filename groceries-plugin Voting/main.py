from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import json, os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Canonical dataset file ===
DATASET_PATH = "dataset_sentiment.csv"

# Allowed labels & normalization mapping
ALLOWED_LABELS = {"positive", "negative", "neutral"}
LABEL_MAP = {
    "positive": "positive", "positif": "positive", "pos": "positive", "good": "positive", "1": "positive", 1: "positive",
    "negative": "negative", "negatif": "negative", "neg": "negative", "bad": "negative", "-1": "negative", -1: "negative",
    "neutral": "neutral", "netral": "neutral", "neu": "neutral", "0": "neutral", 0: "neutral"
}

def _canon(x):
    key = str(x).strip().lower()
    return LABEL_MAP.get(key, None)

def normalize_validate_df(df: pd.DataFrame) -> pd.DataFrame:
    # Check required columns
    req = {"to_sentence", "sentiment"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise HTTPException(status_code=400, detail=f"Kolom wajib hilang: {', '.join(missing)}")
    
    # Drop rows with missing/empty text
    df = df.copy()
    df["to_sentence"] = df["to_sentence"].astype(str).str.strip()
    df = df[df["to_sentence"].str.len() > 0]
    if df.empty:
        raise HTTPException(status_code=400, detail="Semua baris kosong pada kolom 'to_sentence'.")
    
    # Normalize labels
    df["sentiment_norm"] = df["sentiment"].apply(_canon)
    invalid_mask = df["sentiment_norm"].isna()
    if invalid_mask.any():
        bad = df.loc[invalid_mask, "sentiment"].astype(str).str.strip().str.lower().value_counts().head(10)
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        raise HTTPException(
            status_code=400,
            detail=f"Label tidak dikenali: {', '.join(bad.index.tolist())}. "
                   f"Nilai yang diperbolehkan/otomatis dipetakan: {allowed} "
                   "(+ sinonim umum seperti positif/negatif/netral, 1/0/-1, good/bad)."
        )
    
    # Ensure at least 2 classes
    classes = sorted(df["sentiment_norm"].unique().tolist())
    if len(classes) < 2:
        raise HTTPException(status_code=400, detail=f"Dataset harus memiliki minimal 2 kelas label. Ditemukan: {classes}")
    
    # Use normalized
    df["sentiment"] = df["sentiment_norm"]
    df = df.drop(columns=["sentiment_norm"])
    return df

# Globals populated at startup / retrain
tfidf = None
ensemble = None
X_test_tfidf = None
y_test_sentiment = None
model_results = []

def load_dataset_or_none():
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            df = normalize_validate_df(df)
            return df
        except HTTPException as e:
            raise
        except Exception as e:
            print(f"[dataset] Gagal baca dataset: {e}")
            raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {e}")
    return None

def train_ensemble_from_df(df: pd.DataFrame):
    global tfidf, ensemble, X_test_tfidf, y_test_sentiment
    X_sentiment = df['to_sentence']
    y_sentiment = df['sentiment']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_sentiment, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
    )

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train_s)
    X_test_tfidf_local = tfidf.transform(X_test_s)

    estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=3)),
        ('svm', SVC(kernel='rbf', C=1, gamma=0.01, probability=True))
    ]
    ens = VotingClassifier(estimators, voting="soft")  # pakai soft voting untuk proba
    ens.fit(X_train_tfidf, y_train_s)

    ensemble = ens
    X_test_tfidf = X_test_tfidf_local
    y_test_sentiment = y_test_s

def compute_and_write_eval_csv():
    global ensemble, X_test_tfidf, y_test_sentiment, model_results
    path = "hasil_evaluasi_model.csv"
    if ensemble is None or X_test_tfidf is None:
        model_results = []
        return
    y_pred = ensemble.predict(X_test_tfidf)
    acc = accuracy_score(y_test_sentiment, y_pred) * 100
    prec = precision_score(y_test_sentiment, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y_test_sentiment, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test_sentiment, y_pred, average='weighted', zero_division=0) * 100
    eval_df = pd.DataFrame([{
        "Model": "Voting",
        "Akurasi": round(acc,2),
        "Presisi": round(prec,2),
        "Recall": round(rec,2),
        "F1_Score": round(f1,2),
    }])
    eval_df.to_csv(path, index=False)
    model_results = eval_df.to_dict(orient='records')

# 🔹 fungsi tambahan: prediksi + polaritas
def predict_with_polarity(text: str):
    global ensemble, tfidf
    X_t = tfidf.transform([text])
    probas = ensemble.predict_proba(X_t)
    class_labels = ensemble.classes_
    pred = ensemble.predict(X_t)[0]

    # hitung polaritas
    polarity = 0.0
    if "positive" in class_labels and "negative" in class_labels:
        pos_idx = list(class_labels).index("positive")
        neg_idx = list(class_labels).index("negative")
        polarity = probas[0, pos_idx] - probas[0, neg_idx]

    return str(pred), float(polarity)

# === Startup ===
@app.on_event("startup")
def startup():
    try:
        df = load_dataset_or_none()
        if df is not None:
            train_ensemble_from_df(df)
            compute_and_write_eval_csv()
        else:
            print("[startup] Dataset belum tersedia. Upload via /upload untuk mulai melatih model.")
    except HTTPException as e:
        print(f"[startup] Dataset tidak valid: {e.detail}")

# === ROUTES ===
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/upload")
async def upload_page(request: Request, message: str = None):
    return templates.TemplateResponse("upload.html", {"request": request, "message": message})

@app.post("/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus .csv")
    try:
        contents = await file.read()
        with open(DATASET_PATH, "wb") as f:
            f.write(contents)
        df = load_dataset_or_none()
        train_ensemble_from_df(df)
        compute_and_write_eval_csv()
        msg = f"Dataset diunggah ({file.filename}) dan model berhasil dilatih ulang."
        return templates.TemplateResponse("upload.html", {"request": request, "message": msg})
    except HTTPException as e:
        return templates.TemplateResponse("upload.html", {"request": request, "message": f"Gagal: {e.detail}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal unggah/latih: {e}")

@app.post("/predict")
async def predict_sentiment(request: Request, text: str = Form(...)):
    try:
        if ensemble is not None and tfidf is not None:
            sentiment, polarity = predict_with_polarity(text)
        else:
            sentiment = "negative" if "bad" in text.lower() else "positive"
            polarity = 0.0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {e}")
    return templates.TemplateResponse("result.html", {
        "request": request,
        "sentiment": sentiment,
        "input_text": text,
        "polarity": polarity
    })

@app.get("/dashboard")
async def dashboard(request: Request):
    global model_results
    path = "hasil_evaluasi_model.csv"
    if os.path.exists(path):
        try:
            df_eval = pd.read_csv(path)
            model_results = df_eval.to_dict(orient='records')
        except Exception:
            pass
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "model_results": model_results,
        "model_results_json": json.dumps(model_results, ensure_ascii=False)
    })

@app.get("/evaluate")
async def evaluate(request: Request):
    if ensemble is None or X_test_tfidf is None:
        evaluation_results = { "Akurasi": "N/A", "Presisi": "N/A", "Recall": "N/A", "F1_Score": "N/A" }
    else:
        y_pred = ensemble.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test_sentiment, y_pred)
        precision = precision_score(y_test_sentiment, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_sentiment, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_sentiment, y_pred, average='weighted', zero_division=0)
        evaluation_results = {
            "Akurasi": f"{accuracy * 100:.2f}%",
            "Presisi": f"{precision * 100:.2f}%",
            "Recall": f"{recall * 100:.2f}%",
            "F1_Score": f"{f1 * 100:.2f}%"
        }
    return templates.TemplateResponse("evaluation.html", {"request": request, "evaluation_results": evaluation_results})
