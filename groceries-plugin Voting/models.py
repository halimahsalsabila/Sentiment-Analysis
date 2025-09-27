from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os
import numpy as np

DATASET_PATH = "dataset_sentiment.csv"

ALLOWED_LABELS = {"positive", "negative", "neutral"}
LABEL_MAP = {
    "positive": "positive", "positif": "positive", "pos": "positive", "good": "positive", "1": "positive", 1: "positive",
    "negative": "negative", "negatif": "negative", "neg": "negative", "bad": "negative", "-1": "negative", -1: "negative",
    "neutral": "neutral", "netral": "neutral", "neu": "neutral", "0": "neutral", 0: "neutral"
}

# --- Fungsi Normalisasi Label ---
def _canon(x):
    key = str(x).strip().lower()
    return LABEL_MAP.get(key, None)

def normalize_validate_df(df: pd.DataFrame) -> pd.DataFrame:
    req = {"to_sentence", "sentiment"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"Kolom wajib hilang: {', '.join(missing)}")
    df = df.copy()
    df["to_sentence"] = df["to_sentence"].astype(str).str.strip()
    df = df[df["to_sentence"].str.len() > 0]
    if df.empty:
        raise ValueError("Semua baris kosong pada kolom 'to_sentence'.")
    df["sentiment_norm"] = df["sentiment"].apply(_canon)
    if df["sentiment_norm"].isna().any():
        bad = df.loc[df["sentiment_norm"].isna(), "sentiment"].astype(str).str.strip().str.lower().value_counts().head(10)
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        raise ValueError(f"Label tidak dikenali: {', '.join(bad.index.tolist())}. "
                         f"Nilai yang diperbolehkan/otomatis dipetakan: {allowed}")
    classes = sorted(df["sentiment_norm"].unique().tolist())
    if len(classes) < 2:
        raise ValueError(f"Dataset harus memiliki minimal 2 kelas label. Ditemukan: {classes}")
    df["sentiment"] = df["sentiment_norm"]
    df = df.drop(columns=["sentiment_norm"])
    return df

def _load_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        return normalize_validate_df(df)
    raise FileNotFoundError("Dataset tidak ditemukan. "
                            "Unggah 'dataset_sentiment.csv' via endpoint /upload atau taruh file di root.")

# --- Load dataset ---
df = _load_dataset()
X = df['to_sentence']
y = df['sentiment']

# --- Fungsi analisis dengan VADER ---
def analyze_sentiment_vader(text_list):
    analyzer = SentimentIntensityAnalyzer()
    predictions = []
    for text in text_list:
        vs = analyzer.polarity_scores(text)
        polarity_score = vs['compound']
        if polarity_score >= 0.05:
            predictions.append('positive')
        elif polarity_score <= -0.05:
            predictions.append('negative')
        else:
            predictions.append('neutral')
    return np.array(predictions)

# --- Fungsi hitung polaritas ensemble ---
def calculate_polarity(probas, class_labels):
    polarity = np.zeros(len(probas))
    if 'positive' in class_labels and 'negative' in class_labels:
        pos_idx = list(class_labels).index('positive')
        neg_idx = list(class_labels).index('negative')
        polarity = probas[:, pos_idx] - probas[:, neg_idx]
    return polarity

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- KNN Models ---
knn_models = [KNeighborsClassifier(n_neighbors=k) for k in range(3, 11)]
knn_pipelines = [make_pipeline(TfidfVectorizer(), model) for model in knn_models]

# --- SVM Models ---
svm_hparams = [(1, 0.01), (100, 0.001), (10, 0.1), (100, 0.1), (1000, 0.001), (10000, 0.1)]
svm_models = [SVC(kernel='rbf', C=c, gamma=g, probability=True) for c, g in svm_hparams]
svm_pipelines = [make_pipeline(TfidfVectorizer(), model) for model in svm_models]

# --- Fit semua model ---
for pipe in knn_pipelines:
    pipe.fit(X_train, y_train)
for pipe in svm_pipelines:
    pipe.fit(X_train, y_train)

# --- Buat ensemble Voting ---
estimators = []
for idx, model in enumerate(knn_pipelines):
    estimators.append((f'knn{3 + idx}', model))
for idx, model in enumerate(svm_pipelines):
    estimators.append((f'svm{idx + 1}', model))

ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X_train, y_train)

# --- Evaluasi ---
cols = {'Y True': y_test}
knn_names = [f'KNN{k}' for k in range(3, 11)]
svm_names = [f'SVM{i+1}' for i in range(len(svm_pipelines))]

# Prediksi KNN & SVM individu
for name, pipe in zip(knn_names, knn_pipelines):
    cols[name] = pipe.predict(X_test)
for name, pipe in zip(svm_names, svm_pipelines):
    cols[name] = pipe.predict(X_test)

# Prediksi ensemble
voting_pred = ensemble.predict(X_test)
cols['Voting'] = voting_pred

# Probabilitas ensemble untuk polaritas
voting_probas = ensemble.predict_proba(X_test)
class_labels = ensemble.classes_
polarity_scores = calculate_polarity(voting_probas, class_labels)

# Prediksi dengan VADER
vader_pred = analyze_sentiment_vader(X_test)
cols['VADER'] = vader_pred

# DataFrame hasil
df_hasil = pd.DataFrame(cols)
df_hasil['voting_correct'] = (df_hasil['Voting'] == df_hasil['Y True'])
df_hasil['Voting_Polarity_Score'] = polarity_scores

print("--- Contoh Hasil DataFrame dengan Nilai Polaritas ---")
print(df_hasil[['Y True', 'Voting', 'Voting_Polarity_Score', 'voting_correct', 'VADER']].head(10))

# --- Evaluasi per model ---
model_columns = [*knn_names, *svm_names, 'Voting', 'VADER']
evaluation_results = {'Model': [], 'Akurasi': [], 'Presisi': [], 'Recall': [], 'F1_Score': []}

for model_name in model_columns:
    y_pred = df_hasil[model_name]
    evaluation_results['Model'].append(model_name)
    evaluation_results['Akurasi'].append(round(accuracy_score(y_test, y_pred) * 100, 2))
    evaluation_results['Presisi'].append(round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))
    evaluation_results['Recall'].append(round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))
    evaluation_results['F1_Score'].append(round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2))

# Simpan hasil evaluasi
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv('hasil_evaluasi_model.csv', index=False)

# Simpan juga hasil prediksi dengan polaritas
df_hasil.to_csv("hasil_prediksi_dengan_polaritas.csv", index=False)

print("\nSaved 'hasil_evaluasi_model.csv' & 'hasil_prediksi_dengan_polaritas.csv'")
print("\nClassification Report for VotingClassifier:")
print(classification_report(y_test, voting_pred))
