import pandas as pd
import os

# Folder containing  CSV result files
results_folder = "../1. Raw Data"  # change if needed

# Columns to remove
cols_to_remove = [
    "thisRow.t",
    "notes",
    "resp_key",
    "missing_file",
    "participant",
    "session"
]

# Summary rows (one per participant/file)
summary_rows = []

def _lower(s):
    return str(s).strip().lower()

def pick_column(df, candidates):
    """Pick the first matching column from candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def infer_condition_columns(df):
    """
    Try to infer (database_col, emotion_col) by scanning object/category columns
    for values containing 'ai'/'non-ai' (or 'nonai') and 'happy'/'sad'.
    """
    obj_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    database_col = None
    emotion_col = None

    for c in obj_cols:
        vals = df[c].dropna().astype(str).map(_lower).unique()
        if database_col is None and any("ai" in v for v in vals) and any(("non-ai" in v) or ("nonai" in v) for v in vals):
            database_col = c
        if emotion_col is None and any("happy" in v for v in vals) and any("sad" in v for v in vals):
            emotion_col = c

    return database_col, emotion_col

def coerce_correct_series(s):
    """Coerce a correctness series to boolean where possible."""
    if s is None:
        return None
    if s.dtype == bool:
        return s
    # numeric 0/1
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any() and set(s_num.dropna().unique()).issubset({0, 1}):
        return s_num.astype(int).astype(bool)
    # strings like 'true'/'false'
    s_str = s.astype(str).map(_lower)
    if set(s_str.dropna().unique()).issubset({"true", "false"}):
        return s_str == "true"
    return None

for filename in os.listdir(results_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(results_folder, filename)

        df = pd.read_csv(file_path)

        # Drop specified columns if they exist
        df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])

        # Drop any unnamed columns (e.g., index columns like "Unnamed: 13")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        # Identify and remove RT outliers using IQR method
        if "rt" in df.columns:
            # Make sure rt is numeric; non-numeric values become NaN
            df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

            # Initialize outlier column (overwrite if exists)
            df["rt_outlier"] = False

            # Mark missing RTs as outliers
            missing_rt = df["rt"].isna()

            # Compute IQR bounds only on valid RTs
            valid_rt = df.loc[~missing_rt, "rt"]

            if len(valid_rt) >= 4:
                Q1 = valid_rt.quantile(0.25)
                Q3 = valid_rt.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                iqr_outliers = (df["rt"] < lower_bound) | (df["rt"] > upper_bound)
            else:
                iqr_outliers = False

            # Combine missing + IQR outliers
            df["rt_outlier"] = missing_rt | iqr_outliers

            # Report percentage of outliers
            outlier_pct = 100 * df["rt_outlier"].mean()
            print(f"{filename}: {outlier_pct:.2f}% RT outliers")

            # Skip file if too many outliers
            if outlier_pct > 10:
                print(f"Skipping {filename} (more than 10% outliers)")
                continue

            # Remove outliers
            df = df[~df["rt_outlier"]]
            # Drop the helper outlier column before saving
            df = df.drop(columns=["rt_outlier"])

        # ---- Build per-file summary row ----
        row = {
            "database_name": os.path.splitext(filename)[0],
            "ai_sad_avg_rt": float("nan"),
            "ai_sad_accuracy_pct": float("nan"),
            "ai_happy_avg_rt": float("nan"),
            "ai_happy_accuracy_pct": float("nan"),
            "non-ai_sad_avg_rt": float("nan"),
            "non-ai_sad_accuracy_pct": float("nan"),
            "non-ai_happy_avg_rt": float("nan"),
            "non-ai_happy_accuracy_pct": float("nan"),
        }

        # Try to find condition + correctness columns
        db_candidates = ["database", "db", "source", "stim_database", "face_database", "set", "dataset"]
        emo_candidates = ["emotion", "expression", "valence", "mood", "affect"]
        corr_candidates = ["correct", "corr", "accuracy", "resp_corr", "response_corr", "key_resp.corr", "resp.corr", "trial_corr"]

        database_col = pick_column(df, db_candidates)
        emotion_col = pick_column(df, emo_candidates)
        corr_col = pick_column(df, corr_candidates)

        # If not found, attempt inference from values
        if database_col is None or emotion_col is None:
            inf_db, inf_emo = infer_condition_columns(df)
            database_col = database_col or inf_db
            emotion_col = emotion_col or inf_emo

        # Coerce RT to numeric if present
        if "rt" in df.columns:
            df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

        # Coerce correctness if present
        correct_bool = None
        if corr_col is not None:
            correct_bool = coerce_correct_series(df[corr_col])

        if database_col is None or emotion_col is None:
            print(f"Warning: Could not determine condition columns for {filename}. Summary metrics left as NaN.")
        else:
            raw_db = df[database_col].astype(str).map(_lower)
            raw_emo = df[emotion_col].astype(str).map(_lower)

            def normalize_database(x: str):
                # Prefer non-ai detection first so "non-ai" doesn't get classified as "ai"
                if "non-ai" in x or "nonai" in x or "real" in x or "human" in x:
                    return "non-ai"
                if "ai" in x:
                    return "ai"
                return None

            def normalize_emotion(x: str):
                if "sad" in x:
                    return "sad"
                if "happy" in x:
                    return "happy"
                return None

            db_cat = raw_db.map(normalize_database)
            emo_cat = raw_emo.map(normalize_emotion)

            def compute_metrics(db_label, emo_label):
                mask = (db_cat == db_label) & (emo_cat == emo_label)
                sub = df.loc[mask].copy()
                avg_rt = sub["rt"].mean() if "rt" in sub.columns else float("nan")
                acc = float("nan")
                if correct_bool is not None:
                    acc = 100.0 * correct_bool.loc[mask].mean()
                return avg_rt, acc

            row["ai_sad_avg_rt"], row["ai_sad_accuracy_pct"] = compute_metrics("ai", "sad")
            row["ai_happy_avg_rt"], row["ai_happy_accuracy_pct"] = compute_metrics("ai", "happy")
            row["non-ai_sad_avg_rt"], row["non-ai_sad_accuracy_pct"] = compute_metrics("non-ai", "sad")
            row["non-ai_happy_avg_rt"], row["non-ai_happy_accuracy_pct"] = compute_metrics("non-ai", "happy")

        summary_rows.append(row)

        # Save back (overwrite)
        df.to_csv(file_path, index=False)

        print(f"Processed: {filename}")

# Save summary database
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("summary_database.csv", index=False)
    print(f"Saved summary_database.csv with {len(summary_df)} rows")

print("Done!")