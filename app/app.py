import os
import time
import hashlib
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

app = Flask(__name__)
app.secret_key = "replace-with-a-secret-key"

# ========= Load model bundle (pipeline + metadata) =========
MODEL_PATH = os.path.join("models", "best_pipeline.pkl")

def _file_digest(path: str):
    """Return (md5, mtime_str) for a file and print a helpful log line."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    try:
        mtime = time.ctime(os.path.getmtime(path))
    except Exception:
        mtime = "unknown"
    digest = h.hexdigest()
    print(f">> Loading {path} (md5={digest}, mtime={mtime})")
    return digest, mtime

MODEL_MD5, MODEL_MTIME = _file_digest(MODEL_PATH)  # printed on startup

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
cat_cols = bundle.get("cat_cols", [])
num_cols = bundle.get("num_cols", [])
feature_order = bundle.get("feature_order", cat_cols + num_cols)
num_bounds = bundle.get("num_bounds", {})  # NEW: learned quantile bounds

# ========= Derive categorical choices from OneHotEncoder =========
def extract_cat_choices_from_model(
    pipeline,
    preferred_cols=("Galaxy Type", "Environment", "Rotation Curve Shape"),
):
    """Return {col: [allowed_values,...]} from OneHotEncoder.categories_."""
    from sklearn.preprocessing import OneHotEncoder
    choices = {}
    pre = pipeline.named_steps.get("preprocess")
    if pre is None:
        print(">> preprocess step not found; cannot extract categories.")
        return choices

    enc = None
    cat_cols_from_ct = None
    try:
        if hasattr(pre, "transformers_"):
            for name, trans, cols in pre.transformers_:
                # Handle encoder nested in a Pipeline
                if hasattr(trans, "named_steps"):
                    for step in trans.named_steps.values():
                        if isinstance(step, OneHotEncoder):
                            enc = step
                            cat_cols_from_ct = cols
                            break
                # Or direct encoder
                if enc is None and "OneHotEncoder" in type(trans).__name__:
                    enc = trans
                    cat_cols_from_ct = cols
                if enc is not None:
                    break

        if enc is not None and hasattr(enc, "categories_") and cat_cols_from_ct is not None:
            for col, cats in zip(list(cat_cols_from_ct), enc.categories_):
                if col in preferred_cols:
                    clean = [str(c).strip() for c in cats]
                    seen = set()
                    ordered = [x for x in clean if (x not in seen and not seen.add(x))]
                    choices[col] = ordered
        else:
            print(">> OneHotEncoder with categories_ not found; cannot build dropdowns.")
    except Exception as e:
        print(">> Category extraction failed:", e)

    return choices

CHOICES_BY_COL = extract_cat_choices_from_model(pipeline)
print(">> Dropdown choices discovered:", CHOICES_BY_COL)

# ========= Field metadata (hints & ranges for numerics) =========
FIELD_META = {
    # Categorical (UI hint text only; actual choices come from encoder)
    "Galaxy Type": {
        "placeholder": "Select a type",
        "suggestion": "Examples: Elliptical, Spiral, Irregular, Lenticular."
    },
    "Environment": {
        "placeholder": "Select an environment",
        "suggestion": "Examples: Cluster, Group, Field, Void."
    },
    "Rotation Curve Shape": {
        "placeholder": "Select a shape",
        "suggestion": "Common: Flat, Rising, Falling."
    },

    # Numerics (defaults; will be tightened by num_bounds below if available)
    "Lensing Shear |gamma|": {"placeholder": "0.01–0.10", "min": 0.0, "max": 1.0, "unit": "|γ|",
                              "suggestion": "Typical weak-lensing shear magnitudes are ~0.01–0.10."},
    "Redshift": {"placeholder": "0–10", "min": 0.0, "max": 10.0},
    "Baryonic Mass (Msun)": {"placeholder": "1e7–1e12", "min": 1e7, "max": 1e12, "unit": "M☉"},
    "Surface Brightness (mag/arcsec^2)": {"placeholder": "10–30", "min": 5.0, "max": 35.0},
    "Velocity Dispersion (km/s)": {"placeholder": "0–500", "min": 0.0, "max": 500.0, "unit": "km/s"},
    "HI Line Width (km/s)": {"placeholder": "0–600", "min": 0.0, "max": 600.0, "unit": "km/s"},
    "Outer Curve Slope": {"placeholder": "-1.0–1.0", "min": -1.0, "max": 1.0},
    "Mass-to-Light Ratio": {"placeholder": "0–20", "min": 0.0, "max": 20.0},
    "Halo Concentration": {"placeholder": "0–30", "min": 0.0, "max": 30.0},
    "Gas Fraction": {"placeholder": "0–1", "min": 0.0, "max": 1.0},
    "Stellar Age (Gyr)": {"placeholder": "0–13.8", "min": 0.0, "max": 13.8, "unit": "Gyr"},
    "Metallicity [Fe/H]": {"placeholder": "-3–1", "min": -3.0, "max": 1.0},
    "Environment Density": {"placeholder": "0–5", "min": 0.0, "max": 5.0},
    "TF Residual (mag)": {"placeholder": "-2–2", "min": -2.0, "max": 2.0, "unit": "mag"},
}

# NEW: tighten numeric ranges/placeholders using learned dataset quantiles
if num_bounds:
    for col, b in num_bounds.items():
        m = FIELD_META.setdefault(col, {})
        # Validate to p01–p99; show p05–p95 as hint for realism
        m["min"] = b.get("p01", m.get("min"))
        m["max"] = b.get("p99", m.get("max"))
        # Always show data-driven placeholder so users stay in-distribution
        try:
            m["placeholder"] = f"{b['p05']:.4g}–{b['p95']:.4g}"
        except Exception:
            pass

# ========= Form Building =========
def build_form_spec(values=None, errors=None):
    values = values or {}
    errors = errors or {}
    fields = []
    for col in feature_order:
        meta = FIELD_META.get(col, {})
        # UI decision: select if we have model-derived choices, else numeric or text
        if col in CHOICES_BY_COL and CHOICES_BY_COL[col]:
            ui_type = "select"
        elif col in cat_cols:
            ui_type = "text"  # fallback if encoder choices unavailable
        else:
            ui_type = "number"

        base = {
            "name": col,
            "placeholder": meta.get("placeholder", f"Enter {col}"),
            "suggestion": meta.get("suggestion"),
            "value": values.get(col, ""),
            "error": errors.get(col)
        }
        if ui_type == "select":
            fields.append({**base, "type": "select", "choices": CHOICES_BY_COL[col]})
        elif ui_type == "number":
            fields.append({
                **base, "type": "number",
                "min": meta.get("min"), "max": meta.get("max"), "unit": meta.get("unit"),
            })
        else:
            fields.append({**base, "type": "text"})
    return fields

# ========= Validation (now returns OOD flags) =========
def validate_input(form):
    values, errors = {}, {}
    ood_flags = {}  # {col: (val, p01, p99)}
    for col in feature_order:
        raw = (form.get(col) or "").strip()
        meta = FIELD_META.get(col, {})
        bounds = num_bounds.get(col)

        if raw == "":
            errors[col] = f"{col} is required."
            values[col] = ""
            continue

        # Enforce dropdown choice if available
        allowed = CHOICES_BY_COL.get(col)
        if allowed:
            if raw not in allowed:
                errors[col] = f"Please choose a valid value for {col}."
            values[col] = raw
            continue

        # Fallback: categorical/text vs numeric
        is_text = col in cat_cols
        if is_text:
            values[col] = raw
        else:
            try:
                val = float(raw)
                if meta.get("min") is not None and val < meta["min"]:
                    errors[col] = f"{col} must be ≥ {meta['min']:.4g}."
                if meta.get("max") is not None and val > meta["max"]:
                    errors[col] = f"{col} must be ≤ {meta['max']:.4g}."
                # Soft OOD flag if outside learned p01–p99
                if bounds:
                    if val < bounds["p01"] or val > bounds["p99"]:
                        ood_flags[col] = (val, bounds["p01"], bounds["p99"])
                values[col] = raw
            except ValueError:
                errors[col] = f"{col} must be a number."
                values[col] = raw
    return values, errors, ood_flags

# ========= Robust probability/label helpers =========
def _final_classes_of(clf):
    classes = getattr(clf, "classes_", None)
    if classes is not None:
        return np.asarray(classes)
    if hasattr(clf, "named_steps"):
        for step in reversed(list(clf.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    return None

def _positive_int_from_label(y):
    if isinstance(y, (int, np.integer)):
        return int(y == 1)
    s = str(y).strip().lower()
    return 1 if s in {"1", "yes", "true", "detected", "positive", "dark matter detected"} else 0

def _predict_with_confidence(clf, X_df):
    classes = _final_classes_of(clf)
    y_pred = clf.predict(X_df)[0]

    proba_vec = None
    if hasattr(clf, "predict_proba"):
        proba_vec = clf.predict_proba(X_df)[0]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_df)
        if scores.ndim == 1:
            p1 = float(1.0 / (1.0 + np.exp(-scores[0])))
            proba_vec = np.array([1.0 - p1, p1], dtype=float)
        else:
            e = np.exp(scores - np.max(scores))
            proba_vec = (e / e.sum()).ravel()

    # probability to display
    if proba_vec is not None and classes is not None and len(classes) == len(proba_vec):
        cls_list = list(classes)
        if 1 in cls_list:
            show_proba = float(proba_vec[cls_list.index(1)])
        elif y_pred in cls_list:
            show_proba = float(proba_vec[cls_list.index(y_pred)])
        else:
            show_proba = float(np.max(proba_vec))
    elif proba_vec is not None:
        show_proba = float(np.max(proba_vec))
    else:
        show_proba = 0.5

    # predicted class → 0/1
    if classes is not None and 1 in set(classes):
        pred_int = 1 if y_pred == 1 else 0
    else:
        mapped = _positive_int_from_label(y_pred)
        pred_int = mapped if mapped in (0, 1) else int(show_proba >= 0.5)

    show_proba = min(max(show_proba, 0.0), 1.0)
    return pred_int, show_proba, y_pred, classes, proba_vec

# ========= Routes =========
@app.route("/", methods=["GET"])
def index():
    print(">> cat_cols from model:", cat_cols)
    print(">> Dropdowns for:", list(CHOICES_BY_COL.keys()))
    return render_template("index.html", inputs=build_form_spec())

@app.route("/predict", methods=["POST"])
def predict():
    values, errors, ood_flags = validate_input(request.form)
    if errors:
        flash("Please correct the highlighted fields.", "danger")
        return render_template("index.html", inputs=build_form_spec(values, errors)), 400

    # Convert values (respect types)
    row = {}
    for col in feature_order:
        v = values[col]
        if (col in cat_cols) or (col in CHOICES_BY_COL):
            row[col] = v
        else:
            row[col] = float(v)

    X_input = pd.DataFrame([row], columns=feature_order)

    # Debug
    try:
        print(">> INPUT ROW:", row)
        classes_dbg = _final_classes_of(pipeline)
        if classes_dbg is not None:
            print(">> MODEL classes_:", list(classes_dbg))
    except Exception as e:
        print(">> Debug print failed:", e)

    pred_class, proba, y_pred_raw, classes, proba_vec = _predict_with_confidence(pipeline, X_input)

    try:
        if proba_vec is not None and classes is not None:
            print(">> PROBA vector:", {str(c): float(p) for c, p in zip(classes, proba_vec)})
        print(">> y_pred:", y_pred_raw, "=> pred_class:", pred_class, "display_proba:", proba)
    except Exception as e:
        print(">> Debug proba/classes print failed:", e)

    confidence = max(proba, 1 - proba)
    is_ood = int(bool(ood_flags))  # 0/1 for query string

    return redirect(url_for("result",
                            pred=pred_class,
                            proba=f"{proba:.4f}",
                            conf=f"{confidence:.4f}",
                            ood=is_ood))

@app.route("/result", methods=["GET"])
def result():
    pred = int(request.args.get("pred"))
    proba = float(request.args.get("proba"))
    conf = float(request.args.get("conf"))
    ood = bool(int(request.args.get("ood", "0")))

    label = "Dark Matter Detected" if pred == 1 else "Dark Matter Not Detected"
    probability_pct = f"{proba * 100:.2f}%"
    confidence_pct = f"{conf * 100:.2f}%"

    return render_template(
        "result.html",
        label=label,
        probability_pct=probability_pct,
        confidence_pct=confidence_pct,
        is_positive=(pred == 1),
        ood=ood
    )

# ========= Debug endpoints =========
@app.route("/_modelinfo", methods=["GET"])
def modelinfo():
    """Inspect loaded model & encoder — useful to confirm you replaced the pickle."""
    info = {
        "model_path": MODEL_PATH,
        "model_md5": MODEL_MD5,
        "model_mtime": MODEL_MTIME,
        "feature_order": feature_order,
        "cat_cols_from_bundle": cat_cols,
        "num_cols_from_bundle": num_cols,
        "dropdown_choices": CHOICES_BY_COL,
        "classes_": None,
        "n_transformed_features": None,
        "has_num_bounds": bool(num_bounds),
    }
    try:
        clf = pipeline.named_steps.get("clf", None)
        if hasattr(clf, "classes_"):
            info["classes_"] = [int(c) if str(c).isdigit() else str(c) for c in clf.classes_]
    except Exception:
        pass

    try:
        pre = pipeline.named_steps.get("preprocess")
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            fn = pre.get_feature_names_out()
            info["n_transformed_features"] = int(len(fn))
        else:
            if feature_order:
                dummy = {c: 0.0 for c in feature_order}
                for c in cat_cols:
                    dummy[c] = ""
                Xd = pd.DataFrame([dummy], columns=feature_order)
                Xt = pre.transform(Xd)
                info["n_transformed_features"] = int(Xt.shape[1])
            else:
                info["n_transformed_features"] = "unknown"
    except Exception as e:
        info["n_transformed_features"] = f"error: {e}"

    # Small preview of bounds
    try:
        preview = {}
        for k in list(num_bounds.keys())[:5]:
            b = num_bounds[k]
            preview[k] = {"p01": b["p01"], "p99": b["p99"], "p05": b["p05"], "p95": b["p95"]}
        info["num_bounds_preview"] = preview
    except Exception:
        pass

    return jsonify(info)

@app.route("/_selftest", methods=["GET"])
def selftest():
    """Create two distinct in-distribution rows and compare probability vectors."""
    def pick_numeric(col, which="low"):
        b = num_bounds.get(col)
        if not b:
            return 0.0
        return float(b["p05"] if which == "low" else b["p95"])

    rowA, rowB = {}, {}
    for col in feature_order:
        opts = CHOICES_BY_COL.get(col)
        if opts:
            rowA[col] = opts[0]
            rowB[col] = opts[-1] if len(opts) > 1 else opts[0]
        elif col in cat_cols:
            rowA[col] = rowB[col] = "NA"
        else:
            rowA[col] = pick_numeric(col, "low")
            rowB[col] = pick_numeric(col, "high")

    X_A = pd.DataFrame([rowA], columns=feature_order)
    X_B = pd.DataFrame([rowB], columns=feature_order)

    def proba_vec(X):
        if hasattr(pipeline, "predict_proba"):
            return list(map(float, pipeline.predict_proba(X)[0]))
        elif hasattr(pipeline, "decision_function"):
            s = pipeline.decision_function(X)
            if np.ndim(s) == 1:
                p1 = float(1 / (1 + np.exp(-s[0])))
                return [1 - p1, p1]
            e = np.exp(s - np.max(s))
            z = e / e.sum()
            return list(map(float, z.ravel()))
        return [0.5, 0.5]

    classes = getattr(getattr(pipeline, "named_steps", {}).get("clf", pipeline), "classes_", None)
    return jsonify({
        "classes_": [str(x) for x in classes] if classes is not None else None,
        "rowA": rowA, "proba_rowA": proba_vec(X_A),
        "rowB": rowB, "proba_rowB": proba_vec(X_B)
    })

if __name__ == "__main__":
    # Bind to 0.0.0.0 if you deploy; debug=True for local development
    app.run(debug=True)
