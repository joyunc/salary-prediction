"""
modeling.py
Unified training & evaluation pipeline.
Models: CART · Random Forest · MLP · TabTransformer · FT-Transformer
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(model, X_train, X_test, y_train, y_test, model_name="") -> dict:
    results = {}
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(X)
        results[split] = {
            "R2":   round(r2_score(y, pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 2),
            "MAPE": round(mape(y, pred), 2),
            "MAE":  round(mean_absolute_error(y, pred), 2),
        }
    if model_name:
        print(f"\n{'='*55}\n  {model_name}\n{'='*55}")
        for split in ("train", "test"):
            m = results[split]
            print(f"  [{split:5s}]  R²={m['R2']:.4f}  RMSE={m['RMSE']:>10,.0f}"
                  f"  MAPE={m['MAPE']:6.2f}%  MAE={m['MAE']:>10,.0f}")
    return results


# ── Feature helpers ───────────────────────────────────────────────────────────

def split_features(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target].values
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols, X, y


def encode_for_trees(X: pd.DataFrame, cat_cols: list):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc = X.copy()
    X_enc[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
    return X_enc.values, enc


# ── Classical ML ──────────────────────────────────────────────────────────────

def train_cart(X_tr, y_tr, cv=10):
    best_score, best_cp = -np.inf, 0.0
    for cp in np.linspace(0.0, 0.05, 20):
        m = DecisionTreeRegressor(min_impurity_decrease=cp, random_state=2056)
        s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="r2").mean()
        if s > best_score:
            best_score, best_cp = s, cp
    print(f"  Best CP={best_cp:.4f}  (CV R²={best_score:.4f})")
    m = DecisionTreeRegressor(min_impurity_decrease=best_cp, random_state=2056)
    m.fit(X_tr, y_tr)
    return m


def train_rf(X_tr, y_tr, n_features):
    mtry = max(1, n_features // 3)
    m = RandomForestRegressor(
        n_estimators=400, max_features=mtry,
        random_state=2056, n_jobs=-1, oob_score=True,
    )
    m.fit(X_tr, y_tr)
    print(f"  OOB R²={m.oob_score_:.4f}  mtry={mtry}")
    return m


def rf_importance(model, feature_names) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp["pct"] = (imp["importance"] / imp["importance"].sum() * 100).round(2)
    return imp.reset_index(drop=True)


# ── Deep Learning ─────────────────────────────────────────────────────────────

def train_mlp(X_tr_df, y_tr, cat_cols, num_cols, epochs=100):
    from src.mlp import MLPRegressor
    m = MLPRegressor(
        cat_cols=cat_cols, num_cols=num_cols,
        hidden_dims=[256, 128, 64], dropout=0.3,
        epochs=epochs, batch_size=256, lr=1e-3,
    )
    m.fit(X_tr_df, y_tr)
    return m


def train_tab_transformer(X_tr_df, y_tr, cat_cols, num_cols, epochs=80):
    from src.tab_transformer import TabTransformerRegressor
    m = TabTransformerRegressor(
        cat_cols=cat_cols, num_cols=num_cols,
        dim=32, depth=4, heads=4,
        epochs=epochs, batch_size=256, lr=1e-3,
    )
    m.fit(X_tr_df, y_tr)
    return m


def train_ft_transformer(X_tr_df, y_tr, cat_cols, num_cols, epochs=100):
    from src.ft_transformer import FTTransformerRegressor
    m = FTTransformerRegressor(
        cat_cols=cat_cols, num_cols=num_cols,
        d_token=64, n_layers=3, n_heads=8,
        attn_dropout=0.2, ffn_dropout=0.1,
        epochs=epochs, batch_size=256, lr=1e-4,
    )
    m.fit(X_tr_df, y_tr)
    return m


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, target: str, label: str = "", epochs: int = 80):
    """
    End-to-end pipeline for one salary model.
    Trains all 5 models and returns a comparison table.
    """
    cat_cols, num_cols, X, y = split_features(df, target)
    X_enc, _ = encode_for_trees(X, cat_cols)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_enc, y, test_size=0.3, random_state=2056
    )
    X_tr_df = pd.DataFrame(X_tr, columns=X.columns)
    X_te_df = pd.DataFrame(X_te, columns=X.columns)
    for c in cat_cols:
        X_tr_df[c] = X_tr_df[c].astype(str)
        X_te_df[c] = X_te_df[c].astype(str)

    print(f"\n{'#'*60}")
    print(f"  {label}  |  n={len(y)}  features={X.shape[1]}")
    print(f"  train={len(y_tr)}  test={len(y_te)}")
    print(f"{'#'*60}")

    models = {}

    print("\n[1/5] CART")
    models["CART"] = train_cart(X_tr, y_tr)
    cart_res = evaluate(models["CART"], X_tr, X_te, y_tr, y_te, "CART")

    print("\n[2/5] Random Forest")
    models["Random Forest"] = train_rf(X_tr, y_tr, X.shape[1])
    rf_res = evaluate(models["Random Forest"], X_tr, X_te, y_tr, y_te, "Random Forest")
    imp = rf_importance(models["Random Forest"], list(X.columns))
    print("\n  Top-10 Feature Importance:")
    print(imp.head(10).to_string(index=False))

    print("\n[3/5] MLP (baseline)")
    models["MLP"] = train_mlp(X_tr_df, y_tr, cat_cols, num_cols, epochs=epochs)
    mlp_res = evaluate(models["MLP"], X_tr_df, X_te_df, y_tr, y_te, "MLP")

    print("\n[4/5] TabTransformer")
    models["TabTransformer"] = train_tab_transformer(X_tr_df, y_tr, cat_cols, num_cols, epochs=epochs)
    tab_res = evaluate(models["TabTransformer"], X_tr_df, X_te_df, y_tr, y_te, "TabTransformer")

    print("\n[5/5] FT-Transformer")
    models["FT-Transformer"] = train_ft_transformer(X_tr_df, y_tr, cat_cols, num_cols, epochs=epochs)
    ft_res  = evaluate(models["FT-Transformer"], X_tr_df, X_te_df, y_tr, y_te, "FT-Transformer")

    # Comparison table
    all_res = {
        "CART": cart_res, "Random Forest": rf_res,
        "MLP": mlp_res, "TabTransformer": tab_res, "FT-Transformer": ft_res,
    }
    rows = [
        {"Model": name, "Split": split, **res[split]}
        for name, res in all_res.items()
        for split in ("train", "test")
    ]
    comparison = pd.DataFrame(rows)
    print(f"\n{'='*60}\n  Final Comparison — {label}\n{'='*60}")
    print(comparison.to_string(index=False))

    return {
        "models": models,
        "results": all_res,
        "feature_importance": imp,
        "comparison": comparison,
        "splits": (X_tr_df, X_te_df, y_tr, y_te),
    }
