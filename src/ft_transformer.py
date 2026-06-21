"""
ft_transformer.py
Feature Tokenizer + Transformer (FT-Transformer) for tabular regression.

Paper: Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
https://arxiv.org/abs/2106.11959

Key difference from TabTransformer:
  ┌─────────────────────────────────────────────────────────┐
  │  TabTransformer  : only categorical features go through │
  │                    Transformer; continuous are appended  │
  │                    after as raw values                   │
  │                                                         │
  │  FT-Transformer  : ALL features (cat + continuous) are  │
  │                    tokenized into the SAME embedding     │
  │                    space → richer cross-feature attention│
  └─────────────────────────────────────────────────────────┘

Architecture:
  Categorical   → Embedding lookup  ┐
  Continuous    → Linear projection ┘ → [CLS] + N feature tokens
                                          │
                                    Transformer Encoder (× L blocks)
                                          │
                                    [CLS] token → MLP head → output
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Dataset ───────────────────────────────────────────────────────────────────

class FTDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


# ── Feature Tokenizer ─────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Converts every feature (cat + num) into a d-dimensional token.

    Categorical  : standard nn.Embedding
    Continuous   : x_i * W_i + b_i  (per-feature linear projection)
    CLS token    : learnable vector prepended to the sequence
    """

    def __init__(self, cat_dims: list, n_num: int, d_token: int):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n + 1, d_token) for n in cat_dims   # +1 for unknown
        ])
        # Per-feature linear for continuous: weight (n_num, d_token), bias (n_num, d_token)
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias   = nn.Parameter(torch.zeros(n_num, d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
        else:
            self.num_weight = None

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.n_cat = len(cat_dims)
        self.n_num = n_num
        self.d_token = d_token

    def forward(self, x_cat, x_num):
        B = x_cat.shape[0]
        tokens = []

        # Categorical tokens: (B, n_cat, d_token)
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))

        # Continuous tokens: x_i * w_i + b_i → (B, n_num, d_token)
        if self.n_num > 0 and self.num_weight is not None:
            # x_num: (B, n_num) → (B, n_num, 1) * (n_num, d_token) → (B, n_num, d_token)
            num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)

        tokens = torch.cat(tokens, dim=1)             # (B, n_cat+n_num, d_token)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)      # (B, 1+n_features, d_token)
        return tokens


# ── Transformer blocks ────────────────────────────────────────────────────────

class FTTransformerBlock(nn.Module):
    """Pre-LN Transformer block (more stable than post-LN for small data)."""

    def __init__(self, d_token: int, n_heads: int, ffn_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * ffn_factor, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with pre-LN
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        # FFN with pre-LN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x


# ── Full FT-Transformer ───────────────────────────────────────────────────────

class FTTransformer(nn.Module):
    """
    Parameters
    ----------
    cat_dims    : list[int]  number of unique values per categorical feature
    n_num       : int        number of continuous features
    d_token     : int        token (embedding) dimension
    n_layers    : int        number of transformer blocks
    n_heads     : int        attention heads  (d_token must be divisible)
    ffn_factor  : int        FFN hidden = d_token × ffn_factor
    attn_dropout: float
    ffn_dropout : float
    mlp_hidden  : list[int]  MLP head hidden dims
    """

    def __init__(
        self,
        cat_dims: list,
        n_num: int,
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        ffn_factor: int = 4,
        attn_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        mlp_hidden: list = None,
    ):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        if mlp_hidden is None:
            mlp_hidden = [d_token * 2, d_token]

        self.tokenizer   = FeatureTokenizer(cat_dims, n_num, d_token)
        self.transformer = nn.Sequential(*[
            FTTransformerBlock(d_token, n_heads, ffn_factor, attn_dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_token)

        # MLP head operates on [CLS] token output
        layers, in_d = [], d_token
        for h in mlp_hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(ffn_dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        tokens = self.tokenizer(x_cat, x_num)    # (B, 1+n_features, d_token)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]                   # (B, d_token)  — [CLS] token
        return self.head(cls_out).squeeze(-1)


# ── Sklearn-compatible wrapper ────────────────────────────────────────────────

class FTTransformerRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for FT-Transformer.

    Parameters
    ----------
    cat_cols, num_cols : list[str]
    d_token     : int   — token dimension (paper default: 192)
    n_layers    : int   — transformer depth (paper uses 3 for small data)
    n_heads     : int   — attention heads
    ffn_factor  : int
    attn_dropout, ffn_dropout : float
    mlp_hidden  : list[int]
    epochs, batch_size, lr : training hyperparameters
    device      : 'auto' | 'cuda' | 'cpu'
    """

    def __init__(
        self,
        cat_cols: list,
        num_cols: list,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 8,
        ffn_factor: int = 4,
        attn_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        mlp_hidden: list = None,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-4,
        device: str = "auto",
    ):
        self.cat_cols     = cat_cols
        self.num_cols     = num_cols
        self.d_token      = d_token
        self.n_layers     = n_layers
        self.n_heads      = n_heads
        self.ffn_factor   = ffn_factor
        self.attn_dropout = attn_dropout
        self.ffn_dropout  = ffn_dropout
        self.mlp_hidden   = mlp_hidden or [d_token * 2, d_token]
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device

    def _encode(self, X, fit: bool = False):
        import pandas as pd
        df = pd.DataFrame(X, columns=self.cat_cols + self.num_cols) \
             if not isinstance(X, pd.DataFrame) else X[self.cat_cols + self.num_cols].copy()

        X_cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for i, col in enumerate(self.cat_cols):
            vals = df[col].astype(str).fillna("__NA__")
            if fit:
                self.label_encoders_[col] = LabelEncoder()
                X_cat[:, i] = self.label_encoders_[col].fit_transform(vals)
            else:
                le    = self.label_encoders_[col]
                known = set(le.classes_)
                vals  = vals.map(lambda v: v if v in known else le.classes_[0])
                X_cat[:, i] = le.transform(vals)

        if self.num_cols:
            X_num = df[self.num_cols].values.astype(np.float32)
            X_num = self.scaler_.fit_transform(X_num) if fit else self.scaler_.transform(X_num)
        else:
            X_num = np.zeros((len(df), 0), dtype=np.float32)

        return X_cat, X_num

    def fit(self, X, y):
        self.label_encoders_ = {}
        self.scaler_         = StandardScaler()
        X_cat, X_num = self._encode(X, fit=True)

        cat_dims = [len(le.classes_) for le in self.label_encoders_.values()]

        self.model_ = FTTransformer(
            cat_dims     = cat_dims,
            n_num        = X_num.shape[1],
            d_token      = self.d_token,
            n_layers     = self.n_layers,
            n_heads      = self.n_heads,
            ffn_factor   = self.ffn_factor,
            attn_dropout = self.attn_dropout,
            ffn_dropout  = self.ffn_dropout,
            mlp_hidden   = self.mlp_hidden,
        ).to(self.device)

        y_arr = np.array(y, dtype=np.float32)
        self.y_mean_ = y_arr.mean()
        self.y_std_  = y_arr.std() + 1e-8
        y_norm = (y_arr - self.y_mean_) / self.y_std_

        dataset = FTDataset(X_cat, X_num, y_norm)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        criterion = nn.HuberLoss()

        self.train_losses_ = []
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xc, xn, yb in loader:
                xc, xn, yb = xc.to(self.device), xn.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model_(xc, xn)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(yb)

            scheduler.step()
            self.train_losses_.append(epoch_loss / len(dataset))
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1:3d}/{self.epochs}]  loss={self.train_losses_[-1]:.4f}")
        return self

    def predict(self, X):
        X_cat, X_num = self._encode(X, fit=False)
        dataset = FTDataset(X_cat, X_num, np.zeros(len(X_cat)))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        preds = []
        with torch.no_grad():
            for xc, xn, _ in loader:
                xc, xn = xc.to(self.device), xn.to(self.device)
                preds.append(self.model_(xc, xn).cpu().numpy())
        preds = np.concatenate(preds)
        return preds * self.y_std_ + self.y_mean_

    def get_attention_weights(self, X) -> np.ndarray:
        """
        Extract attention weights from the first transformer block.
        Useful for visualising which feature pairs the model attends to.
        Returns shape: (n_samples, n_heads, n_tokens, n_tokens)
        """
        X_cat, X_num = self._encode(X, fit=False)
        xc = torch.tensor(X_cat, dtype=torch.long).to(self.device)
        xn = torch.tensor(X_num, dtype=torch.float32).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            tokens = self.model_.tokenizer(xc, xn)
            block  = self.model_.transformer[0]
            x_norm = block.norm1(tokens)
            _, attn_w = block.attn(x_norm, x_norm, x_norm, average_attn_weights=False)
        return attn_w.cpu().numpy()
