"""
tab_transformer.py
TabTransformer for tabular regression — PyTorch implementation.

Architecture (Huang et al., 2020):
  ┌─────────────────────────────────────────────┐
  │  Categorical features                       │
  │    → Embedding → Transformer Encoder        │
  │  Continuous features                        │
  │    → LayerNorm                              │
  │  Concat → MLP head → scalar output          │
  └─────────────────────────────────────────────┘
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Dataset ──────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


# ── Transformer building blocks ───────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # each (B, T, H, head_dim)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, H, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = self.drop(attn.softmax(dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ── Full TabTransformer ───────────────────────────────────────────────────────

class TabTransformer(nn.Module):
    """
    Parameters
    ----------
    cat_dims      : list[int]  — number of unique values per categorical feature
    num_continuous: int        — number of continuous features
    dim           : int        — embedding / transformer hidden size
    depth         : int        — number of transformer blocks
    heads         : int        — attention heads
    attn_dropout  : float
    ff_dropout    : float
    mlp_hidden    : list[int]  — MLP head hidden layer sizes
    """

    def __init__(
        self,
        cat_dims: list,
        num_continuous: int,
        dim: int = 32,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden: list = None,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [128, 64]

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats + 1, dim)   # +1 for unknown / padding
            for n_cats in cat_dims
        ])

        # Transformer encoder over categorical embeddings
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, dropout=attn_dropout) for _ in range(depth)]
        )

        # Continuous feature normalisation
        self.num_norm = nn.LayerNorm(num_continuous) if num_continuous > 0 else None

        # MLP head
        mlp_input = dim * len(cat_dims) + num_continuous
        layers = []
        in_dim = mlp_input
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(ff_dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        # Embed each categorical feature → (B, n_cat, dim)
        embs = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        embs = self.transformer(embs)                        # (B, n_cat, dim)
        embs = embs.flatten(1)                               # (B, n_cat * dim)

        if x_num.shape[1] > 0 and self.num_norm is not None:
            x_num = self.num_norm(x_num)

        x = torch.cat([embs, x_num], dim=1)
        return self.mlp(x).squeeze(-1)


# ── Sklearn-compatible wrapper ────────────────────────────────────────────────

class TabTransformerRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper around TabTransformer.

    Parameters
    ----------
    cat_cols   : list[str]  — names of categorical columns
    num_cols   : list[str]  — names of numerical columns
    dim        : int
    depth      : int
    heads      : int
    attn_dropout, ff_dropout : float
    mlp_hidden : list[int]
    epochs     : int
    batch_size : int
    lr         : float
    device     : 'cuda' | 'cpu' | 'auto'
    """

    def __init__(
        self,
        cat_cols: list,
        num_cols: list,
        dim: int = 32,
        depth: int = 4,
        heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden: list = None,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        self.cat_cols    = cat_cols
        self.num_cols    = num_cols
        self.dim         = dim
        self.depth       = depth
        self.heads       = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout  = ff_dropout
        self.mlp_hidden  = mlp_hidden or [128, 64]
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.device      = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device

    # ------------------------------------------------------------------
    def _encode(self, X: np.ndarray, fit: bool = False):
        """Label-encode categoricals, scale numericals."""
        import pandas as pd
        df = pd.DataFrame(X, columns=self.cat_cols + self.num_cols)

        X_cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for i, col in enumerate(self.cat_cols):
            if fit:
                self.label_encoders_[col] = LabelEncoder()
                X_cat[:, i] = self.label_encoders_[col].fit_transform(
                    df[col].astype(str).fillna("__NA__")
                )
            else:
                le = self.label_encoders_[col]
                vals = df[col].astype(str).fillna("__NA__")
                # Handle unseen labels
                known = set(le.classes_)
                vals  = vals.map(lambda v: v if v in known else le.classes_[0])
                X_cat[:, i] = le.transform(vals)

        X_num = df[self.num_cols].values.astype(np.float32) if self.num_cols else np.zeros((len(df), 0), dtype=np.float32)
        if self.num_cols:
            if fit:
                X_num = self.scaler_.fit_transform(X_num)
            else:
                X_num = self.scaler_.transform(X_num)
        return X_cat, X_num

    # ------------------------------------------------------------------
    def fit(self, X, y):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X[self.cat_cols + self.num_cols].values

        self.label_encoders_ = {}
        self.scaler_         = StandardScaler()
        X_cat, X_num = self._encode(X, fit=True)

        # Infer cat_dims from label encoders
        cat_dims = [len(le.classes_) for le in self.label_encoders_.values()]

        self.model_ = TabTransformer(
            cat_dims       = cat_dims,
            num_continuous = X_num.shape[1],
            dim            = self.dim,
            depth          = self.depth,
            heads          = self.heads,
            attn_dropout   = self.attn_dropout,
            ff_dropout     = self.ff_dropout,
            mlp_hidden     = self.mlp_hidden,
        ).to(self.device)

        y_arr = np.array(y, dtype=np.float32)
        self.y_mean_ = y_arr.mean()
        self.y_std_  = y_arr.std() + 1e-8
        y_norm = (y_arr - self.y_mean_) / self.y_std_

        dataset = TabularDataset(X_cat, X_num, y_norm)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.HuberLoss()

        self.model_.train()
        self.train_losses_ = []
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
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1:3d}/{self.epochs}]  loss={self.train_losses_[-1]:.4f}")
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X[self.cat_cols + self.num_cols].values
        X_cat, X_num = self._encode(X, fit=False)
        dataset = TabularDataset(X_cat, X_num, np.zeros(len(X_cat)))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        preds = []
        with torch.no_grad():
            for xc, xn, _ in loader:
                xc, xn = xc.to(self.device), xn.to(self.device)
                preds.append(self.model_(xc, xn).cpu().numpy())
        preds = np.concatenate(preds)
        return preds * self.y_std_ + self.y_mean_
