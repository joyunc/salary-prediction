"""
mlp.py
Simple MLP baseline for tabular regression.
Serves as DL baseline to compare against TabTransformer and FT-Transformer.

Architecture:
  Input (cat embeddings + continuous)
    → BatchNorm
    → Linear → ReLU → Dropout  (× N layers)
    → Linear → scalar output
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MLPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible MLP for tabular regression.
    Categorical features → learned embeddings (dim = min(50, (n_cats+1)//2))
    Continuous features  → StandardScaler
    """

    def __init__(
        self,
        cat_cols: list,
        num_cols: list,
        hidden_dims: list = None,
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        self.cat_cols    = cat_cols
        self.num_cols    = num_cols
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.device      = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device

    def _get_embed_dim(self, n_cats: int) -> int:
        return min(50, (n_cats + 1) // 2 + 1)

    def _prepare(self, X, fit: bool = False):
        import pandas as pd
        df = pd.DataFrame(X, columns=self.cat_cols + self.num_cols) \
             if not isinstance(X, pd.DataFrame) else X.copy()

        cat_tensors = []
        for col in self.cat_cols:
            vals = df[col].astype(str).fillna("__NA__")
            if fit:
                self.label_encoders_[col] = LabelEncoder()
                encoded = self.label_encoders_[col].fit_transform(vals)
            else:
                le = self.label_encoders_[col]
                known = set(le.classes_)
                vals = vals.map(lambda v: v if v in known else le.classes_[0])
                encoded = le.transform(vals)
            cat_tensors.append(torch.tensor(encoded, dtype=torch.long))

        if self.num_cols:
            num_arr = df[self.num_cols].values.astype(np.float32)
            if fit:
                num_arr = self.scaler_.fit_transform(num_arr)
            else:
                num_arr = self.scaler_.transform(num_arr)
        else:
            num_arr = np.zeros((len(df), 0), dtype=np.float32)

        return cat_tensors, torch.tensor(num_arr, dtype=torch.float32)

    def fit(self, X, y):
        import pandas as pd
        self.label_encoders_ = {}
        self.scaler_         = StandardScaler()

        cat_tensors, num_tensor = self._prepare(X, fit=True)

        # Build embedding layers
        self.embeddings_ = nn.ModuleList()
        embed_out_dim = 0
        for col in self.cat_cols:
            n_cats = len(self.label_encoders_[col].classes_)
            emb_dim = self._get_embed_dim(n_cats)
            self.embeddings_.append(nn.Embedding(n_cats + 1, emb_dim))
            embed_out_dim += emb_dim

        input_dim = embed_out_dim + num_tensor.shape[1]
        self.model_ = MLPNet(input_dim, self.hidden_dims, self.dropout).to(self.device)
        self.embeddings_.to(self.device)

        y_arr = np.array(y, dtype=np.float32)
        self.y_mean_ = y_arr.mean()
        self.y_std_  = y_arr.std() + 1e-8
        y_norm = torch.tensor((y_arr - self.y_mean_) / self.y_std_, dtype=torch.float32)

        # Build dataset manually (cat + num + y)
        n = len(y_arr)
        dataset = list(zip(
            *[t for t in cat_tensors],
            num_tensor,
            y_norm,
        ))

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        params = list(self.model_.parameters()) + list(self.embeddings_.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.HuberLoss()

        self.train_losses_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            self.embeddings_.train()
            epoch_loss = 0.0
            for batch in loader:
                *cat_batch, num_batch, y_batch = batch
                cat_batch = [t.to(self.device) for t in cat_batch]
                num_batch = num_batch.to(self.device)
                y_batch   = y_batch.to(self.device)

                embs = [e(c) for e, c in zip(self.embeddings_, cat_batch)]
                x = torch.cat(embs + [num_batch], dim=1)

                optimizer.zero_grad()
                pred = self.model_(x)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)

            scheduler.step()
            self.train_losses_.append(epoch_loss / n)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1:3d}/{self.epochs}]  loss={self.train_losses_[-1]:.4f}")
        return self

    def predict(self, X):
        cat_tensors, num_tensor = self._prepare(X, fit=False)
        dataset = list(zip(*[t for t in cat_tensors], num_tensor))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        self.embeddings_.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                *cat_batch, num_batch = batch
                cat_batch = [t.to(self.device) for t in cat_batch]
                num_batch = num_batch.to(self.device)
                embs = [e(c) for e, c in zip(self.embeddings_, cat_batch)]
                x = torch.cat(embs + [num_batch], dim=1)
                preds.append(self.model_(x).cpu().numpy())

        preds = np.concatenate(preds)
        return preds * self.y_std_ + self.y_mean_
