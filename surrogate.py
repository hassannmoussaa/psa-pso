import numpy as np
import torch
import torch.nn as nn
# ---- Simple MLP ----

class MLP(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---- Train ----
def train_surrogate(model , D=10, n=20000, epochs=30, lr=1e-3, bounds=(-0.5, 0.5), function = None,  device=None ):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # data
    X = np.random.uniform(bounds[0], bounds[1], size=(n, D)).astype(np.float32)
    y = function(X).astype(np.float32)
    y[~np.isfinite(y)] = 1e12

    # simple scaling for stability
    Xmean, Xstd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    ymean, ystd = y.mean(), y.std() + 1e-8
    Xs = (X - Xmean) / Xstd
    ys = (y - ymean) / ystd

    Xs = torch.tensor(Xs, device=device)
    ys = torch.tensor(ys, device=device)


    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    bs = 512
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            pred = model(Xs[b])
            loss = loss_fn(pred, ys[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        

        
            
    # return model + scalers so you can predict on real scale
    scalers = (Xmean, Xstd, ymean, ystd)
    return model, scalers, device

# ---- Predict ----
@torch.no_grad()
def predict(model, scalers, X, device):
    Xmean, Xstd, ymean, ystd = scalers
    X = np.asarray(X, dtype=np.float32)
    Xs = (X - Xmean) / Xstd
    Xs = torch.tensor(Xs, device=device)
    ys_pred = model(Xs).cpu().numpy()
    y_pred = ys_pred * ystd + ymean
    return y_pred

@torch.no_grad()
def clamp_(x, lo, hi):
    x.clamp_(lo, hi)
    return x

def minimize_surrogate_gd(model, scalers, D, bounds=(-2,2), steps=50, lr=0.05, n_restarts=2, device="cpu"):
    Xmean, Xstd, ymean, ystd = scalers
    lo, hi = bounds

    best_x = None
    best_val = float("inf")

    for _ in range(n_restarts):
        # start from random point in bounds (in original space)
        x0 = torch.empty(D, device=device).uniform_(lo, hi).requires_grad_(True)
        opt = torch.optim.Adam([x0], lr=lr)

        for _ in range(steps):
            # normalize x0 like training
            x_norm = (x0 - torch.tensor(Xmean.squeeze(0), device=device)) / torch.tensor(Xstd.squeeze(0), device=device)
            y_pred_std = model(x_norm.unsqueeze(0)).squeeze(0)   # standardized y
            # minimizing standardized y is same as minimizing original y (affine transform)
            loss = y_pred_std

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                clamp_(x0, lo, hi)

        with torch.no_grad():
            x_norm = (x0 - torch.tensor(Xmean.squeeze(0), device=device)) / torch.tensor(Xstd.squeeze(0), device=device)
            y_std = model(x_norm.unsqueeze(0)).item()
            y = y_std * ystd + ymean

            if y < best_val:
                best_val = y
                best_x = x0.detach().cpu().numpy().copy()

    return best_x, best_val
