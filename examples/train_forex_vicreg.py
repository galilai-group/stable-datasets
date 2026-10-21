
"""Train VICReg on Forex and evaluate learned representations."""

from copy import deepcopy

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_datasets.timeseries import Forex
from stable_pretraining.methods.vicreg import VICReg


torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(2)


class ForexDataset(Dataset):
    def __init__(self, split, limit):
        self.source = Forex(split=split).with_format("numpy")
        self.length = min(limit, len(self.source))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        series = np.asarray(
            self.source[index]["series"], dtype=np.float32
        ).copy()
        return torch.from_numpy(series)


class TimeSeriesEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        # Compatibility with the framework's image-shaped encoder probe.
        if x.ndim == 4 and x.shape == (1, 3, 224, 224):
            x = x.new_zeros((1, 30, 4))
        return self.network(x)


def augment(x):
    return x + torch.randn_like(x) * 0.005


def evaluate_ssl(model, loader):
    model.eval()
    losses = []
    embeddings = []

    with torch.no_grad():
        for x in loader:
            output = model(augment(x), augment(x))
            losses.append(output.loss.item())
            embeddings.append(model(x).embedding)

    z = torch.cat(embeddings, dim=0)
    return float(np.mean(losses)), z.std(dim=0).mean().item()


def extract_probe_data(encoder, loader):
    encoder.eval()
    all_embeddings = []
    all_features = []
    all_targets = []

    with torch.no_grad():
        for x in loader:
            # Forex windows are normalized relative to the first close.
            close = 1.0 + x[:, :, 3]
            log_returns = torch.log(
                close[:, 1:] / close[:, :-1]
            )

            # Target: realized volatility within the input window.
            volatility = log_returns.std(dim=1, unbiased=True)

            # Handcrafted features exclude the exact volatility target.
            features = torch.stack(
                [
                    log_returns.mean(dim=1),
                    log_returns.abs().mean(dim=1),
                    x[:, :, 0].mean(dim=1),
                    x[:, :, 1].mean(dim=1),
                    x[:, :, 2].mean(dim=1),
                    x[:, :, 3].mean(dim=1),
                ],
                dim=1,
            )

            all_embeddings.append(encoder(x).cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_targets.append(volatility.cpu().numpy())

    return (
        np.concatenate(all_embeddings),
        np.concatenate(all_features),
        np.concatenate(all_targets),
    )


def run_probe(name, x_train, y_train, x_test, y_test):
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(x_train, y_train)

    predictions = regressor.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{name}: MSE={mse:.8f}, R2={r2:.4f}")


def main():
    train_loader = DataLoader(
        ForexDataset("train", 2048),
        batch_size=64,
        shuffle=False,
    )
    val_loader = DataLoader(
        ForexDataset("validation", 512),
        batch_size=64,
    )
    test_loader = DataLoader(
        ForexDataset("test", 512),
        batch_size=64,
    )

    encoder = TimeSeriesEncoder()
    random_encoder = deepcopy(encoder)

    model = VICReg(
        encoder_name=encoder,
        projector_dims=(128, 128, 64),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("=== Forex VICReg pretraining ===")

    for epoch in range(10):
        model.train()
        losses = []

        for x in train_loader:
            optimizer.zero_grad()

            output = model(augment(x), augment(x))
            loss = output.loss

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite VICReg loss")

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val_loss, val_std = evaluate_ssl(model, val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={np.mean(losses):.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_embedding_std={val_std:.4f}"
        )

    test_loss, test_std = evaluate_ssl(model, test_loader)

    print(f"Final test VICReg loss: {test_loss:.4f}")
    print(f"Final test embedding std: {test_std:.4f}")

    print("\n=== Forex linear probe evaluation ===")

    z_train, basic_train, y_train = extract_probe_data(
        encoder, train_loader
    )
    z_test, basic_test, y_test = extract_probe_data(
        encoder, test_loader
    )

    random_train, _, _ = extract_probe_data(
        random_encoder, train_loader
    )
    random_test, _, _ = extract_probe_data(
        random_encoder, test_loader
    )

    run_probe(
        "Trained VICReg", z_train, y_train, z_test, y_test
    )
    run_probe(
        "Random encoder",
        random_train,
        y_train,
        random_test,
        y_test,
    )
    run_probe(
        "Simple financial features",
        basic_train,
        y_train,
        basic_test,
        y_test,
    )


if __name__ == "__main__":
    main()
