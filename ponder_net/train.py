import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import PonderNet, ReconstructionLoss, RegularizationLoss
from data import ParityDataset

def train():
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    train_n_batches = 500
    batch_size = 128
    n_elements = 8
    h_dim = 64
    max_steps = 20
    lambda_prior = 0.2
    test_n_batches = 32
    beta = 0.01
    grad_norm_clip = 1.0
    lr = 3e-4

    model = PonderNet(n_elements, h_dim, max_steps).to(device)
    loss_rec = ReconstructionLoss(nn.BCEWithLogitsLoss(reduction="none")).to(device)
    loss_reg = RegularizationLoss(lambda_prior, max_steps).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(ParityDataset(batch_size * train_n_batches, n_elements), batch_size=batch_size)
    test_loader = DataLoader(ParityDataset(batch_size * test_n_batches, n_elements), batch_size=batch_size)

    print("")
    accu = 0
    loss = 0
    for epoch in range(epochs):
        for idx, (X, y_grnd) in enumerate(train_loader):
            optim.zero_grad()

            X, y_grnd = X.to(device), y_grnd.to(device).to(torch.float)

            ps, ys, p_halt, y_halt = model(X)
            recon_loss = loss_rec(ps, ys, y_grnd)
            regul_loss = loss_reg(ps)
            loss = recon_loss + beta * regul_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
            optim.step()

            # TODO(Abid): Check this, calculates the expected number of steps taken
            steps = torch.arange(1, ps.shape[0] + 1, device=ps.device)
            expected_steps = (ps * steps[:, None]).sum(dim=0)

            y_discreet = y_halt > 0
            compare = (y_discreet == y_grnd).to(torch.int)
            accu = compare.sum() / compare.shape[0]

            # if idx % 200 == 0:
            #     print("")
            #     print(f"{y_halt[0].item() = }")
            #     print(f"y_grnd[0] = {y_grnd.to(torch.float)[0]}\n")

            message = (f"E: {epoch+1}",
                       f"\033[92mAccuracy: {(100.*accu):.3f}%\033[0m",
                       f"TotalLoss: {loss:.3f}",
                       f"RecLoss: {recon_loss:.3f}", f"RegLoss: {regul_loss:.3f}",
                       f"Avg Steps: {expected_steps.mean():.3f}",)
            print("\x1b[2K", end="")
            print(*message, sep=" | ", end="\r")

        print("")
    torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "loss": loss,
                "accuracy": accu},
                "model_checkpoint_2.pt")

if __name__ == "__main__":
    train()
