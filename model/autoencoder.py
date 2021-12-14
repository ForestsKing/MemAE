from torch import nn

from model.memorymodule import MemModule


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=(15,), padding=(7,)),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=(5,), padding=(2,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.mem = MemModule(mem_dim=64, fea_dim=32)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=2, out_channels=4, kernel_size=(5,), padding=(2,)),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=4, out_channels=8, kernel_size=(15,), padding=(7,)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, X):
        X = X.permute(0, 2, 1)
        Z = self.encoder(X)
        ZZ, att = self.mem(Z)
        out = self.decoder(ZZ)
        out = out.permute(0, 2, 1)
        return out, att
