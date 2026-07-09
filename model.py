# model.py - PANN (CNN14) adapted for binary wheeze detection
# Supports locally downloaded pretrained weights (no internet download needed)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.ones_(self.bn1.weight);  nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight);  nn.init.zeros_(self.bn2.bias)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        return x


# ─── CNN14 Backbone ───────────────────────────────────────────────────────────

class CNN14(nn.Module):
    """
    CNN14 from 'PANNs: Large-Scale Pretrained Audio Neural Networks'
    (Kong et al., 2020).

    Input shape adapted for 4000 Hz / 15s / 64-mel:
      (B, 1, 64, ~1875)  log-mel spectrogram
    Output: (B, 2048) embedding
    """
    def __init__(self, n_mels: int = config.N_MELS):
        super().__init__()
        self.bn0   = nn.BatchNorm2d(n_mels)

        self.conv1 = ConvBlock(1,    64)
        self.conv2 = ConvBlock(64,   128)
        self.conv3 = ConvBlock(128,  256)
        self.conv4 = ConvBlock(256,  512)
        self.conv5 = ConvBlock(512,  1024)
        self.conv6 = ConvBlock(1024, 2048)

        self.fc   = nn.Linear(2048, 2048, bias=True)
        self.drop = nn.Dropout(0.2)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x: (B, 1, Mel, T)
        x = x.transpose(1, 2)      # (B, Mel, 1, T) — BN over mel axis
        x = self.bn0(x)
        x = x.transpose(1, 2)      # back to (B, 1, Mel, T)

        x = self.conv1(x, (2, 2), "avg+max");  x = self.drop(x)
        x = self.conv2(x, (2, 2), "avg+max");  x = self.drop(x)
        x = self.conv3(x, (2, 2), "avg+max");  x = self.drop(x)
        x = self.conv4(x, (2, 2), "avg+max");  x = self.drop(x)
        x = self.conv5(x, (2, 2), "avg+max");  x = self.drop(x)
        x = self.conv6(x, (1, 1), "avg+max");  x = self.drop(x)

        # Global avg+max pooling → (B, 2048)
        x = x.mean(dim=[2, 3]) + x.amax(dim=[2, 3])
        x = F.relu_(self.fc(x))
        x = self.drop(x)
        return x    # (B, 2048)


# ─── Full Wheeze-Detection Model ──────────────────────────────────────────────

class WheezeDetector(nn.Module):
    """
    CNN14 backbone + binary classification head.
    Returns (logit, probability) for the wheeze class.
    """
    def __init__(self, pretrained: bool = config.PRETRAINED,
                 n_mels: int = config.N_MELS):
        super().__init__()
        self.backbone = CNN14(n_mels=n_mels)
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.xavier_uniform_(self.head[3].weight)

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        """
        Load locally downloaded AudioSet CNN14 pretrained weights.
        Set PRETRAINED_PATH in config.py to your local .pth file.
        No internet download is attempted.
        """
        path = config.PRETRAINED_PATH

        if not os.path.exists(path):
            print(f"  [WARNING] Pretrained file not found at: {path}")
            print("  Update PRETRAINED_PATH in config.py to your local .pth file.")
            print("  Training from scratch instead.")
            return

        print(f"  Loading pretrained weights from: {path}")
        try:
            ckpt  = torch.load(path, map_location="cpu")
            state = ckpt.get("model", ckpt)

            # Filter out front-end layers (spectrogram/logmel extractor)
            # that exist in the original PANN but not in our pipeline
            filtered = {
                k: v for k, v in state.items()
                if not k.startswith("spectrogram_extractor")
                and not k.startswith("logmel_extractor")
            }

            missing, unexpected = self.backbone.load_state_dict(
                filtered, strict=False
            )
            loaded = len(filtered) - len(unexpected)
            print(f"  Pretrained weights loaded: {loaded} layers matched")
            if missing:
                print(f"  Missing (random init): {len(missing)} layers")
            if unexpected:
                print(f"  Ignored (not in model): {len(unexpected)} layers")

        except Exception as e:
            print(f"  [WARNING] Could not load pretrained weights: {e}")
            print("  Training from scratch instead.")

    def forward(self, x):
        emb   = self.backbone(x)             # (B, 2048)
        logit = self.head(emb).squeeze(-1)   # (B,)
        prob  = torch.sigmoid(logit)
        return logit, prob


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> WheezeDetector:
    model = WheezeDetector(pretrained=config.PRETRAINED, n_mels=config.N_MELS)
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel  : WheezeDetector (CNN14 backbone)")
    print(f"  N_MELS           : {config.N_MELS}")
    print(f"  Sample Rate      : {config.SAMPLE_RATE} Hz")
    print(f"  Clip Duration    : {config.CLIP_DURATION}s  ({config.CLIP_SAMPLES} samples)")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    return model