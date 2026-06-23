import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        self.pool  = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x)

class Cnn14FineTuned(nn.Module):
    def __init__(self, num_classes=3, pretrained_path=None):
        super().__init__()
        self.block1 = ConvBlock(1,    64)
        self.block2 = ConvBlock(64,   128)
        self.block3 = ConvBlock(128,  256)
        self.block4 = ConvBlock(256,  512)
        self.block5 = ConvBlock(512,  1024)
        self.block6 = ConvBlock(1024, 2048)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc1    = nn.Linear(2048, 2048)
        self.relu   = nn.ReLU()
        self.dropout= nn.Dropout(0.5)
        self.fc_out = nn.Linear(2048, num_classes)

        if pretrained_path:
            self._load_pretrained(pretrained_path)
            self._freeze_backbone()

    def _load_pretrained(self, path):
        checkpoint  = torch.load(path, map_location="cpu")
        model_state = checkpoint.get("model", checkpoint)
        own_state   = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in model_state.items():
            if "bn0" in name or "fc_out" in name:
                skipped += 1
                continue
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own_state)
        print(f"  ✅ Pretrained weights loaded: {loaded} layers, {skipped} skipped")

    def _freeze_backbone(self):
        for name, param in self.named_parameters():
            if "fc_out" not in name:
                param.requires_grad = False
        print(f"  🔒 Backbone frozen — only fc_out trains")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("  🔓 All layers unfrozen for full fine-tuning")

    def forward(self, x):
        # x: (B, 1, 64, 101) — already correct shape
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)   # (B, 2048)
        x = self.relu(self.fc1(self.dropout(x)))
        return self.fc_out(self.dropout(x))
