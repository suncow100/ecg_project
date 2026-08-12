from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """
    1D Residual Block

    Conv1D -> BN -> ReLU -> Conv1D -> BN
                         |
                    Residual Add
                         |
                       ReLU

    stride > 1 또는 channel 변경 시 shortcut projection 사용
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)


        # shortcut path
        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        else:
            self.shortcut = nn.Identity()



    def forward(self, x: torch.Tensor):

        identity = self.shortcut(x)


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)


        out = out + identity

        out = self.relu(out)

        return out



class ResNet1D(nn.Module):
    """
    1D ResNet for ECG beat classification

    Input:
        (batch, 1, 250)

    Architecture:

    Stem
      Conv1D(1 -> 32)

    Stage1
      32 channels

    Stage2
      64 channels + downsampling

    Stage3
      128 channels + downsampling

    Stage4
      256 channels + downsampling

    GAP

    Dropout

    Fully Connected

    Output:
        N/S/V/F (4 classes)
    """


    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()



        # Initial feature extraction
        self.stem = nn.Sequential(

            nn.Conv1d(
                in_channels,
                32,
                kernel_size=15,
                stride=1,
                padding=7,
                bias=False,
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(inplace=True),

        )



        # Residual stages

        self.stage1 = nn.Sequential(

            ResBlock1D(
                32,
                32,
                stride=1
            ),

            ResBlock1D(
                32,
                32,
                stride=1
            ),

        )



        self.stage2 = nn.Sequential(

            ResBlock1D(
                32,
                64,
                stride=2
            ),

            ResBlock1D(
                64,
                64,
                stride=1
            ),

        )



        self.stage3 = nn.Sequential(

            ResBlock1D(
                64,
                128,
                stride=2
            ),

            ResBlock1D(
                128,
                128,
                stride=1
            ),

        )



        self.stage4 = nn.Sequential(

            ResBlock1D(
                128,
                256,
                stride=2
            ),

            ResBlock1D(
                256,
                256,
                stride=1
            ),

        )



        # Classification head

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.fc = nn.Linear(
            256,
            num_classes
        )



    def forward(
        self,
        x: torch.Tensor
    ):

        # x:
        # (batch,1,250)

        x = self.stem(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)


        # (batch,256,1)

        x = self.gap(x)


        # (batch,256)

        x = torch.flatten(
            x,
            start_dim=1
        )


        x = self.dropout(x)


        logits = self.fc(x)


        return logits




if __name__ == "__main__":


    model = ResNet1D(
        num_classes=4,
        dropout=0.3
    )


    dummy = torch.randn(
        8,
        1,
        250
    )


    output = model(dummy)


    assert output.shape == (
        8,
        4
    ), output.shape



    total_params = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        f"output shape: {tuple(output.shape)}"
    )

    print(
        f"total params: {total_params:,}"
    )

    print(
        f"trainable params: {trainable_params:,}"
    )