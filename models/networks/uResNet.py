import torch.nn as nn

class ResEle(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size) -> None:
        super().__init__()
        assert len(list(kernel_size)) == 2, "Module supports only 2D convs"
        self.main_conv = nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=kernel_size, padding='same')
        self.skip_conv =  nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=(1,1), padding='same')
        self.norm = nn.BatchNorm2d(num_features=out_dims)
        self.act = nn.ReLU()

    def forward(self, x):
        convs_out = self.main_conv(x) + self.skip_conv(x)
        norm_out = self.norm(convs_out)
        return self.act(norm_out)

class uResNet_Network(nn.Module):
    def __init__(self, in_dims=1, first_out_dims=32, num_down=3) -> None:
        super().__init__()

        self.ResEle_downs = nn.ModuleList([ResEle(in_dims=in_dims, out_dims=first_out_dims, kernel_size=(3,3))])
        self.ResEle_downs.extend(
            [ResEle(in_dims=first_out_dims*(2**i), out_dims=first_out_dims*(2**(i+1)), kernel_size=(3,3)) for i in range(num_down-1)]
            )
        self.pooling = nn.ModuleList([nn.MaxPool2d(kernel_size=(2,2), stride=2) for _ in range(num_down)])   
        
        self.bottleneck = nn.Sequential(
            ResEle(in_dims=first_out_dims*(2**(num_down-1)), out_dims=first_out_dims*(2**num_down), kernel_size=(3,3)),
            nn.ConvTranspose2d(in_channels=first_out_dims*(2**num_down), out_channels=first_out_dims*(2**(num_down-1)), kernel_size=(3,3), stride=(2,2), padding=1, output_padding=1),
            ResEle(in_dims=first_out_dims*(2**(num_down-1)), out_dims=first_out_dims*(2**(num_down-1)), kernel_size=(3,3))
        )

        # First block should be changed so that the ResElem is 128 --> 128 and the ConvTrans is 128 --> 64. Diagram in paper is wrong.
        self.up_path = nn.ModuleList([
            nn.Sequential(
                ResEle(in_dims=128, out_dims=128, kernel_size=(3,3)),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=1)
            ),
            nn.Sequential(
                ResEle(in_dims=64, out_dims=64, kernel_size=(3,3)),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2),  padding=1, output_padding=1)
            ),
            nn.Sequential(
                ResEle(in_dims=32, out_dims=32, kernel_size=(3,3)),
                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1,1))
            )
        ])

    def forward(self, x):
        residual_tensors = []

        # Down path
        for ResEle, pool in zip(self.ResEle_downs, self.pooling):
            x = ResEle(x)
            residual_tensors.append(x)
            x = pool(x)
        # Bottleneck
        x = self.bottleneck(x)

        # Up path
        for skip, up_block in zip(residual_tensors[-1::-1], self.up_path):
            x = up_block(skip + x)
        
        return x