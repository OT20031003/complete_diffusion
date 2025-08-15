import torch
from embedingver import SinusoidalPositionEmbeddings
class MnistUNet(torch.nn.Module):
    def __init__(self, inp_channel=1, time_embedding_dim=128, unet_dim = 64):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inp_channel + time_embedding_dim, out_channels=unet_dim, kernel_size=3, stride=1, padding=1)
        self.act1 = torch.nn.ReLU()
        self.si = SinusoidalPositionEmbeddings(time_embedding_dim)

        self.conv2 = torch.nn.Conv2d(in_channels=unet_dim, out_channels=unet_dim, kernel_size=3, padding=1, stride=1)
        self.act2 = torch.nn.ReLU()
        self.maxpooling1 = torch.nn.MaxPool2d(2, stride=2)
        
        self.conv3 = torch.nn.Conv2d(in_channels=unet_dim, out_channels=2*unet_dim, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=2*unet_dim, out_channels=2*unet_dim, kernel_size=3, padding=1)
        self.maxpooling2 = torch.nn.MaxPool2d(2, stride=2)
        self.act3 = torch.nn.ReLU()
        self.act4 = torch.nn.ReLU()

        self.conv5 = torch.nn.Conv2d(in_channels=2*unet_dim, out_channels=4*unet_dim, kernel_size=3, padding=1)
        self.act5 = torch.nn.ReLU()
        
        self.up1 = torch.nn.Upsample(scale_factor=2)
        self.conv6 = torch.nn.Conv2d(in_channels=6*unet_dim, out_channels=2*unet_dim, kernel_size=3, padding=1)
        self.act6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(in_channels=2*unet_dim, out_channels=2*unet_dim, kernel_size=3, padding=1)
        self.act7 = torch.nn.ReLU()
        self.up2 = torch.nn.Upsample(scale_factor=2)
        self.conv8 = torch.nn.Conv2d(in_channels=3*unet_dim, out_channels=unet_dim, kernel_size=3, padding=1)
        self.act8 = torch.nn.ReLU()
        self.conv9 = torch.nn.Conv2d(in_channels=unet_dim, out_channels=unet_dim, kernel_size=3, padding=1)
        self.act9 = torch.nn.ReLU()

        #出力
        self.conv10 = torch.nn.Conv2d(in_channels=unet_dim, out_channels=1, kernel_size=1)
        

    def forward(self, x, timestep:torch.tensor):
        t = self.si(timestep) # t.shape = [batch, 128]
        t = t.unsqueeze(-1).unsqueeze(-1)
        t = torch.nn.functional.interpolate(t, size=x.shape[2:])
        # タイムステップの結合
        x = torch.cat((x, t), dim = 1) #(16,129,32,32)
        #print(f"x.shape = {x.shape}")
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        h1 = x #(16, 64, 32, 32)
        #print(f"x.shape = {x.shape}") 
        x = self.maxpooling1(x) #(16, 64, 16, 16)
        #print(f"x.shape = {x.shape}")
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        h2 = x #(16, 128, 16, 16)
        #print(f"h2.shape = {x.shape}")
        x = self.maxpooling2(x)
        #bottom
        x = self.conv5(x)
        x = self.act5(x) #(16, 256, 8, 8)
        #print(f"x.shape = {x.shape}") 
        x = self.up1(x) #(16, 256, 16, 16)
        #print(f"xup.shape = {x.shape}") 
        x = torch.cat((x, h2), dim=1)
        #print(f"x6.shape = {x.shape}") 
        x = self.conv6(x)
        x = self.act6(x)
        x = self.conv7(x)
        x = self.act7(x)
        x = self.up2(x)
        x = torch.cat((x, h1),dim=1)
        x = self.conv8(x)
        x = self.act8(x)
        x = self.conv9(x)
        x = self.act9(x)

        x = self.conv10(x)
        return x




        

