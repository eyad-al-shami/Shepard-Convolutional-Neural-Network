import torch
import torch.nn as nn
import torch.nn.functional as F

class ShConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, initial_weight=True, threshold = 0.1):
        super(ShConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.kernels = nn.ParameterList()
        for i in range(out_channels):
            self.kernels.append(nn.Parameter(torch.Tensor(in_channels, 1, kernel_size, kernel_size)))

        if initial_weight:
            self.__init_weights()

    def __init_weights(self):
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel)
        nn.init.constant_(self.bias, 0)
    
    def forward(self, masks, x):
        
        # get the device of one of the kernels
        kernel_device = self.kernels[0].device

        # defining the final output
        output_features_map = torch.Tensor().to(kernel_device)
        output_mask = torch.Tensor().to(kernel_device)

        # computing the output for the kernels
        for i, kernel in enumerate(self.kernels):
            intermediate_features_maps = F.conv2d(x, kernel, padding=self.padding, stride=self.stride, groups=x.size(1))
            intermediate_masks = F.conv2d(masks, kernel, padding=self.padding, stride=self.stride, groups=x.size(1))
            intermediate_features_maps = intermediate_features_maps / (intermediate_masks + 1e-8)
            feature_map = intermediate_features_maps.sum(dim=1, keepdim=True) + self.bias[i]
            mask = intermediate_masks.sum(dim=1, keepdim=True)
            output_features_map = torch.cat((output_features_map, feature_map), dim=1)
            output_mask = torch.cat((output_mask, mask), dim=1)
            # thresholding
            with torch.no_grad():
                output_mask[output_mask >= self.threshold] = 1
                output_mask[output_mask < self.threshold] = 0
        
        return output_features_map, output_mask


# if __name__ == "__main__":
#     batch = 13
#     in_channels = 8
#     out_channels = 512
#     kernel_size = 5
#     stride = 1
#     padding = 1
#     # TODO: accept both int and string for padding
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     shconv = ShConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#     shconv.to(device)
    
#     masks = torch.randn(batch, in_channels, 32, 32)
#     x = torch.randn(batch, in_channels, 32, 32)
#     x, masks = x.to(device), masks.to(device)
#     output_features_map, output_mask = shconv(masks, x)
#     print(output_features_map.shape)
#     print(output_mask.shape)

