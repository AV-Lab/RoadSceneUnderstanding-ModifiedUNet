import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF


'''
# Double Convolution X -> y ... y -> y
'''

class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DoubleConv, self).__init__()

		self.conv_3 = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias=False), # False to use batchnorm
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias=False), # False to use batchnorm
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
			)

		self.conv_5 = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size= 5, stride= 1, padding= 2, bias=False), # False to use batchnorm
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels, kernel_size= 5, stride= 1, padding= 2, bias=False), # False to use batchnorm
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
			)

	def forward(self, x):
		return torch.cat((self.conv_3(x), self.conv_5(x)), dim=1)

class UNET(nn.Module):

	def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]):
		super(UNET, self).__init__()
		self.ups = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

		# Down Part
		for feature in features:

			self.downs.append(DoubleConv(in_channels, feature))
			in_channels = feature * 2

		# Up Part
		for feature in reversed(features):
			self.ups.append(
					nn.ConvTranspose2d(
							feature *2 * 2, feature * 2, kernel_size = 2, stride = 2
						)
				)

			self.ups.append(
					DoubleConv(feature * 2 * 2, feature)
				)

		# Middle
		self.bottleneck = DoubleConv(features[-1] * 2, features[-1] *2 )
		self.final_conv = nn.Conv2d(features[0]  * 2, out_channels, kernel_size = 1)


	def forward(self, x):
		skip_connections = []

		for down in self.downs:
			x = down(x)
			skip_connections.append(x)
			x = self.pool(x)

		x = self.bottleneck(x)

		skip_connections = skip_connections[::-1] #reverse

		for idx in range(0, len(self.ups), 2):
			x = self.ups[idx](x)
			skip_connection =skip_connections[idx//2]

			if x.shape != skip_connection.shape:
				x = TF.resize(x, size=skip_connection.shape[2:])

			concat_skip = torch.cat((skip_connection, x), dim = 1)
			x = self.ups[idx + 1](concat_skip)
		return self.final_conv(x)


# Testing
def test():
	x = torch.randn((3,1,160, 160))
	model = UNET(in_channels = 1, out_channels = 1)
	preds = model(x)
	
	print(preds.shape)
	print(x.shape)
	
	assert preds.shape == x.shape
