import torch
from torch import nn, functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=(3,3), stride=(1,1), padding=(1, 1))
        self.conv2 = nn.Conv2d(4, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(3, 4, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(4, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

net = Net()
net.eval()
x = torch.normal(0, 1, size=(1,3,720,720), dtype=torch.float32)
y = net(x)
print(y.shape)

#  ``dynamic_axes = {'input_1':{0:'batch',
#                              1:'width',
#                              2:'height'},
#                   'input_2':{0:'batch'},
#                   'output':{0:'batch',
#                             1:'detections'}}``
torch.onnx.export(net, args=(x,), input_names=('x',), output_names=('y',), dynamic_axes={'x': {0: 'b'}}, f="fixed_batch.onnx")
torch.onnx.export(net, args=(x,), input_names=('x',), output_names=('y',), dynamic_axes={'x': {0: 'b', 2: 'h'}}, f="dyn_batch.onnx")
