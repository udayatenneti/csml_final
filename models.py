import torch
import torch.nn.functional as F

class CNNClassifierOLD(torch.nn.Module):
    def __init__(self, layers=[10,20,30], n_input_channels=3, n_output_channels=2, kernel_size=5):
        super().__init__()

        # linear_layers = []
        #
        # c = n_input_channels
        # for l in layers:
        #     linear_layers.append(torch.nn.Conv2d(c, l, kernel_size, stride=1, padding=kernel_size//2))
        #     linear_layers.append(torch.nn.BatchNorm2d(l)) #TODO: affine=False ?
        #     linear_layers.append(torch.nn.ReLU())
        #     c = l
        # linear_layers.append(torch.nn.MaxPool2d(kernel_size))
        # self.network = torch.nn.Sequential(*linear_layers)
        #
        # output_linear_layers = []
        # output_layers = layers[::-1]
        # output_layers.append(n_output_channels)
        # for i in range(len(output_layers) - 1):
        #     output_linear_layers.append( torch.nn.Linear(output_layers[i], output_layers[i+1]) )
        #     output_linear_layers.append(torch.nn.ReLU())
        # self.classifier = torch.nn.Sequential(*output_linear_layers)
        self.log_softmax = torch.nn.LogSoftmax()#dim=2)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(12)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv4 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(24)
        self.conv5 = torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(24)
        #self.flatten = torch.nn.flatten() #TODO: ADD IN!
        self.fc1 = torch.nn.Linear(24 * 354 * 634, n_output_channels)

        #flatten potentially!

    def forward(self, x):

        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 354 * 634)
        output = self.fc1(output)
        return self.log_softmax(output)
        #return self.log_softmax(self.classifier(self.network(x))) #.mean(dim=[2, 3])))

class CNNClassifier(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_output_channels=2, kernel_size=5):
        super().__init__()

        self.log_softmax = torch.nn.LogSoftmax()  # dim=2

        self.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=10, kernel_size=5, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=30, kernel_size=5, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(30)
        self.pool = torch.nn.MaxPool2d(2, 2)
        # self.flatten = torch.nn.flatten()  # TODO: ADD IN!
        self.fc1 = torch.nn.Linear(30 * 357 * 637, 20)
        self.fc2 = torch.nn.Linear(20, 10)
        self.fc3 = torch.nn.Linear(10, n_output_channels)

        #flatten potentially!

    def forward(self, x):

        output = F.relu(self.bn1(self.conv1(x)))
        output = self.pool(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn3(self.conv3(output)))
        output = self.pool(output)
        output = output.view(-1, 30 * 357 * 637)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        return self.log_softmax(output)


def save_model(model, name=None):
    from torch import save
    from os import path
    if name is None:
        name = 'cnn.th'
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(name=None):
    from torch import load
    from os import path
    r = CNNClassifier()
    if name is None:
        name = 'cnn.th'
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r
