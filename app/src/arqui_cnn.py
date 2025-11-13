import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# ARQUITECTURA DEL MODELO 1.0
# ==============================================================================


class BaseModel(nn.Module):
    """
    Modelo base con arquitectura fully connected de 2 capas.
    Para clasificación de imágenes en 10 clases usando datos de CIFAR10
    """

    def __init__(self):
        super(BaseModel, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 10)
        )
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        """Forward pass"""
        return self.model(inputs)

    def predict(self, inputs):
        """Predicción con softmax"""
        return self.final_activation(self.model(inputs))


# ==============================================================================


# ==============================================================================
# OPCIÓN 1: CNN Simple
# ==============================================================================
class SimpleCNN(nn.Module):
    """
    CNN básica con 3 bloques convolucionales

    Arquitectura:
    - 3 bloques Conv -> ReLU -> MaxPool
    - 2 capas fully connected
    - Dropout para regularización

    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Bloque 1: 3 -> 32 canales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Bloque 2: 32 -> 64 canales
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Bloque 3: 64 -> 128 canales
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4

        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        # Regularización
        self.dropout = nn.Dropout(0.5)

        # Activación final
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # Bloque 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Bloque 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Bloque 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(-1, 128 * 4 * 4)

        # FC con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        return self.final_activation(self.forward(x))


# ==============================================================================
# OPCIÓN 2: CNN Mejorada con Batch Normalization
# ==============================================================================
class ImprovedCNN(nn.Module):
    """
    CNN con Batch Normalization y arquitectura más profunda

    Arquitectura:
    - 4 bloques Conv -> BatchNorm -> ReLU -> MaxPool
    - 2 capas fully connected con BatchNorm
    - Dropout para regularización

    Parámetros: ~340K
    Accuracy esperado: ~75-80%

    """

    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Bloque 1: 3 -> 64 canales
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Bloque 2: 64 -> 128 canales
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Bloque 3: 128 -> 256 canales
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Bloque 4: 256 -> 256 canales
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Bloque 5: 256 -> 512 canales
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4

        # Fully connected con BatchNorm
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

        # Regularización
        self.dropout = nn.Dropout(0.5)

        # Activación final
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # Bloque 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Bloque 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # Bloque 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Bloque 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout(x)

        # Bloque 5
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(-1, 512 * 4 * 4)

        # FC con BatchNorm y Dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        return self.final_activation(self.forward(x))


# ==============================================================================
# OPCIÓN 3: ResNet-like con Skip Connections
# ==============================================================================
class ResidualBlock(nn.Module):
    """Bloque residual básico con skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection con ajuste de dimensión
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection
        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet adaptado para CIFAR-10 con skip connections

    Arquitectura:
    - Capa inicial convolucional
    - 3 grupos de bloques residuales
    - Global Average Pooling
    - Fully connected final

    """

    def __init__(self, num_blocks=[2, 2, 2]):
        super(ResNetCIFAR, self).__init__()

        # Capa inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Grupos de bloques residuales
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)

        # Clasificador
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

        # Activación final
        self.final_activation = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # Primer bloque puede cambiar dimensiones
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Bloques subsecuentes mantienen dimensiones
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Capa inicial
        x = F.relu(self.bn1(self.conv1(x)))

        # Bloques residuales
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global Average Pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        # Clasificador
        x = self.fc(x)

        return x

    def predict(self, x):
        return self.final_activation(self.forward(x))
