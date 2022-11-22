import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ELU = nn.ELU()  # Fonction d'activation ReLU. Il n'y a pas de poids dans cette fonction donc on peut la définir une seule fois.

        self.conv_layer1 = nn.Conv2d(1, 64, 3)  # Couche avec 3 entrées, 6 sorties (Taille des channels) et de kernels 3*3
        torch.nn.init.xavier_uniform(self.conv_layer1.weight)
        self.conv_Layer2 = nn.Conv2d(64, 64, 3)
        torch.nn.init.xavier_uniform(self.conv_layer2.weight)
        self.pool_layer3 = nn.MaxPool2d(2,stride=2)
        torch.nn.init.xavier_uniform(self.pool_layer3.weight)
        self.conv_layer4 = nn.Conv2d(64, 128, 3)
        torch.nn.init.xavier_uniform(self.conv_layer4.weight)
        self.conv_layer5 = nn.Conv2d(128, 128, 3)
        torch.nn.init.xavier_uniform(self.conv_layer5.weight)
        self.pool_layer6 = nn.MaxPool2d(2, stride=2)
        torch.nn.init.xavier_uniform(self.pool_layer6.weight)
        self.conv_layer7 = nn.Conv2d(128 , 256, 3)
        torch.nn.init.xavier_uniform(self.conv_layer7.weight)
        self.conv_layer8 = nn.Conv2d(256, 256, 3)
        torch.nn.init.xavier_uniform(self.conv_layer8.weight)
        self.pool_layer9 = nn.MaxPool2d(2, stride=2)
        torch.nn.init.xavier_uniform(self.pool_layer9.weight)
        self.fc1 = nn.Linear(2048, 7)
        self.batch_normalization = nn.BatchNorm2d()

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.conv_layer2(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.pool_layer3(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.conv_layer4(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.conv_layer5(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.pool_layer6(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.conv_layer7(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.conv_layer8(x)
        x = self.ELU(x)
        x = self.batch_normalization(x)
        x = self.pool_layer9(x)
        # Avec view, on met le feature map de dimension N (taille du batch), C (nb channels), H (hauteur), L(largeur)  sous la dimension N, C*H*L, et donc sous forme de vecteur
        # Deux solutions, soit on calcule C*H*W en connaissant les dimensions, soit on utilise -1 qui va faire ce calcul et conserver N car on a mis 1 pour cette dimension

        # Flatten : Comme dans mon exemple de correction la taille du batch est de 10, plusieurs solutions s'offrent à nous pour le .view()
        x = x.view(-1, 2048)  # On indique directement les dimensions de l'output
        # x = x.view(10, 4056) # On indique que la taille du batch est de 10 et la taille du vecteur de caractéristiques (CxHxW)
        # x = x.view(-1, -1) # Ne fonctionnera pas, car le modèle ne peut pas deviner deux dimensions à configurer automatiquement
        x = self.fc1(x)
        return x