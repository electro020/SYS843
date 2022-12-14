#DISCLAIMER this code is inspired from this repository :https://github.com/ankur219/ECG-Arrhythmia-classification


from __future__ import division, print_function
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import torch.optim as optim
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    from torch import optim
    import torch.nn as nn
    from tqdm.auto import tqdm
    from torchvision import models
    import gc
    from collections import Counter

    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import time


    y_pred = []
    y_true = []

    #import normal_preprocessing as NO
    #import cropping as crop
    #import left_bundle_preprocessing as LB
    #import right_bundle_preprocessing as RB
    #import atrial_premature_preprocessing as A
    #import ventricular_escape_beat_preprocessing as E
    #import paced_beat_preprocessing as PB
    #import ventricular_premature_contraction_preprocessing as V
    #import create_dataset

    #########################################################
    #Image Generation
    #########################################################
    #NO.normal_image_generation()
    #LB.left_bundle_image_generation()
    #RB.right_bundle_image_generation()
    #A.atrial_premature_image_generation()
    #E.ventricular_escape_image_generation()
    #PB.paced_beat_image_generation()
    #V.ventricular_premature_image_generation()

    #########################################################
    #Data augmentation
    #########################################################
    #crop.directory_selection_cropping('images_normal')
    #crop.directory_selection_cropping('images_left_bundle')
    #crop.directory_selection_cropping('images_right_bundle')
    #crop.directory_selection_cropping('images_atrial_premature')
    #crop.directory_selection_cropping('images_ventricular_escape')
    #crop.directory_selection_cropping('images_paced_beat')
    #crop.directory_selection_cropping('images_ventricular_premature')
    #########################################################
    #Datase creation
    #########################################################

    eval_losses = []
    eval_accu = []

    #we clear the GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    classes =('APC', 'LBB', 'NOR', 'PAB', 'RBB', 'VEB', 'PVB')

    transform = transforms.Compose(
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset=torchvision.datasets.ImageFolder(root="/home/ens/AP69690/SYS843/database",transform=transform)

    #D:\geordi\ecole\Automn2022\SYS843\pycharmprojects\database
    #/home/ens/AP69690/SYS843/database
    print(dataset.class_to_idx)
    print(dict(Counter(dataset.targets)))
    Confusion_matrix = np.zeros((7,7))
    N = len(dataset)
    print(N)
    # generate & shuffle indices
    indices = np.arange(N)
    indices = np.random.permutation(indices)

    # select train/test, for demo I am using 80,20 trains/test
    #train_indices = indices[:int(0.01 * N)]
    #test_indices = indices[int(0.01 * N):int(0.02*N)]

    train_indices = indices[:int(0.8 * N)]
    train_indices_acc = indices[:int(0.8 * N)]
    validation_indices = indices[int(0.8 * N):int(N)]
    #test_indices = indices[int(0.9 * N):int(N)]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, validation_indices)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=8)
    trainloader_acc = torch.utils.data.DataLoader(train_indices_acc, batch_size=1, shuffle=True, num_workers=8)
    validationloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    class HeartNet(nn.Module):
        def __init__(self, num_classes=7):
            super(HeartNet, self).__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64, eps=0.001),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64, eps=0.001),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(128, eps=0.001),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(128, eps=0.001),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(256, eps=0.001),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(256, eps=0.001),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.classifier = nn.Sequential(
                nn.Linear(16 * 16 * 256, 2048),
                nn.ELU(inplace=True),
                nn.BatchNorm1d(2048, eps=0.001),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 16 * 16 * 256)
            x = self.classifier(x)
            return x

    net = HeartNet()

    # On utilise la descente de gradient stochastique comme optimiseur. D'autres m??thodes sont existante mais celle-ci reste tr??s utilis??e.
    optimizer = optim.Adam(net.parameters(),lr=0.0001)

    torch.cuda.empty_cache()
    gpu = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Model et optimizer d??j?? d??finis dans les questions pr??c??dentes
    criterion = nn.CrossEntropyLoss() # Fonction de co??t qui permettra le calcul de l'erreur
    net.to(gpu)
    for epoch in range(10): # loop sur le dataset 5 fois
      running_loss = 0.0
      net.train()
      for i, data in tqdm(enumerate(trainloader, 0)): # En mettant data de cette fa??on, data est un tuple tel que data = (image, label)

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(gpu), data[1].to(gpu)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # Forward propagation (On passe notre input au mod??le ?? travers les diff??rentes couches)
        loss = criterion(outputs, labels) # Calcul de l'erreur
        loss.backward() # On r??tropropage l'erreur dans le r??seau, donc on calcul le gradient de l'erreur pour chaque param??tre
        optimizer.step() # On actualise les poids en fonction des gradients

        # print statistics
        running_loss += loss.item() # .item() retourne la valeur dans le tenseur et non le tenseur lui m??me
        if i % 5000 == 4999: # print every 1000 mini-batche

            with open('loss/loss.txt', 'a') as the_file:
                the_file.write(str((running_loss/ 5000))+'\n')
            print(f"[epoch {epoch + 1}, batch {i+1}/{int(len(dataset.targets)/10*0.8)}], loss : {running_loss / 5000}")
            running_loss = 0.0
            ##############################################################################################################################
            ###############################################ACCURACY EVALUATION############################################################
            ##############################################################################################################################
            net.eval()
            correct = 0
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            Confusion_matrix = Confusion_matrix * 0
            print("******************************************************************")
            print("****************************EVALUATION****************************")
            print("******************************************************************")
            for increment_val, data_val in tqdm(enumerate(validationloader, 0)):
                inputs, labels = data_val[0].to(gpu), data_val[1].to(gpu)
                outputs = net(inputs)
                pred = outputs.argmax()  # mon_tenseur.argmax() donne l'index de l'??l??ment le plus ??lev?? de l'output, et donc on r??cup??re la classe pr??dite par notre algo
                # mon_tenseur.argmax(-1) donnera le m??me r??sultat
                if pred == labels:  # On est pas oblig?? de sortir la donn??e via pred[0] et labels[0] car il n'y a qu'une valeur dans le tenseur, mais on peut, les deux reviennent au m??me
                    correct += 1
                if labels == 0:  # Atrial_premature
                    Confusion_matrix[0][pred] += 1
                if labels == 1:  # Left_bundle
                    Confusion_matrix[1][pred] += 1
                if labels == 2:  # Normal
                    Confusion_matrix[2][pred] += 1
                if labels == 3:  # Paced_beat
                    Confusion_matrix[3][pred] += 1
                if labels == 4:  # Right_bundle
                    Confusion_matrix[4][pred] += 1
                if labels == 5:  # Ventricular_escape
                    Confusion_matrix[5][pred] += 1
                if labels == 6:  # Ventricular_premature
                    Confusion_matrix[6][pred] += 1
                if ((labels == 0) and (pred==0)) or ((labels == 1) and (pred==1)) or ((labels == 3) and (pred==3)) or ((labels == 4) and (pred==4)) or ((labels == 5) and (pred==5)) or ((labels == 6) and (pred==6)):
                    true_positive += 1
                if ((labels == 2) and (pred==2)) :
                    true_negative += 1
                if ((labels == 0) and (pred!=0)) or ((labels == 1) and (pred!=1)) or ((labels == 3) and (pred!=3)) or ((labels == 4) and (pred!=4)) or ((labels == 5) and (pred!=5)) or ((labels == 6) and (pred!=6)):
                    false_positive += 1
                if ((labels == 2) and (pred != 2)):
                    false_negative += 1

                output_trans = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                y_pred.extend(output_trans)  # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)  # Save Truth
            print(f"Epoch : {epoch + 1} - Taux de classification = {correct / len(validationloader)}")
            print("true positive : " + str(true_positive))
            print("true negative : " + str(true_negative))
            print("false positive : " + str(false_positive))
            print("false negative : " + str(false_negative))
            print(Confusion_matrix.astype(int))
            with open('sensitivity/sens.txt', 'a') as the_file:
                the_file.write(str((Confusion_matrix[0][0]/(Confusion_matrix[0][0]+Confusion_matrix[1][0]+Confusion_matrix[2][0]+Confusion_matrix[3][0]+Confusion_matrix[4][0]+Confusion_matrix[5][0]+Confusion_matrix[6][0])))+",")
                the_file.write(str((Confusion_matrix[1][1]/(Confusion_matrix[0][1]+Confusion_matrix[1][1]+Confusion_matrix[2][1]+Confusion_matrix[3][1]+Confusion_matrix[4][1]+Confusion_matrix[5][1]+Confusion_matrix[6][1])))+",")
                the_file.write(str((Confusion_matrix[2][2]/(Confusion_matrix[0][2]+Confusion_matrix[1][2]+Confusion_matrix[2][2]+Confusion_matrix[3][2]+Confusion_matrix[4][2]+Confusion_matrix[5][2]+Confusion_matrix[6][2])))+",")
                the_file.write(str((Confusion_matrix[3][3]/(Confusion_matrix[0][3]+Confusion_matrix[1][3]+Confusion_matrix[2][3]+Confusion_matrix[3][3]+Confusion_matrix[4][3]+Confusion_matrix[5][3]+Confusion_matrix[6][3])))+",")
                the_file.write(str((Confusion_matrix[4][4]/(Confusion_matrix[0][4]+Confusion_matrix[1][4]+Confusion_matrix[2][4]+Confusion_matrix[3][4]+Confusion_matrix[4][4]+Confusion_matrix[5][4]+Confusion_matrix[6][4])))+",")
                the_file.write(str((Confusion_matrix[5][5]/(Confusion_matrix[0][5]+Confusion_matrix[1][5]+Confusion_matrix[2][5]+Confusion_matrix[3][5]+Confusion_matrix[4][5]+Confusion_matrix[5][5]+Confusion_matrix[6][5])))+'\n')
                the_file.write(str((Confusion_matrix[6][6] / (
                            Confusion_matrix[0][6] + Confusion_matrix[1][6] + Confusion_matrix[2][6] +
                            Confusion_matrix[3][6] + Confusion_matrix[4][6] + Confusion_matrix[5][6] +
                            Confusion_matrix[6][6]))) + '\n')

            ###################################################################################'\n'
            cf_matrix = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 7, index=[i for i in classes],
                                 columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            plt.savefig('outputs/output_val_' + current_time + '_epoch_' + str(epoch) + '.png')
            ##################################################################################
            print("******************************************************************")
            plt.close('all')
            ##############################################################################################################################
            #################################################ACCURACY TRAINING############################################################
            ##############################################################################################################################
            correct = 0

            Confusion_matrix = Confusion_matrix * 0
            """
            print("******************************************************************")
            print("****************************TRAINING******************************")
            print("******************************************************************")
            for increment_train, data_train in tqdm(enumerate(trainloader_acc, 0)):
                inputs, labels = data_train[0].to(gpu), data_train[1].to(gpu)
                outputs = net(inputs)
                pred = outputs.argmax()  # mon_tenseur.argmax() donne l'index de l'??l??ment le plus ??lev?? de l'output, et donc on r??cup??re la classe pr??dite par notre algo
                # mon_tenseur.argmax(-1) donnera le m??me r??sultat
                if pred == labels:  # On est pas oblig?? de sortir la donn??e via pred[0] et labels[0] car il n'y a qu'une valeur dans le tenseur, mais on peut, les deux reviennent au m??me
                    correct += 1
                if labels == 0:  # Atrial_premature
                    Confusion_matrix[0][pred] += 1
                if labels == 1:  # Left_bundle
                    Confusion_matrix[1][pred] += 1
                if labels == 2:  # Normal
                    Confusion_matrix[2][pred] += 1
                if labels == 3:  # Paced_beat
                    Confusion_matrix[3][pred] += 1
                if labels == 4:  # Right_bundle
                    Confusion_matrix[4][pred] += 1
                if labels == 5:  # Ventricular_escape
                    Confusion_matrix[5][pred] += 1
                if labels == 6:  # Ventricular_premature
                    Confusion_matrix[6][pred] += 1


                output_trans = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                y_pred.extend(output_trans)  # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)  # Save Truth
            print(f"Epoch : {epoch + 1} - Taux de classification = {correct / len(trainloader_acc)}")
            print(Confusion_matrix.astype(int))
            ###################################################################################
            cf_matrix = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 7, index=[i for i in classes],
                                 columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            plt.savefig('outputs/output_train_' + current_time + '_epoch_' + str(epoch) + '.png')
            plt.close('all')
            ##################################################################################
            print("******************************************************************")
            """
    print('Finished Training')

