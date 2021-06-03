import argparse
import torchvision
from torchvision.transforms import transforms

from AE import *
import matplotlib.pyplot as plt
import numpy as np


batch_size = 50

learning_rate = 0.001
momentum = 0.5
log_interval = 10
num_epochs = 5
random_seed = 1
input_size = 28
sequence_size = 28
hidden_size = 64
optimizer = torch.optim.Adam

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def get_loaders(batch_size):


    test_data =   torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    train_set =  torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

    train_set, val_set = torch.utils.data.random_split(train_set, [36000, 24000])
    test_loader = torch.utils.data.DataLoader(test_data,
      batch_size=batch_size, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader,validation_loader,test_loader

'''
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
#fig.show()
'''


def run_model(model,loader,MSEcriterion,CEcriterion,epoch,to):
    output = 0
    last_labels = []
    last_props = []
    model.eval()
    last_seq = 0
    loss = 0
    accuracy = 0
    for i, seq in enumerate(loader):


        cls = seq[-1].to(device)
        last_labels = cls
        seq = seq[0].float()
        seq = seq.reshape(len(seq),sequence_size,input_size).to(device)

        last_seq = seq

        output,props = model(seq)
        last_props = nn.Softmax(dim=1)(props)
        lossCE = CEcriterion(props, cls.long())
        lossMS = MSEcriterion(output, seq)
        overall_loss = lossCE+lossMS
        loss += overall_loss.item()

        count = 0
        for j in range(0, len(last_props)):
            if torch.argmax(last_props[j]) == last_labels[j]:
                count += 1
        accuracy +=count / len(last_props)
        if (i + 1) % 100 == 0:
                print(f'{to}: Epoch [{epoch + 1}/{num_epochs}],Loss: {overall_loss.item():.4f}')

    return loss/len(loader),accuracy/len(loader), output,last_seq,last_props,last_labels

def validation(model,validation_loader):
    MSEcriterion = nn.MSELoss()
    CEcriterion = nn.CrossEntropyLoss()
    output, seq =0,0
    loss = []
    for epoch in range(0,num_epochs):
        val_loss,accuracy, output,seq,last_props,last_labels= run_model(model,validation_loader,MSEcriterion,CEcriterion,epoch,'Validation')
        loss.append(val_loss)
        print(f'val_loss: {val_loss}')
        print(f'val_accuracy: {accuracy}')

    return sum(loss)/len(loss)

def test(model,test_loader,config):
    MSEcriterion = nn.MSELoss()
    CEcriterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    for epoch in range(0,config['epochs']):
        test_loss,accuracy, output,seq,last_props,last_labels= run_model(model,test_loader,MSEcriterion,CEcriterion,epoch,'Test')
        losses.append(test_loss)
        accuracies.append(accuracy)
        print(f'test_loss: {test_loss}')
        print(f'test_accuracy: {accuracy}')

    x = [i for i in range(0,config["epochs"])]

    show(x, accuracies, [], "Epochs", "Percentage(%)", "Accuracy", "", config, '2')

    plt.imshow(output[0].tolist(), cmap='gray', interpolation='none')
    plt.show()
    plt.imshow(seq[0].tolist(), cmap='gray', interpolation='none')
    plt.show()
    return sum(losses)/len(losses)

def show(x,y,z, label_x,label_y,legend_1,legend_2, configs,number):
    plt.title(f'lr: {configs["lr"]},batch_s:{configs["batch_size"]},epochs: {configs["epochs"]},hidden_s: {configs["hidden_size"]}, clipping: {configs["clipping"]}')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(x,y,color='red',label=legend_1)
    if len(z)>0:
        plt.plot(x,z,color='blue',label=legend_2)
    plt.legend(loc="upper left")
    #plt.savefig("PDL_HW2_Results/MNIST/"+str(configs["lr"]).replace('.','')+str(configs["hidden_size"]).replace('.','')+str(configs["clipping"]).replace('.','')+number+'.png')
    plt.show()
    #plt.clf()


def train(model, train_loader, MSEcriterion, CEcriterion, optim, epoch, clip):
    loss = 0
    output = 0
    last_labels = []
    last_props = []
    model.train()
    last_seq = 0
    accuracy = 0
    for i, seq in enumerate(train_loader):
        cls = seq[-1].to(device)
        last_labels = cls

        seq = seq[0].float()
        seq = seq.reshape(len(seq), sequence_size, input_size).to(device)
        last_seq = seq

        output, props = model(seq)
        last_props = nn.Softmax(dim=1)(props)

        lossCE = CEcriterion(props, cls.long())
        lossMS = MSEcriterion(output, seq)
        overall_loss = lossCE + lossMS
        optim.zero_grad()
        overall_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()
        loss += overall_loss.item()

        count = 0
        for j in range(0, len(last_props)):
            if torch.argmax(last_props[j]) == last_labels[j]:
                count += 1
        accuracy +=count / len(last_props)

        if (i + 1) % 100 == 0:
            print(f'Training: Epoch [{epoch + 1}/{num_epochs}],Loss: {overall_loss.item():.4f}')

    return loss / len(train_loader),accuracy/len(train_loader), output, last_seq

def Ae_MNIST(train_loader,config):


    trained_model = LSTM_AE(sequence_size,input_size,int(config['hidden_size']/2),classification=True).to(device)

    CEcriterion = nn.CrossEntropyLoss()

    MSEcriterion = nn.MSELoss()
    optim = optimizer(trained_model.parameters(),lr=config['lr'])
    accuracies = []
    losses = []
    for epoch in range(0,config["epochs"]):
        train_loss,accuracy,output,last_seq= train(trained_model,train_loader,MSEcriterion,CEcriterion,optim,epoch,config['clipping'])
        losses.append(100*train_loss)
        accuracies.append(100*accuracy)
        print(f'Loss: {train_loss}')
        print(f'accuracy: {accuracy}')
    x = [i for i in range(0,config["epochs"])]


    show(x,losses,[],"Epochs","","Loss","",config,'1')
    show(x,accuracies,[],"Epochs","Percentage(%)","Accuracy","",config,'2')
    plt.imshow(output[0].tolist(), cmap='gray', interpolation='none')
    plt.show()
    plt.imshow(last_seq[0].tolist(), cmap='gray', interpolation='none')
    plt.show()

    plt.imshow(output[1].tolist(), cmap='gray', interpolation='none')
    plt.show()
    plt.imshow(last_seq[1].tolist(), cmap='gray', interpolation='none')
    plt.show()

    plt.imshow(output[4].tolist(), cmap='gray', interpolation='none')
    plt.show()
    plt.imshow(last_seq[4].tolist(), cmap='gray', interpolation='none')
    plt.show()


    return trained_model


def train_and_validate(configs,train_loader,validation_loader):
    trained_model = Ae_MNIST(train_loader, configs)
    loss_avg = validation(trained_model, validation_loader)
    return {'loss': loss_avg,'hypers': configs, 'model': trained_model}


def test_and_show(hypers_loss,test_loader):
    min_loss = hypers_loss[0]['loss']
    best_model = hypers_loss[0]['model']
    best_configs = hypers_loss[0]
    for d in hypers_loss:
        if d['loss'] < min_loss:
            min_loss = d['loss']
            best_model = d['model']
            best_configs = d
    print(f'best configs:{best_configs["hypers"]}, with Avg loss of {best_configs["loss"]}')

    test(best_model, test_loader, best_configs['hypers'])

'---------------------------------------------------------------------------------------------------------------'



def specific():
    global num_epochs
    num_epochs = 70
    batch_size = 50
    hypers_loss = []
    lrs = [0.01,0.001,0.0001]
    clippings = [0.5,0.8,1.0]
    hidden_sizes = [32,64,128]
    train_loader, validation_loader, test_loader = get_loaders(batch_size)
    for lr in lrs:
        for clipping in clippings:
            for hidden_size in hidden_sizes:
                configs  = {"lr":lr,"batch_size":batch_size,"epochs":num_epochs,"clipping":clipping,"hidden_size":hidden_size}
                hypers_loss.append(train_and_validate(configs,train_loader,validation_loader))

    test_and_show(hypers_loss,test_loader)

def given_hypers():
    optimizers ={'Adam':torch.optim.Adam,'SGD':torch.optim.SGD}
    parser = argparse.ArgumentParser(description='Get Hyperparameters')
    parser.add_argument("Epochs",type=int)
    parser.add_argument("Optimizer", type=str)
    parser.add_argument("Learning_Rate", type=float)
    parser.add_argument("Gradient_Clipping", type=float)
    parser.add_argument("Batch_Size", type=int)
    parser.add_argument("Hidden_State_Size", type=int)
    args = parser.parse_args()
    global  num_epochs
    global optimizer
    optimizer = optimizers[args.Optimizer]
    num_epochs = args.Epochs
    hypers_loss = []
    train_loader, validation_loader, test_loader = get_loaders(args.Batch_Size)
    configs = {'lr': args.Learning_Rate,'batch_size':args.Batch_Size,'epochs':args.Epochs,'clipping':args.Gradient_Clipping,'hidden_size':args.Hidden_State_Size}
    hypers_loss.append(train_and_validate(configs, train_loader, validation_loader))
    test_and_show(hypers_loss,test_loader)

#specific()
#given_hypers()
