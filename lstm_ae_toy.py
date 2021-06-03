import argparse
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from AE import *

lower = 0
upper = 1
mu = 0.5
sigma = 0.1
sequence_size = 50
N = 50
dataset_size = 10000
input_size = 1

num_epochs = 20
optimizer = torch.optim.Adam
def get_loaders(batch_size):
    dataset = np.zeros((dataset_size, N))
    for i in range(0, dataset_size):
        samples = scipy.stats.truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
        dataset[i, :] = samples
    training_set_size = np.math.floor(dataset_size*0.6)
    validation_set_size = np.math.floor(dataset_size*0.2)



    training_set = dataset[0:training_set_size,:]
    validation_set = dataset[training_set_size:training_set_size+validation_set_size,:]
    test_set = dataset[training_set_size+validation_set_size:dataset_size,:]

    train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

    return train_loader, validation_loader, test_loader
def train(model,train_loader,criterion,optim,epoch,clip):
    loss = 0
    model.train()
    for i, seq in enumerate(train_loader):
        seq = seq.float()
        seq = seq.reshape( len(seq),sequence_size, input_size).to(device)

        output = model(seq)
        loss = criterion(output, seq)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()
        if (i + 1) % 10 == 0:
            print(f'Training: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return loss


'''
For validation and testing
'''
def run_model(model,loader,criterion,epoch,to):
    loss = 0
    model.eval()
    output = 0
    seq = 0
    for i, seq in enumerate(loader):
        seq = seq.float()
        seq = seq.reshape(len(seq),sequence_size, input_size).to(device)

        output = model(seq)

        loss = criterion(output, seq)
        if (i + 1) % 10 == 0:
            print(f'{to}: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return loss,output, seq

def show_signals(dataloader):
    x = [j for j in range(0, sequence_size)]

    for i, seq in enumerate(dataloader):
        seq = seq[0]

        plt.clf()
        if i == 3: break
        plt.xlabel("Time")
        plt.ylabel("Signal Value")
        plt.title('Signal')
        plt.plot(x,seq,color='blue')
        plt.show()


'''
create the model and train it
'''
def Ae_Toy(train_loader,config):
    #show_signals(train_loader)
    model = LSTM_AE(sequence_size,input_size,int(config['hidden_size']/2)).to(device)    #LSTM(input_size,config['hidden_size'],batch_size).to(device)
    criterion = nn.MSELoss()
    optim = optimizer(model.parameters(),lr=config['lr'])
    x,y,z = [],[],[]
    for epoch in range(0,num_epochs):
        train_loss= train(model,train_loader,criterion,optim,epoch,config['clipping'])
        #x.append(epoch)
        #z.append(train_loss.item())
    return model


def validation(model,validation_loader):

    criterion = nn.MSELoss()
    output, seq =0,0
    loss = []
    for epoch in range(0,num_epochs):
        val_loss, output,seq= run_model(model,validation_loader,criterion,epoch,'Validation')
        loss.append(val_loss.item())

    return sum(loss)/len(loss)
def test(model,test_loader,configs):
    criterion = nn.MSELoss()
    output1, seq1 =0,0
    output2, seq2 =0,0
    loss = []
    for epoch in range(0,num_epochs):
        val_loss, output1,seq1= run_model(model,test_loader,criterion,epoch,'Testing')
        if epoch == num_epochs - 2:
            output2 = output1
            seq2 = seq1
        loss.append(val_loss.item())

    x = [i for i in range(0,sequence_size)]

    output = output1.reshape(len(output1),sequence_size,input_size)[-1].tolist()
    seq = seq1.reshape(len(seq1),sequence_size,input_size)[-1].tolist()
    output2 = output1.reshape(len(output1),sequence_size,input_size)[-2].tolist()
    seq2 = seq1.reshape(len(seq1),sequence_size,input_size)[-2].tolist()
    y = [output[i][0] for i in range(0,sequence_size)]
    z = [seq[i][0] for i in range(0,sequence_size)]
    show(x,y,z,"Time","Signal Value","Prediction","Real",configs,'1')

    y = [output2[i][0] for i in range(0,sequence_size)]
    z = [seq2[i][0] for i in range(0,sequence_size)]
    show(x,y,z,"Time","Signal Value","Prediction","Real",configs,'2')

    return sum(loss)/len(loss)

def show(x,y,z, label_x,label_y,legend_1,legend_2, configs,number):
    plt.title(f'lr: {configs["lr"]},batch_s:{configs["batch_size"]},epochs: {configs["epochs"]},hidden_s: {configs["hidden_size"]}, clipping: {configs["clipping"]}')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(x,y,color='red',label=legend_1)
    plt.plot(x,z,color='blue',label=legend_2)
    plt.legend(loc="upper left")
    #plt.savefig("PDL_HW2_Results/"+str(configs["lr"]).replace('.','')+str(configs["hidden_size"]).replace('.','')+str(configs["clipping"]).replace('.','')+number+'.png')
    plt.show()
    plt.clf()

def train_and_validate(configs,train_loader,validation_loader):
    trained_model = Ae_Toy(train_loader, configs)
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

    with open('best.txt', 'w') as f:
        f.write(f'best configs:{best_configs["hypers"]}, with Avg loss of {best_configs["loss"]}')

    test(best_model, test_loader, best_configs['hypers'])

'-------------------------------------------------------------------------------------------------'
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


#given_hypers()
#specific()