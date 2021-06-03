import random
from collections import deque

import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib
#matplotlib.use('Agg')
from AE import *
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import preprocessing
import pandas
batch_size = 50
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
num_epochs = 1000
random_seed = 1
input_size = 2
sequence_size = 1006
hidden_size = 64
optimizer = torch.optim.Adam
scaler = MinMaxScaler()
df = pandas.read_csv('stocks.csv')
clipping = 0.8
def show_stock_graph(df,symbol,stock):
    g = df[df['symbol']==symbol]
    print(len(g))
    dates = []
    max = []
    i = 0
    for idx, row in g.iterrows():
        dates.append(i)
        max.append(row[stock])
        i+=1
        #break
    print(len(dates))
    plt.title(symbol+f' Stock({stock})')
    plt.xlabel('Time (day)')
    plt.ylabel('Price ($)')
    plt.plot(dates,max,color='red')
    plt.show()


def get_stocks(df,symbol):
    g = df[df['symbol']==symbol]
    close_value = []
    for idx, row in g.iterrows():
        close_value.append(row['close'])
    return torch.from_numpy(np.array(close_value))

def get_dataset(df,stock):
    if os.path.exists('data_file.pt'):
        return torch.load('data_file.pt')
    symbols = df['symbol']
    dataset = {}
    for symbol in symbols:
        dataset.setdefault(symbol,[])
    symbols = list(dict.fromkeys(symbols))
    for idx, row in df.iterrows():
        dataset[row['symbol']].append(float(row[stock]))
    to_remove = []
    for k, s in dataset.items():
        dataset[k] = np.squeeze(np.array(scaler.fit_transform(np.array(s).reshape(-1, 1))).T).tolist()
        if len(s) < 1007:
           to_remove.append(k)
    for k in to_remove:
        dataset.pop(k)
    tens = torch.from_numpy(np.array(list(dataset.values())))
    torch.save(tens,'data_file.pt')
    return tens


def get_shifted_data(tens):
    if os.path.exists('shifted_data.pt'):
        return torch.load('shifted_data.pt')
    shifted_data = []
    for seq in tens:
        seq = seq.tolist()
        shifted_seq = []
        for i in range(0,len(seq)-1):
                shifted_seq.append([seq[i],seq[i+1]])
        shifted_data.append(shifted_seq)
    tens = torch.from_numpy(np.array(shifted_data))
    torch.save(tens,'shifted_data.pt')
    return tens


def get_shifted_seq(seq):
    items = deque(seq)
    items.rotate(1)
    items.popleft()
    return list(items)


def get_data(data,indexes):
    d = []
    for i in indexes:
        d.append(data[i])
    return d


def show(x,y,z, label_x,label_y,legend_1,legend_2, configs,number):
    plt.title(f'lr: {configs["lr"]},batch_s:{configs["batch_size"]},epochs: {configs["epochs"]},hidden_s: {configs["hidden_size"]}, clipping: {configs["clipping"]}')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(x,y,color='red',label=legend_1)
    if len(z)>0:
        plt.plot(x,z,color='blue',label=legend_2)
    plt.legend(loc="upper left")
    #plt.savefig("PDL_HW2_Results/S&P500."+str(configs["lr"]).replace('.','')+str(configs["hidden_size"]).replace('.','')+str(configs["clipping"]).replace('.','')+number+'.png')
    plt.show()
    plt.clf()
def run_model(model,data,criterion_recon,criterion_pred,epoch,to):
    loss = 0

    model.eval()
    last_seq = 0
    with torch.no_grad():
        for i,seq in enumerate(data):
            seq = seq.float().to(device)
            seq = seq.reshape(len(seq),sequence_size,input_size).to(device)
            last_seq = seq
            output = model(seq)
            if sequence_size == 1006:
                recon = output[:,:,0]
                pred = output[:,:,1]

                vals = seq[:,:,0]
                vals_shifted = vals[:,1:]
                pred = pred[:,1:]
                loss = criterion_recon(recon,vals) + criterion_pred(pred,vals_shifted)
            else:
                loss = criterion_recon(output, seq)

            if (i + 1) % 1 == 0:
                    print(f'{to}: Epoch [{epoch + 1}/{num_epochs}],Loss_Recon: {loss.item():.4f}')

    return loss, output,last_seq
def train(model,data,criterion_recon,criterion_pred,optim,epoch,clip):
    loss = 0

    model.train()
    last_seq = 0
    last_output = 0
    output = 0
    for i,seq in enumerate(data):
        #print(len(seq))
        seq = seq.float().to(device)
        seq = seq.reshape(len(seq),sequence_size,input_size).to(device)
        last_seq = seq
        output = model(seq)
        if sequence_size ==1006: # if we train to predict next day's stock value
            recon = output[:,:,0]
            pred = output[:,:,1]

            vals = seq[:,:,0]
            vals_shifted = vals[:,1:]
            pred = pred[:,1:]
            loss = criterion_recon(recon,vals) + criterion_pred(pred,vals_shifted)
        else:
            loss = criterion_recon(output,seq)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()
        if (i + 1) % 1 == 0:
                print(f'Training: Epoch [{epoch + 1}/{num_epochs}],Loss_Recon: {loss.item():.4f}')

    return loss, output,last_seq




def reconstruct(data):
    kf5 = KFold(n_splits=5, shuffle=False)
    for_recon = data[0:3,:].float().reshape(3,1007,1).to(device)
    data = data[3:,:]

    print(data.shape)
    criterion_recon = nn.MSELoss()
    model = LSTM_AE(sequence_size,input_size,hidden_size).to(device)
    optim = optimizer(model.parameters(), lr=learning_rate)
    losses_test = []
    models_configs = []
    losses = []
    for i,(train_idx,test_idx) in enumerate(kf5.split(data)):
        train_data = torch.utils.data.DataLoader(get_data(data,train_idx),batch_size=batch_size)
        test_data = torch.utils.data.DataLoader(get_data(data,test_idx),batch_size=batch_size)
        for epoch in range(0,num_epochs):
            loss, output,last_seq = train(model,train_data,criterion_recon,[],optim,epoch,clipping)
            losses.append(loss)
        for epoch in range(0, num_epochs):
            loss, output, last_seq = run_model(model, test_data, criterion_recon, [], epoch,"Predict")
            losses_test.append(loss)
        models_configs.append({'model':model,'losses_test':losses_test,'losses':losses})
        losses_test.clear()
        losses.clear()
    x  = [i for i in range(0,1007)]
    min_loss = sum(models_configs[0]['losses_test'])
    best_configs = models_configs[0]
    for configs in models_configs:
        curr_loss = sum(configs['losses_test'])
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_configs = configs
    best_model = best_configs['model']
    out = best_model(for_recon)
    conf = {"lr":learning_rate,"batch_size":batch_size,"epochs":num_epochs,"clipping":clipping,"hidden_size":hidden_size}
    show(x,np.squeeze(np.array(out[0].tolist())),np.squeeze(np.array(for_recon[0].tolist())),"Time","Stock Value","Reconstruction","Real",conf,'1')
    show(x,np.squeeze(np.array(out[1].tolist())),np.squeeze(np.array(for_recon[1].tolist())),"Time","Stock Value","Reconstruction","Real",conf,'2')
    show(x,np.squeeze(np.array(out[2].tolist())),np.squeeze(np.array(for_recon[2].tolist())),"Time","Stock Value","Reconstruction","Real",conf,'3')





def reconstruct_and_predict(data):
    kf5 = KFold(n_splits=5, shuffle=False)


    print(data.shape)

    x = [i for i in range(0, num_epochs)]
    criterion_recon = nn.MSELoss()
    criterion_pred = nn.MSELoss()

    model = LSTM_AE(sequence_size, input_size, hidden_size).to(device)
    optim = optimizer(model.parameters(), lr=learning_rate)
    models_configs = []
    for i,(train_idx,test_idx) in enumerate(kf5.split(data)):
        print(f'fold {i+1} of {5}')
        losses = []
        losses_test = []
        train_data = torch.utils.data.DataLoader(get_data(data,train_idx),batch_size=batch_size)
        test_data = torch.utils.data.DataLoader(get_data(data,test_idx),batch_size=batch_size)

        for epoch in range(0,num_epochs):
            loss, output,last_seq = train(model,train_data,criterion_recon,criterion_pred,optim,epoch,clipping)
            losses.append(loss)



        for epoch in range(0, num_epochs):
            loss, output, last_seq = run_model(model, test_data, criterion_recon, criterion_pred, epoch,"Predict")
            losses_test.append(loss)

        models_configs.append({'model':model,'losses_test':losses_test,'losses':losses})


    min_loss = sum(models_configs[0]['losses_test'])
    best_configs = models_configs[0]
    for configs in models_configs:
        curr_loss = sum(configs['losses_test'])
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_configs = configs


    show(x,best_configs['losses'],[],"Epochs","","Loss","",{"lr":learning_rate,"batch_size":batch_size,"epochs":num_epochs,"clipping":clipping,"hidden_size":hidden_size},'1')
    show(x,best_configs['losses_test'],[],"Epochs","","Loss","",{"lr":learning_rate,"batch_size":batch_size,"epochs":num_epochs,"clipping":clipping,"hidden_size":hidden_size},'2')



'------------------------------------------------------------------------------------------------------'
def specific():
    data = get_dataset(df, 'close')

    reconstruct_and_predict(get_shifted_data(data))

    reconstruct(data)
    show_stock_graph(df, "AMZN", 'high')
    show_stock_graph(df, "GOOGL", 'high')




def given_hypers():
    optimizers ={'Adam':torch.optim.Adam,'SGD':torch.optim.SGD}
    parser = argparse.ArgumentParser(description='Get Hyperparameters')
    parser.add_argument("Epochs",type=int)
    parser.add_argument("Optimizer", type=str)
    parser.add_argument("Learning_Rate", type=float)
    parser.add_argument("Gradient_Clipping", type=float)
    parser.add_argument("Batch_Size", type=int)
    parser.add_argument("Hidden_State_Size", type=int)
    parser.add_argument("Sequence_Size", type=int)
    parser.add_argument("Input_Size", type=int)
    parser.add_argument("Stock", default='close', type=str)
    args = parser.parse_args()
    global  num_epochs
    global optimizer
    global learning_rate
    global clipping
    global batch_size
    global hidden_size
    global sequence_size
    global input_size

    optimizer = optimizers[args.Optimizer]
    num_epochs = args.Epochs
    learning_rate = args.Learning_Rate
    clipping = args.Gradient_Clipping
    batch_size = args.Batch_Size
    hidden_size = args.Hidden_State_Size
    sequence_size = args.Sequence_Size
    input_size = args.Input_Size
    data = get_dataset(df, args.Stock)
    if sequence_size ==1006:
        shifted_data = get_shifted_data(data)
        reconstruct_and_predict(shifted_data)
    else:
        reconstruct(data)


#given_hypers()
#specific()


