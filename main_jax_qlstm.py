from train import train_model
import load_data_jax_metrics
import training_jax_lossvariance as train
import numpy as np




#dataset_list=['frequencies_low','frequencies_high','frequencies_noisy','frequencies_small','suma','mackey','legendre3','btc','sp500'] 
#first with only one dataset
dataset_list=['frequencies_low']
#sample_size_list=[5,5,5,5,5,4,5,5,5] #the sample size for each dataset, for euro and legendre this is 5.  For mackey is 4
sample_size_list=[5]
#qubits_list=[16,20,24,32] #in the paper we utilize 2,4,6 or 8 
qubits_list=[2]
hidden_list=[16]#[16,32,48,64]
key_list=[0,1,2,3,4,5,6,7,8,9] #the key for the random number generator
kernel_size=2
architecture='super_parallel' #options are no_reupload, parallel and super_parallel
#point_list=[268,518,768,1018,1268,1518,1768,2018,2268,3518,2768,3018,3268,3518,3768,4018]
point_list=[1000]
 # if you want to work with parallel ansatz indicate the number of layers in the loop below in variable n_layers
model="QLSTM" #options are LSTM, QLSTM
convergence=False




times=np.zeros((int(len(point_list)),int(len(key_list))))
for p in range(len(point_list)):  
    points=point_list[p] 
    for k in range(len(key_list)):
        key=key_list[k]
        for d in range(len(dataset_list)):
            dataset=dataset_list[d]
            seq_len=sample_size_list[d]
            
            X_train,Y_train,X_test,Y_test,trainloader,testloader,data,features=load_data_jax_metrics.data(dataset,points)
            print(features)
            target_size=1
            for q in range(len(qubits_list)):
                n_qubits=qubits_list[q]
                if architecture=='super_parallel':
                    n_layers=n_qubits//kernel_size
                elif architecture=='parallel' or architecture=='no_reupload':
                    n_layers=4
                for h in range(len(hidden_list)):
                    concat_size=hidden_list[h]
                    #run_Name=dataset+ansatz+str(out_channels)+str(n_layers)+str(architecture)+str(key)
                    run_name=dataset+str(concat_size)+str(n_qubits)+str(key)+str('small8Q25')
                    train.train_model(X_train,Y_train,X_test,Y_test,trainloader,testloader,data,run_name,dataset, seq_len,n_layers,n_qubits,concat_size,target_size,key,model,convergence)
                        
