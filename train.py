import optax
from Qlstm import QLSTM
import jax
import jax.numpy as jnp
import mlflow
import time
import numpy as np
import torch 
import os
#(X_train,Y_train,X_test,Y_test,kernel_size,n_layers,ansatz,out_size,out_channels,trainloader,testloader,run_Name,dataset,original_dataset,architecture,key)
def train_model(X_train,Y_train,X_test,Y_test,trainloader,testloader,original_dataset,run_name,dataset, seq_len,n_layers,n_qubits,concat_size,target_size,key):
    experiment_name = dataset
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # Función de pérdida y optimizador
  
    optimizer= optax.adam(learning_rate=5e-4)
    loss_list = []
    net=QLSTM(seq_len,n_layers,n_qubits,concat_size,target_size)
    key2 = jax.random.PRNGKey(key)
    sample_input = jnp.array(X_train[:16,:,:])
    input_shape = sample_input.shape
    #input_shape = (1,) + input_shape
    
    params = net.init(key2, jnp.ones(input_shape))
    opt_state = optimizer.init(params)
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        def mse_loss(params,inputs, targets):
            predictions = net.apply(params,inputs)
            loss = jnp.mean((predictions - targets) ** 2)
            return loss
        loss, grads = jax.value_and_grad(mse_loss)(params,inputs,targets)
        #print(f"Gradients for QuanvLayer1D: {grads['QuanvLayer1D']}")
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        #print("Gradients for QuanvLayer1D:", grads)
        return params, opt_state, loss
    @jax.jit
    def get_resta(inputs, targets,max_dataset,min_dataset):
        Predictions = net.apply(params,inputs)
        #print('esta es el predictions',Predictions)
        Predictions = Predictions*(max_dataset-min_dataset)+min_dataset
        targets=targets*(max_dataset-min_dataset)+min_dataset
        resta=Predictions-targets
        resta_l.append(resta)
        forecasted.append(Predictions)
        return resta_l,forecasted
    # Entrenamiento del modelo
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('concat size',concat_size)
        mlflow.log_param('layers',n_layers)
        mlflow.log_param('qubits',n_qubits)
        
        for epoch in range(30):
            epoch_loss = 0.0
            if epoch==1:
              start_time = time.time()
            for data in trainloader:
                inputs = data[0]
                #print(inputs.shape)
                targets = data[1]
                params, opt_state, loss = train_step(params, opt_state, inputs, targets)
            epoch_loss += inputs.shape[0] * loss
            
            
        epoch_loss = epoch_loss/len(X_train)
        print('[Epoch %d] loss: %.6f' % (epoch + 1, epoch_loss))
        mlflow.log_metric('loss',epoch_loss)
        
        end_time = time.time()
        #Calculate and print the total execution time
        total_time = end_time - start_time
        mlflow.log_metric('training time',total_time)
        n_points_test = len(Y_test)
        print('points test',n_points_test)
        print('we begin testing')
        resta_l=[]
        forecasted=[]
        targ=[]
        #for data in testloader:
            #inputs = data[0]
            #targets = data[1]
        max_dataset=jnp.max(original_dataset,axis=0)
        #print('el max dataset',max_dataset)
        min_dataset=jnp.min(original_dataset,axis=0)
        for data in testloader:
            inputs = data[0]
            targets = data[1]
            #print('el shape del input',inputs.shape)
            p = net.apply(params,inputs)
            targ.append(targets)
            forecasted.append(p)
        
        print('el shape del input',inputs.shape)
        forecasted= np.concatenate(forecasted, axis=0)
        print('el forecasted',forecasted.shape)
        targ=np.concatenate(targ,axis=0)
        print('el target',forecasted.shape)
        forecasted = forecasted*(max_dataset-min_dataset)+min_dataset
        targ=targ*(max_dataset-min_dataset)+min_dataset
        resta_l=forecasted-targ
        resta_l=np.array(resta_l)
        
        # We save loss_hist for later use:
        folder_name = dataset
        filenameLoss=f'{folder_name}/test_forecasted{run_name}.pt'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(forecasted, filenameLoss)
        print('max',np.max(original_dataset))
        RMSE = np.sqrt(np.sum(np.square(resta_l))/n_points_test)
        print('RMSE',RMSE)
        mlflow.log_metric('RMSE_TEST',RMSE)
        mae = (np.sum(np.abs(resta_l))/n_points_test)
        print("MAE=",mae)
        mlflow.log_metric('MAE_TEST',mae)
        #print('los ytest',np.absolute(Y_test))
        #print('los points',n_points_test)
        #MAPE = np.sum(np.absolute(resta_l)/np.absolute(Y_test))/n_points_test
        #mlflow.log_metric('MAPE_TEST',MAPE)
       # print("MAPE=",MAPE)
    return time