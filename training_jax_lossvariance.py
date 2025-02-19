import optax
from Qlstm import QLSTM
from LSTM import ClassicalLSTM as LSTM
import jax
import jax.numpy as jnp
import mlflow
import time
import numpy as np
import torch 
import os

jax.config.update('jax_enable_x64', True)
def train_model(X_train,Y_train,X_test,Y_test,trainloader,testloader,original_dataset,run_name,dataset, seq_len,n_layers,n_qubits,concat_size,target_size,key,model):
    experiment_name = dataset
    experiment = mlflow.get_experiment_by_name(experiment_name)
    #experiment_id = experiment.experiment_id

    #net = CNN(kernel_size,n_layers,ansatz,out_size,out_channels,architecture)
    sample_output = jnp.array(Y_train[0])
    output_shape = sample_output.shape
    sample_input = jnp.array(X_train[0,:,:])
    input_shape = sample_input.shape
    input_shape = (1,) + input_shape
    print(input_shape)
    features=input_shape[0]
    if model=='QCNN':
        net = QLSTM(seq_len,n_layers,n_qubits,concat_size,target_size)
    elif model=='LSTM':
        net=LSTM(seq_len,concat_size,target_size)
    key2 = jax.random.PRNGKey(key)
    sample_input = jnp.array(X_train[:16,:,:])
    input_shape = sample_input.shape
    #input_shape = (1,) + input_shape
    optimizer= optax.adam(learning_rate=5e-4)
    params = net.init(key2, jnp.ones(input_shape))
    opt_state = optimizer.init(params)
    #variables = net.init_with_output(key2, jnp.ones((2,2,3)))
    #f_jitted=jax.jit(nn.init(CNN,net))
    #variables = f_jitted(key2, jnp.ones(input_shape))
    #params = variables['params']
    #weights = jnp.ones([2,2,3])
    #params = {"weights": weights}
    lr=5e-4
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)
    #for key, value in params.items():
        #print(f"Param before {key}: {value}")
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        def mse_loss(params,inputs, targets):
            predictions = net.apply(params,inputs)
            loss = jnp.mean((predictions - targets) ** 2)
            return loss
        loss, grads = jax.value_and_grad(mse_loss)(params,inputs,targets)
        #print(f"Gradients for QuanvLayer1D: {grads['QuanvLayer1D']}")
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        #print("Gradients for QuanvLayer1D:", grads)
        return params, opt_state, loss
    resta_l=[]
    @jax.jit
    def validation_step(params, inputs, targets):
        """Calculate the validation loss for a batch."""
        predictions = net.apply(params, inputs)
        loss = jnp.mean((predictions - targets) ** 2)  # MSE Loss
        return loss
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
    # Run the training loop
    epochs = 200 #for time complexity measurments set to 5
    Loss_hist=[]
    #for the rescaling
    #print('el shape del original es',original_dataset.shape)
    max_dataset=jnp.max(original_dataset,axis=0)
    #print('el max dataset',max_dataset)
    min_dataset=jnp.min(original_dataset,axis=0)
    #print('el max dataset y el min dataset son',max_dataset,min_dataset)
    #rescaled_target=targets*(max_dataset-min_dataset)+min_dataset
    # Record the start time
    #start_time = time.time()
    loss_history = []
    val_loss_history = []
    variance_threshold = 5e-6  # Example threshold for loss variance
    converged = False
    previous_losses=[]
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('concat size',concat_size)
        mlflow.log_param('layers',n_layers)
        mlflow.log_param('qubits',n_qubits)
        mlflow.log_param('learning rate',lr)
        mlflow.log_param('init','xavier')
        for epoch in range(epochs):
            if epoch == 0:
                start_time = time.time()

            print(f'\nStarting epoch {epoch + 1}')
            epoch_loss = 0.0
            epoch_losses = []  # To store loss for each batch in the epoch
            
            # Training Loop
            for data in trainloader:
                inputs = data[0]
                targets = data[1]
                
                params, opt_state, loss = train_step(params, opt_state, inputs, targets)
                epoch_loss += inputs.shape[0] * loss
                epoch_losses.append(loss)  # Store batch loss
            
            # Compute average training loss for the epoch
            epoch_loss = epoch_loss / len(X_train)
            loss_history.append(epoch_loss)  # Store epoch loss history
            
        
            epoch_losses_val=[]
            # Validation Loss Calculation
            val_loss = 0.0
            for val_data in testloader:
                val_inputs = val_data[0]
                val_targets = val_data[1]
                val_loss += validation_step(params, val_inputs, val_targets) * val_inputs.shape[0]
                epoch_losses_val.append(val_loss)
                # Track loss history for variance calculation
            val_loss /= len(X_test)  # Average validation loss
            previous_losses.append(val_loss)

            if len(previous_losses) >= 5:  # Calculate variance over last 5 epochs
                loss_variance = np.var(previous_losses[-5:])
                print(f'Loss Variance (last 5 epochs): {loss_variance:.8e}')
                
                if loss_variance < variance_threshold:
                    print(f'Convergence achieved! Stopping at epoch {epoch+1}')
                    convergence_ep=epoch+1
                    break
            
            val_loss_history.append(val_loss)
            #variance in the batch of current epoch
            loss_variance = np.var(epoch_losses_val)
            
            # Print epoch statistics
            print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.6f}, Variance = {loss_variance:.6e}, Validation Loss = {val_loss:.6f}")
            
            # Convergence Check Based on Loss Variance
            

        # Final Check if Convergence Was Not Reached
        if not converged:
            print("\nTraining completed without reaching the variance threshold.")
        end_time = time.time()
        #Calculate and print the total execution time
        total_time = end_time - start_time 
        print('El tiempo de entrenamiento fue:',total_time)
        
        mlflow.log_metric('training time',total_time)
        mlflow.log_metric('loss',epoch_loss)
        mlflow.log_metric('loss val',val_loss)  
        mlflow.log_metric('convergence epoch',convergence_ep)
    
        

        
        # Process is complete.
        print('Training process has finished.')
        for key, value in params.items():
            print(f"Param after {key}: {value}")
        Loss_hist_QCNN=np.array(loss_history)
        Loss_histV_QCNN=np.array(val_loss_history)
        # We save loss_hist for later use:
        folder_name = dataset
        filenameLoss=f'{folder_name}/LossT{run_name}.pt'
        filenameLossV=f'{folder_name}/LossV{run_name}.pt'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(Loss_hist_QCNN, filenameLoss)
        torch.save(Loss_histV_QCNN, filenameLossV)
        n_points_train = len(Y_train)
        print('Los puntos de entrenamiento son',n_points_train)
        
        #####################################################
        ######################################################
        ##########TESTING ##############
        n_points_test = len(Y_test)
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
    
        