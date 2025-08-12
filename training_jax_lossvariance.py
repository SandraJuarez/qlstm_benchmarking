import optax
from Qlstm import QLSTM
from LSTM import ClassicalLSTM as LSTM
from scipy.stats import pearsonr
import jax
import jax.numpy as jnp
import mlflow
import scipy.fftpack
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

def train_model(
    X_train, Y_train, X_test, Y_test, trainloader, testloader,
    original_dataset, run_name, dataset, seq_len, n_layers, n_qubits,
    concat_size, target_size, key, model, convergence=False,
    plot=True, return_all_hidden=False
):
    """
    Train either a QLSTM or a classical LSTM on the given data.
    Optionally return hidden states at each epoch if `return_all_hidden=True`.
    """

    # -----------------
    # 1) Setup mlflow experiment
    # -----------------
    experiment_name = dataset
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # If needed: experiment_id = experiment.experiment_id

    # -----------------
    # 2) Initialize model
    batch_size_for_init = 16
    sample_input = jnp.array(X_train[:batch_size_for_init, :, :])  # shape: (16, seq_len, features)
    input_shape = sample_input.shape
    features=sample_input.shape[2]
    # -----------------
    if model == 'QLSTM':
        net = QLSTM(seq_len, n_layers, n_qubits, concat_size, target_size, return_all_hidden=False)
        # Note: We'll call net with return_all_hidden=True only when we want to get hidden states.
    elif model == 'LSTM':
        # For your classical LSTM, ensure it also can handle return_all_hidden if needed
        net = LSTM(seq_len,features, concat_size, target_size)
    else:
        raise ValueError("Unknown model type. Choose 'QLSTM' or 'LSTM'.")

    
    
    key2 = jax.random.PRNGKey(key)

    # -----------------
    # 3) Initialize parameters and optimizer
    # -----------------
    print('el shape es',input_shape)
    params = net.init(key2, jnp.ones(input_shape))  # init with dummy input
    lr = 5e-4
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # -----------------
    # 4) Define training/validation step functions (JIT-compiled)
    # -----------------
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        def mse_loss(params, inputs, targets):
            predictions = net.apply(params, inputs)  # uses default return_all_hidden=False
            loss = jnp.mean((predictions - targets) ** 2)
            return loss

        loss, grads = jax.value_and_grad(mse_loss)(params, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def validation_step(params, inputs, targets):
        predictions = net.apply(params, inputs)
        loss = jnp.mean((predictions - targets) ** 2)
        return loss,predictions

    # -----------------
    # 5) Training Loop Setup
    # -----------------
    if convergence:
        epochs = 200
    else:
        epochs = 25

    loss_history = []
    val_loss_history = []
    previous_losses = []
    variance_threshold = 5e-6
    converged = False
    convergence_ep = None

    # For saving hidden states at each epoch (optional)
    hidden_states_per_epoch = {}  # { epoch_index: hidden_seq_array }

    # Useful for re-scaling if needed
    max_dataset = jnp.max(original_dataset, axis=0)
    min_dataset = jnp.min(original_dataset, axis=0)

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_param('concat size', concat_size)
        mlflow.log_param('layers', n_layers)
        mlflow.log_param('qubits', n_qubits)
        mlflow.log_param('learning rate', lr)
        mlflow.log_param('init', 'xavier')

        # -----------------
        # 6) Train / Validate Over Epochs
        # -----------------
        epoch_preds=[]
        for epoch in range(epochs):
            if epoch == 0:
                start_time = time.time()

            print(f"\nStarting epoch {epoch + 1}")
            epoch_loss = 0.0
            epoch_losses_batchwise = []

            # --------- Training Loop ----------
            for data in trainloader:
                inputs, targets = data[0], data[1]
                params, opt_state, loss = train_step(params, opt_state, inputs, targets)
                epoch_loss += inputs.shape[0] * loss
                epoch_losses_batchwise.append(loss)

            # Average training loss over entire training set
            epoch_loss /= len(X_train)
            loss_history.append(epoch_loss)

            # --------- Validation Loop ----------
            val_loss = 0.0
            epoch_losses_val = []
            forecasted_epoch=[]
            for val_data in testloader:
                val_inputs, val_targets = val_data[0], val_data[1]
                batch_val_loss, predictions= validation_step(params, val_inputs, val_targets)
                forecasted_epoch.append(predictions)
                val_loss += batch_val_loss * val_inputs.shape[0]
                epoch_losses_val.append(batch_val_loss)
            forecasted_epoch = jnp.concatenate(forecasted_epoch, axis=0)
            epoch_preds.append(forecasted_epoch)

            val_loss /= len(X_test)
            val_loss_history.append(val_loss)
            previous_losses.append(val_loss)

            # Print info
            print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.6f}, Validation Loss = {val_loss:.6f}")

            # --------- Check Convergence (Variance) ----------
            if convergence and len(previous_losses) >= 20:
                loss_variance = np.var(previous_losses[-5:])
                print(f"Loss Variance (last 5 epochs): {loss_variance:.8e}")
                if loss_variance < variance_threshold:
                    print(f"Convergence achieved! Stopping at epoch {epoch+1}")
                    convergence_ep = epoch + 1
                    break

            # --------- (Optional) Retrieve Hidden States This Epoch ----------
            if return_all_hidden:
                # We'll do it on a small batch from testloader (or trainloader). 
                # For instance, let's pick the *first* batch from testloader:
                sample_data = next(iter(testloader))  
                sample_inputs, _ = sample_data
                
                preds, hidden_seq = net.apply(
                    params,  # model parameters
                    sample_inputs,  # input
                    return_all_hidden=True  # override the default
                )
                # hidden_seq.shape -> (batch_size, seq_length, hidden_size)

                hidden_states_per_epoch[epoch] = np.asarray(hidden_seq)  
                # You can store them in a dict keyed by epoch

        # End of epochs

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTraining completed. Total time: {total_time:.2f} seconds")
        sampling_rate = 1 
        frequencies = np.fft.fftfreq(len(epoch_preds[0]), d=1/sampling_rate)
        fft_magnitudes = []

        for pred in epoch_preds:
            pred_np = np.array(pred)
            fft_vals = np.abs(scipy.fftpack.fft(pred_np))
            fft_magnitudes.append(fft_vals)

        # Log final metrics / time
        mlflow.log_metric('training time', total_time)
        mlflow.log_metric('loss', epoch_loss)
        mlflow.log_metric('loss val', val_loss)
        if convergence_ep is not None:
            mlflow.log_metric('convergence epoch', convergence_ep)

        # Save training/validation loss
        folder_name = dataset
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        filenameLoss = f'{folder_name}/LossT_{run_name}.pt'
        filenameLossV = f'{folder_name}/LossV_{run_name}.pt'
        torch.save(np.array(loss_history), filenameLoss)
        torch.save(np.array(val_loss_history), filenameLossV)

        # 7) Final Inference on Test Set
        forecasted = []
        targ = []
        for data in testloader:
            inputs, targets = data[0], data[1]
            p = net.apply(params, inputs)  # default final output
            targ.append(targets)
            forecasted.append(p)

        forecasted = np.concatenate(forecasted, axis=0)
        targ = np.concatenate(targ, axis=0)

        # Error metrics
        resta_l = forecasted - targ
        n_points_test = len(Y_test)
        RMSE = np.sqrt(np.sum(np.square(resta_l)) / n_points_test)
        mae = (np.sum(np.abs(resta_l)) / n_points_test)

        print('RMSE:', RMSE)
        print("MAE:", mae)
        mlflow.log_metric('RMSE_TEST', RMSE)
        mlflow.log_metric('MAE_TEST', mae)
        # Aplanar los arreglos por si tienen forma (N, 1)
        forecasted_flat = forecasted.flatten()
        targ_flat = targ.flatten()

        # Calcular correlación de Pearson
        pearson_corr, _ = pearsonr(forecasted_flat, targ_flat)

        print(f"Pearson Correlation (Test): {pearson_corr:.4f}")
        mlflow.log_metric('Pearson_corr', pearson_corr)
        # --- Compute Spectral Bias Index (SBI) curve across epochs ---
        target_freqs = [0.01, 0.4]  # Low and High frequency
        f_indices = [np.argmin(np.abs(frequencies - f)) for f in target_freqs]

        sbi_curve = []
        for fft in fft_magnitudes:
            low_mag = fft[f_indices[0]]
            high_mag = fft[f_indices[1]]
            sbi = low_mag / high_mag if high_mag > 0 else float('inf')
            sbi_curve.append(sbi)

        # Save the full SBI curve
        sbi_curve = np.array(sbi_curve)
        sbi_path = f'{folder_name}/SBI_{run_name}.npy'
        np.save(sbi_path, sbi_curve)
        mlflow.log_artifact(sbi_path)

        # Also log the final SBI value for quick reference
        print(f"Final Spectral Bias Index (SBI): {sbi_curve[-1].item():.4f}")
        mlflow.log_metric('SBI_final', sbi_curve[-1].item())


        # Optional plotting
        if plot:
            plt.plot(forecasted, label='Predicted')
            plt.plot(targ, label='Real')
            plt.legend()
            plt.show()
            target_freqs = [0.01, 0.05, 0.1, 0.2, 0.4]

            f_indices = [np.argmin(np.abs(frequencies - f)) for f in target_freqs]

            for i, freq in zip(f_indices, target_freqs):
                freq_component = [fft[i] for fft in fft_magnitudes]
                plt.plot(freq_component, label=f'{freq} Hz')

            plt.xlabel('Epoch')
            plt.ylabel('Magnitude in prediction')
            plt.legend()
            plt.title('Learning of each frequency')
            plt.show()

            #spectral bias index curve
            plt.figure()
            plt.plot(sbi_curve, label='Spectral Bias Index (0.01Hz / 0.4Hz)')
            plt.xlabel('Epoch')
            plt.ylabel('SBI')
            plt.title('Spectral Bias Index over Training')
            plt.grid(True)
            plt.legend()
            plt.show()

    # You may also want to return your hidden states dict if `return_all_hidden=True`
    # so you can analyze or plot them outside:
    return params, hidden_states_per_epoch if return_all_hidden else None
