# ==== Opcional: pon esto al inicio de tu entrypoint (NO dentro de funciones) ====
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

from jax import config
config.update("jax_enable_x64", False)                    # evita float64 salvo que lo necesites
#config.update("jax_default_matmul_precision", "lowest")   # menor precisión de matmul para +mem y +vel

import jax
import jax.numpy as jnp
import numpy as np
import optax
from Qlstm import QLSTM
from LSTM import ClassicalLSTM as LSTM
import mlflow
import time, os, torch
from scipy.stats import pearsonr

# Elige tu dtype de trabajo:
DTYPE = jnp.bfloat16   # si tu GPU no soporta BF16, usa jnp.float16; si prefieres estabilidad usa jnp.float32


# =========================
# Trainer: compila una vez
# =========================
class Trainer:
    def __init__(self, net, input_shape, lr=5e-4, use_checkpoint=True, warmup=True, warmup_bs=4,micro=2):
        self.net = net
        self.lr = lr
        self.optimizer = optax.adam(learning_rate=lr)
        self.micro = micro

        dummy_key = jax.random.PRNGKey(0)
        self.params0 = self.net.init(dummy_key, jnp.ones(input_shape, DTYPE))
        self.params0 = jax.tree_map(lambda p: p.astype(DTYPE) if hasattr(p, "dtype") else p, self.params0)
        self.opt_state0 = self.optimizer.init(self.params0)

        def _apply(params, inputs):
            return self.net.apply(params, inputs)

        self.apply_fn = jax.checkpoint(_apply) if use_checkpoint else _apply

        def mse_loss(params, inputs, targets):
            preds = self.apply_fn(params, inputs)
            return jnp.mean((preds.astype(DTYPE) - targets.astype(DTYPE)) ** 2)

        # --- define funciones "normales" ---
        def _train_step(params, opt_state, inputs, targets):
            micro = self.micro  # atributo del Trainer
            if micro <= 1:
                loss, grads = jax.value_and_grad(mse_loss)(params, inputs, targets)
            else:
                B = inputs.shape[0]
                assert B % micro == 0, f"batch {B} no divisible por micro={micro}"
                bs = B // micro

                def body(p, i):
                    x_mb = jax.lax.dynamic_slice(
                        inputs,  (i*bs, 0, 0), (bs, inputs.shape[1], inputs.shape[2])
                    )
                    y_mb = jax.lax.dynamic_slice(
                        targets, (i*bs, 0),    (bs, targets.shape[1])
                    )
                    l, g = jax.value_and_grad(mse_loss)(p, x_mb, y_mb)
                    return p, (l, g)   # regresamos el mismo params sin modificar

                # ⬇️ carry es params (no tuple); scan devuelve params_out y (losses, grads)
                _, (losses, grads_stacked) = jax.lax.scan(body, params, jnp.arange(micro))

                loss  = jnp.mean(losses)                          # (micro,) -> escalar
                grads = jax.tree_map(lambda g: jnp.mean(g, 0), grads_stacked)  # promedio eje micro

            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss


        def _val_step(params, inputs, targets):
            preds = self.apply_fn(params, inputs)
            loss = jnp.mean((preds.astype(DTYPE) - targets.astype(DTYPE)) ** 2)
            return loss, preds

        # --- y aquí las JITeas asignando al atributo ---
        self.train_step = jax.jit(_train_step, donate_argnums=(0, 1))
        self.val_step   = jax.jit(_val_step)
        if warmup:
            tiny_shape = (min(warmup_bs, input_shape[0]), input_shape[1], input_shape[2])
            _ = self.train_step(
                self.params0, self.opt_state0,
                jnp.ones(tiny_shape, DTYPE),
                jnp.ones((tiny_shape[0], 1), DTYPE)
            )
        

    def init_params(self, key, input_shape):
        params = self.net.init(jax.random.PRNGKey(key), jnp.ones(input_shape, DTYPE))
        params = jax.tree_map(lambda p: p.astype(DTYPE) if hasattr(p, "dtype") else p, params)
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def step_train(self, params, opt_state, inputs, targets):
        return self.train_step(params, opt_state, inputs, targets)

    def step_val(self, params, inputs, targets):
        return self.val_step(params, inputs, targets)



# ==========================================
# train_model: usa Trainer (sin recompilar)
# ==========================================
def train_model(
    X_train, Y_train, X_test, Y_test, trainloader, testloader,
    original_dataset, run_name, dataset, seq_len, n_layers, n_qubits,
    concat_size, target_size, key, model, convergence=False,
    plot=False, return_all_hidden=False, trainer=None
):
    """
    Espera un 'trainer' creado afuera con la misma config y shapes de entrada.
    Así NO recompila al llamar por distintas seeds.
    """

    # ----------------- 1) Setup -----------------
    experiment_name = dataset
    _ = mlflow.get_experiment_by_name(experiment_name)

    # Input shape para init (asegúrate que coincide con batches reales)
    batch_size_for_init = 16
    sample_input = jnp.asarray(X_train[:batch_size_for_init, :, :], dtype=DTYPE)
    input_shape = sample_input.shape
    features = input_shape[2]

    # Crea net si trainer == None (modo retrocompatible, compilará una vez)
    if trainer is None:
        if model == 'QLSTM':
            net = QLSTM(seq_len, n_layers, n_qubits, concat_size, target_size, return_all_hidden=False)
        elif model == 'LSTM':
            net = LSTM(seq_len, features, concat_size, target_size)
        else:
            raise ValueError("Unknown model type. Choose 'QLSTM' or 'LSTM'.")
        trainer = Trainer(net, input_shape, lr=5e-4, use_checkpoint=True)

    # Cast datasets a DTYPE (si tus loaders devuelven ya jnp/DTYPE, esto no hace falta)
    X_train = jnp.asarray(X_train, dtype=DTYPE)
    Y_train = jnp.asarray(Y_train, dtype=DTYPE)
    X_test  = jnp.asarray(X_test,  dtype=DTYPE)
    Y_test  = jnp.asarray(Y_test,  dtype=DTYPE)

    # ----------------- 2) Init params/opt por SEED -----------------
    params, opt_state = trainer.init_params(key, input_shape)

    epochs = 200 if convergence else 25
    loss_history, val_loss_history = [], []
    previous_losses = []
    variance_threshold = 5e-6
    convergence_ep = None

    hidden_states_per_epoch = {}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('concat size', concat_size)
        mlflow.log_param('layers', n_layers)
        mlflow.log_param('qubits', n_qubits)
        mlflow.log_param('learning rate', trainer.lr)
        mlflow.log_param('init', 'xavier')
        mlflow.log_param('dtype', str(DTYPE))

        start_time = time.time()

        # ----------------- 3) Loop de épocas -----------------
        for epoch in range(epochs):
            print(f"\nStarting epoch {epoch + 1}")
            epoch_loss = 0.0

            # ---- Train ----
            for data in trainloader:
                inputs, targets = data[0].astype(DTYPE), data[1].astype(DTYPE)
                params, opt_state, loss = trainer.step_train(params, opt_state, inputs, targets)
                epoch_loss += inputs.shape[0] * loss

            epoch_loss = epoch_loss / len(X_train)
            loss_history.append(float(epoch_loss))

            # ---- Val ----
            val_loss = 0.0
            for val_data in testloader:
                val_inputs, val_targets = val_data[0].astype(DTYPE), val_data[1].astype(DTYPE)
                batch_val_loss, _ = trainer.step_val(params, val_inputs, val_targets)
                val_loss += batch_val_loss * val_inputs.shape[0]

            val_loss   = val_loss   / len(X_test)

            # Trae a host y castea a float (evita problemas con JAX Array)
            epoch_loss_host = float(jax.device_get(epoch_loss))
            val_loss_host   = float(jax.device_get(val_loss))

            loss_history.append(epoch_loss_host)
            val_loss_history.append(val_loss_host)
            previous_losses.append(val_loss_host)

            # Print info
            print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss_host:.6f}, Validation Loss = {val_loss_host:.6f}")


            # ---- Convergencia por varianza ----
            if convergence and len(previous_losses) >= 20:
                lv = np.var(previous_losses[-5:])
                print(f"Loss Variance (last 5 epochs): {lv:.8e}")
                if lv < variance_threshold:
                    print(f"Convergence achieved! Stopping at epoch {epoch+1}")
                    convergence_ep = epoch + 1
                    break

            # ---- Hidden states opcional (muestra pequeña) ----
            if return_all_hidden:
                sample_data = next(iter(testloader))
                sample_inputs, _ = sample_data
                sample_inputs = sample_inputs[:8].astype(DTYPE)
                preds, hidden_seq = trainer.net.apply(params, sample_inputs, return_all_hidden=True)
                hidden_states_per_epoch[epoch] = np.asarray(hidden_seq)

        total_time = time.time() - start_time
        print(f"\nTraining completed. Total time: {total_time:.2f} seconds")

        # ----------------- 4) Log & guardado -----------------
        mlflow.log_metric('training time', float(total_time))
        mlflow.log_metric('loss', float(epoch_loss))
        mlflow.log_metric('loss val', float(val_loss))
        if convergence_ep is not None:
            mlflow.log_metric('convergence epoch', int(convergence_ep))

        folder_name = dataset
        os.makedirs(folder_name, exist_ok=True)
        torch.save(np.array(loss_history), f'{folder_name}/LossT_{run_name}.pt')
        torch.save(np.array(val_loss_history), f'{folder_name}/LossV_{run_name}.pt')

        # ----------------- 5) Inference & métricas -----------------
        forecasted, targ = [], []
        for data in testloader:
            inputs, targets = data[0].astype(DTYPE), data[1].astype(DTYPE)
            p = trainer.net.apply(params, inputs)
            forecasted.append(np.array(p))
            targ.append(np.array(targets))

        forecasted = np.concatenate(forecasted, axis=0)
        targ = np.concatenate(targ, axis=0)

        resta_l = forecasted - targ
        n_points_test = len(Y_test)
        RMSE = np.sqrt(np.sum(np.square(resta_l)) / n_points_test)
        MAE = (np.sum(np.abs(resta_l)) / n_points_test)

        print('RMSE:', RMSE)
        print("MAE:", MAE)
        mlflow.log_metric('RMSE_TEST', float(RMSE))
        mlflow.log_metric('MAE_TEST', float(MAE))

        pearson_corr, _ = pearsonr(forecasted.flatten(), targ.flatten())
        print(f"Pearson Correlation (Test): {pearson_corr:.4f}")
        mlflow.log_metric('Pearson_corr', float(pearson_corr))

    return params, hidden_states_per_epoch if return_all_hidden else None
