
import load_data_jax_metrics
from train_memory_efficient import train_model, Trainer
from Qlstm import QLSTM
from LSTM import ClassicalLSTM as LSTM

import numpy as np
import time
import gc, jax

# 2) Limpia cachés internas de ejecutables (no siempre reduce VRAM, pero ayuda)
jax.clear_caches()




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




# Opcional: caché persistente de compilación entre ejecuciones
# from jax.experimental.compilation_cache import compilation_cache
# compilation_cache.initialize_cache(os.path.expanduser("~/.jax_compilation_cache"))

times = {}  # paso a dict para soportar todas las combinaciones

for d, dataset in enumerate(dataset_list):
    seq_len = sample_size_list[d]

    for p, points in enumerate(point_list):
        # Carga de datos depende de dataset y points → cambia shapes/longitudes
        X_train, Y_train, X_test, Y_test, trainloader, testloader, data, features = (
            load_data_jax_metrics.data(dataset, points)
        )
        print("features:", features)
        target_size = 1

        for q, n_qubits in enumerate(qubits_list):
            if architecture == 'super_parallel':
                n_layers = n_qubits // kernel_size
            elif architecture in ('parallel', 'no_reupload'):
                n_layers = 4
            else:
                raise ValueError("architecture inválida")

            for h, concat_size in enumerate(hidden_list):
                # === Construye el modelo y el Trainer UNA sola vez por config ===
                if model == "QLSTM":
                    net = QLSTM(seq_len, n_layers, n_qubits, concat_size, target_size, return_all_hidden=False)
                elif model == "LSTM":
                    net = LSTM(seq_len, features, concat_size, target_size)
                else:
                    raise ValueError("Unknown model")

                # Usa un batch fijo para inicializar/compilar (coincidir con tus loaders)
                batch_init = 16
                input_shape = (batch_init, seq_len, features)
                trainer = Trainer(net, input_shape, lr=5e-4, use_checkpoint=True)  # <-- compila 1 vez

                # Estructura para tiempos por seed
                key_times = []

                # === Ahora sí: SEEDS ADENTRO → reusa compilación ===
                for k, key in enumerate(key_list):
                    run_name = f"{dataset}{concat_size}{n_qubits}{key}small8Q25"

                    t0 = time.time()
                    train_model(
                        X_train, Y_train, X_test, Y_test,
                        trainloader, testloader, data,
                        run_name, dataset, seq_len, n_layers, n_qubits,
                        concat_size, target_size, key, model,
                        convergence=False, plot=False, return_all_hidden=False,
                        trainer=trainer       # <--- reusa train_step/val_step compilados
                    )
                    key_times.append(time.time() - t0)

                # Guarda tiempos por configuración
                cfg_key = (dataset, points, n_qubits, n_layers, concat_size)
                times[cfg_key] = key_times

# (opcional) convertir times a algo tabular si quieres guardar CSV
