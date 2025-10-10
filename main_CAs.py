from train import train_model
import load_data_jax_metrics
from train_memory_efficient import train_model, Trainer
import importlib 
import numpy as np
import time
from Qlstm import QLSTM
from LSTM import ClassicalLSTM as LSTM
# Definimos los experimentos de alto costo a ejecutar
'''
experiments_to_run = [
    # Combinaciones para hidden=16 y ansatz=random
    {'hidden': 32, 'qubits': 16, 'ansatz': 'random', 'initialization': 'normal'},
    {'hidden': 32, 'qubits': 18, 'ansatz': 'random', 'initialization': 'normal'},
    {'hidden': 32, 'qubits': 20, 'ansatz': 'random', 'initialization': 'normal'},
    
    # Combinación específica para hidden=32 y qubits=20
    # Esta compensa el cambio que propones para cubrir el caso de ansatz='basic'
    {'hidden': 16, 'qubits': 18, 'ansatz': 'basic', 'initialization': 'xavier'},
]
'''
experiments_to_run = [
    # Combinaciones para hidden=16 y ansatz=random
    {'hidden': 16, 'qubits': 12, 'ansatz': 'basic', 'initialization': 'xavier'},
    {'hidden': 16, 'qubits': 14, 'ansatz': 'basic', 'initialization': 'xavier'},
    {'hidden': 16, 'qubits': 16, 'ansatz': 'basic', 'initialization': 'xavier'},
    {'hidden': 16, 'qubits': 18, 'ansatz': 'basic', 'initialization': 'xavier'},
    {'hidden': 16, 'qubits': 20, 'ansatz': 'basic', 'initialization': 'xavier'}
]
# Parámetros fijos
dataset = 'sp500'
seq_len = 5
target_size = 1
kernel_size = 2
model = "QLSTM"
convergence = False
points = 1518  # Número de puntos a cargar
architecture = 'super_parallel'  # Opciones: no_reupload, parallel, super_parallel

# Semillas para la reproducibilidad
key_list = [0]

# Bucle principal para ejecutar cada experimento del MLCA
times={}
for exp in experiments_to_run:
    hidden = exp['hidden']
    qubits = exp['qubits']
    ansatz = exp['ansatz']
    initializer = exp['initialization']
    X_train, Y_train, X_test, Y_test, trainloader, testloader, data, features = (
            load_data_jax_metrics.data(dataset, points)
        )
    
    # Asigna el tipo de inicialización según la lógica de tu proyecto
    if initializer == 'xavier':
        initializer_type = 'xavier_uniform'
    elif initializer == 'normal':
        initializer_type = 'normal'
    else:
        # Aquí puedes manejar cualquier otro caso
        initializer_type = initializer

    # Determina n_layers
    if architecture == 'super_parallel':
        n_layers = qubits // kernel_size
    elif architecture in ('parallel', 'no_reupload'):
        n_layers = 4
    else:
        raise ValueError("architecture inválida")
    
    # === Construye el modelo y el Trainer UNA SOLA VEZ para la configuración actual ===
    if model == "QLSTM":
        net = QLSTM(seq_len, n_layers, qubits, hidden, target_size,initializer_type)
    elif model == "LSTM":
        # Nota: Aquí falta 'features' en tu llamada original.
        net = LSTM(seq_len, features, hidden, target_size)
    else:
        raise ValueError("Unknown model")
    
    # Compila el trainer UNA SOLA VEZ
    batch_init = 16
    input_shape = (batch_init, seq_len, features)
    trainer = Trainer(net, input_shape, lr=5e-4, use_checkpoint=True)

    # Estructura para tiempos por seed
    key_times = []

    # === Ahora sí: Bucle de SEEDS ===
    for key in key_list:
        run_name = f"{dataset}_{hidden}_{qubits}_{key}_{ansatz}_{initializer}"

        t0 = time.time()
        train_model(
            X_train, Y_train, X_test, Y_test,
            trainloader, testloader, data,
            run_name, dataset, seq_len, n_layers, qubits,
            hidden, target_size, key, model, initializer_type,
            convergence=False, plot=False, return_all_hidden=False,
            trainer=trainer
        )
        key_times.append(time.time() - t0)

    # Guarda los tiempos por configuración
    cfg_key = (dataset, points, qubits, hidden, ansatz, initializer)
    times[cfg_key] = key_times
    print(f"Tiempos para {cfg_key}: {key_times}")