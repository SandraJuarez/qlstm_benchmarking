
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import jax
#from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import collections
import jax.numpy as jnp
#from statsmodels.tsa.statespace.sarimax import SARIMAX
import jax_dataloader as jdl
np.random.seed(0)  


def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples = 1):
    '''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild 
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    delta_t = 10
    history_len = tau * delta_t 
    # Initial conditions for the history of the system
    timeseries = 1.2
    
    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len,1))
        
        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries
        
        # Squash timeseries through tanh
        # inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples

# class to construct a custom dataloader: 

def data(dataset,sample_len):
    scaler = MinMaxScaler()
    if dataset=='sp500_md': #este dataset es multidimensional
        #cargamos el csv
        data=np.genfromtxt('sp500.csv', delimiter=',',skip_header=1)
        #elegimos cuáles columnas son las que nos interesan
        data=data[0:3780,1:3]
        #data=np.transpose(data)
        #normalizamos los datos (dejarlos en valores entre 0 y 1)
        scaled_data_sp500 = (data - data.min(axis=0,keepdims=True)) / (data.max(axis=0,keepdims=True) - data.min(axis=0,keepdims=True))
        
        features=int(data.shape[1])
        print(features)
        #abrimos una matriz de tamaño números de muestras, features, tamaño de subsecuencia +1 
        # (el +1 es el dato que vamos a predecir, ese lo vamos a guardar después en otra variable)
        
        scaled_data = np.zeros((3760,6,features))
        #en esta matriz vamos a ir guardando nuestras subsecuencias con el siguiente ciclo
        for t in range(4,3764):
            idx = t-4 # row index. We have 1000 rows (0-999)
            #el np.transpose se necesita para todos los multidimensionales
            scaled_data[idx,:,:] =[scaled_data_sp500[t-4],scaled_data_sp500[t-3],scaled_data_sp500[t-2],
                                    scaled_data_sp500[t-1],scaled_data_sp500[t],scaled_data_sp500[t+1]]
            # We use the first 300 points to train the model and the rest 
            # for the testing phase:
        #una parte de los datos son para entrenamiento
        scaled_data_train = scaled_data[0:3000,:,0:6]
        #la otra parte de los datos son de test, sirven para validar que nuestro modelo funciona
        scaled_data_test = scaled_data[3000:,:,0:6]

        #Una parte de los datos son los puntos que vamos a agarrar para predecir el siguiente punto. 
        #estos se guardan en la variable X. Tenemos un X para entrenamiento y otro para test.
        #En la variable Y guardamos el dato que vamos a predecir. En el modelo comparamos el que predecimos con el que está guardado en Y
        #En la variable Y tenemos solamente un renglón de los datos. Las dimensiones que tendría es de (numero de muestras, features, 1)
        scaled_data_train = scaled_data[0:3000,0:6,:]
        scaled_data_test = scaled_data[3000:,0:6,:]
        X_train = scaled_data_train[:,:5,:]  
        Y_train = scaled_data_train[:,5,:]
        X_test = scaled_data_test[:,:5,:]  
        Y_test = scaled_data_test[:,5,:]
    elif dataset=='sp500': #este dataset es unidimensional
        data=np.genfromtxt('sp500.csv', delimiter=',',skip_header=1)
        data=data[0:3780,1:2] #solo queremos la columna 1
        features=int(data.shape[1])
        scaled_data_sp=(data-np.min(data))/(np.max(data)-np.min(data))
        #este es parecido al caso anterior pero como solo hay un feature (una sola columna), 
        # la matriz que abrimos no tiene esa dimensión extra que si tenía el caso anterior
        scaled_data = np.zeros((3760,6))
        for t in range(4,3764):
            idx = t-4 # row index. We have 1000 rows (0-999)
            scaled_data[idx,:] = np.transpose([scaled_data_sp[t-4],scaled_data_sp[t-3],scaled_data_sp[t-2],
                                    scaled_data_sp[t-1],scaled_data_sp[t],scaled_data_sp[t+1]])
            # We use the first 300 points to train the model and the rest 
            # for the testing phase:
        scaled_data_train = scaled_data[0:500,0:6]
        scaled_data_test = scaled_data[3000:,0:6]
        X_train = scaled_data_train[:,:5]  
        Y_train = scaled_data_train[:,5]
        X_test = scaled_data_test[:,:5]  
        Y_test = scaled_data_test[:,5]
        #esto que sigue lo tienen todos los unidimensionales
        # Añadir una dimensión extra para el feature
        X_train = np.expand_dims(X_train, axis=2)  
        Y_train = np.expand_dims(Y_train, axis=1) 
        X_test = np.expand_dims(X_test, axis=2)  
        Y_test = np.expand_dims(Y_test, axis=1)
        
    elif dataset=='btc_md': #este dataset es multidimensional
        data=np.genfromtxt('BTC.csv', delimiter=',',skip_header=1)
        data=data[:,1:-1]
        #features=4#int(data.shape[0])
        #data=np.transpose(data)
        scaled_data_btc=(data-np.min(data))/(np.max(data)-np.min(data))
        
        features=int(data.shape[1])
        scaled_data = np.zeros((3760,6,4))
        for t in range(4,3764):
            idx = t-4 # row index. 
            scaled_data[idx,:,:] =[scaled_data_btc[t-4],scaled_data_btc[t-3],scaled_data_btc[t-2],
                                    scaled_data_btc[t-1],scaled_data_btc[t],scaled_data_btc[t+1]]
          
        scaled_data_train = scaled_data[0:3000,0:6,:]
        scaled_data_test = scaled_data[3000:,0:6,:]
        X_train = scaled_data_train[:,:5,:]  
        Y_train = scaled_data_train[:,5,:]
        X_test = scaled_data_test[:,:5,:]  
        Y_test = scaled_data_test[:,5,:]
    
    elif dataset=='suma':
        """
        Generate data for the Adding Task.

        :param num_samples:  Number of sequences to generate
        :param seq_length:   Length of each sequence (number of time steps)
        :param seed:         Optional random seed for reproducibility
        :return: (X, y)
            X shape: (num_samples, seq_length, 2)
            y shape: (num_samples, 1)
        """
        seed=42
        num_samples = 1000
        seq_length=5
        if seed is not None:
            np.random.seed(seed)

        # X will hold the sequences: for each time step we have (value, marker).
        X = np.zeros((num_samples, seq_length, 2), dtype=np.float32)
        # y will hold the sum of the two selected values (one per sequence).
        y = np.zeros((num_samples, 1), dtype=np.float32)

        for i in range(num_samples):
            # 1) Random values in the first channel (feature 0)
            values = np.random.rand(seq_length).astype(np.float32)

            # 2) Select two random time steps to place the flags
            idxs = np.random.choice(seq_length, size=2, replace=False)
            idx1, idx2 = idxs[0], idxs[1]

            # Mark these positions in the second channel (feature 1)
            markers = np.zeros(seq_length, dtype=np.float32)
            markers[idx1] = 1.0
            markers[idx2] = 1.0

            # The target is the sum of the two chosen values
            target = values[idx1] + values[idx2]

            # Construct the full sequence (value, marker)
            X[i, :, 0] = values
            X[i, :, 1] = markers
            y[i, 0] = target
            #complete data
            data=X
            X_train = X[:750]
            Y_train = y[:750]
            X_test = X[750:]
            Y_test = y[750:]
            features=2
        
    elif dataset=='btc': #este dataset es unidimensional
        data=np.genfromtxt('BTC.csv', delimiter=',',skip_header=1)
        #print('sample',data[3,2])
        data=data[:,2]
        features=1#int(data.shape[1])
        #print('el shape de data',data.shape)
        #print('sample',data[3])
        scaled_data_btc=(data-np.min(data))/(np.max(data)-np.min(data))
        scaled_data = np.zeros((3760,6))
        for t in range(4,3764):
            idx = t-4 # row index. We have 1000 rows (0-999)
            scaled_data[idx,:] = [scaled_data_btc[t-4],scaled_data_btc[t-3],scaled_data_btc[t-2],
                                    scaled_data_btc[t-1],scaled_data_btc[t],scaled_data_btc[t+1]]
            # We use the first 300 points to train the model and the rest 
            # for the testing phase:
        scaled_data_train = scaled_data[0:3000,0:6]
        scaled_data_test = scaled_data[3000:,0:6]
        X_train = scaled_data_train[:,:5]  
        Y_train = scaled_data_train[:,5]
        X_test = scaled_data_test[:,:5]  
        Y_test = scaled_data_test[:,5]
         #esto que sigue lo tienen todos los unidimensionales
        # Añadir una dimensión extra para el feature
        X_train = np.expand_dims(X_train, axis=2)  
        Y_train = np.expand_dims(Y_train, axis=1) 
        X_test = np.expand_dims(X_test, axis=2)  
        Y_test = np.expand_dims(Y_test, axis=1)
        
    elif dataset=='euro': #este dataset es unidimensional
        data = np.genfromtxt('USD-EURO-Time-Series.csv', delimiter=',')
        features=1#int(data.shape[1])
        # The data is normalized is scaled in the range [0,1]
        scaled_data_usd_euro = (data-np.min(data))/(np.max(data)-np.min(data))#scaler.fit_transform(data.reshape(-1,1))

        scaled_data = np.zeros((376,6))
        for t in range(4,380):
            idx = t-4 # row index. We have 1000 rows (0-999)
            scaled_data[idx,:] = [scaled_data_usd_euro[t-4],scaled_data_usd_euro[t-3],scaled_data_usd_euro[t-2],
                                    scaled_data_usd_euro[t-1],scaled_data_usd_euro[t],scaled_data_usd_euro[t+1]]
            # We use the first 300 points to train the model and the rest 
            # for the testing phase:
        scaled_data_train = scaled_data[0:300,0:6]
        scaled_data_test = scaled_data[300:,0:6]
        
        X_train = scaled_data_train[:,:5]  
        Y_train = scaled_data_train[:,5]
        X_test = scaled_data_test[:,:5]  
        Y_test = scaled_data_test[:,5]  
         #esto que sigue lo tienen todos los unidimensionales
        # Añadir una dimensión extra para el feature
        X_train = np.expand_dims(X_train, axis=2)  
        Y_train = np.expand_dims(Y_train, axis=1) 
        X_test = np.expand_dims(X_test, axis=2)  
        Y_test = np.expand_dims(Y_test, axis=1)

    
        
    
        
        
    elif dataset=='mackey': #este dataset es unidimensional
        points=sample_len+28
        data = mackey_glass(points,seed=0)[0]
        features=int(data.shape[1])
        scaled_data_mackey_glass = (data-np.min(data))/(np.max(data)-np.min(data)) #scaler.fit_transform(data)
        datasize=sample_len-18
        print(data.shape)
        scaled_data = np.zeros((datasize,5))
        for t in range(18,sample_len):
            idx = t-18 # row index. We have 1000 rows (0-999)
            
            scaled_data[idx,:] = [scaled_data_mackey_glass[t-18][0], scaled_data_mackey_glass[t-12][0],
                                    scaled_data_mackey_glass[t-6][0], scaled_data_mackey_glass[t][0],
                                    scaled_data_mackey_glass[t+6][0]]
        # We use the first 500 points to train the model and the rest 
        # for the testing phase:
        #scaled_data_train = scaled_data[0:datasize,0:5]
        #scaled_data_test = scaled_data[datasize:,0:5]
        scaled_data_train = scaled_data[0:500,0:5]
        scaled_data_test = scaled_data[750:,0:5]
        X_train = scaled_data_train[:,:4]  
        
        Y_train = scaled_data_train[:,4]
        X_test = scaled_data_test[:,:4]  
        Y_test = scaled_data_test[:,4]
        #hacemos un reshape a (puntos, features,sequence_lenght)
         #esto que sigue lo tienen todos los unidimensionales
        # Añadir una dimensión extra para el feature
        X_train = np.expand_dims(X_train, axis=2)  
        Y_train = np.expand_dims(Y_train, axis=1) 
        X_test = np.expand_dims(X_test, axis=2)  
        Y_test = np.expand_dims(Y_test, axis=1)
        
        #X_train=X_train.reshape(500,1,4)
        #Y_train=Y_train.reshape(500,1)

        #X_test=X_test.reshape(250,1,4)
        #Y_test=Y_test.reshape(250,1)
    elif dataset == 'frequencies':
        sample_len = 1000
        noise_std = 0.05  # 0.05 for consistent noise

        # Lista de frecuencias a incluir (todas con la misma amplitud)
        freqs = [0.01, 0.05, 0.1, 0.2, 0.4]
        amplitude = 1.0

        t = np.arange(sample_len)
        
        # Sumar componentes sinusoidales de todas las frecuencias
        composite_signal = np.zeros(sample_len)
        for f in freqs:
            composite_signal += amplitude * np.sin(2 * np.pi * f * t)
        
        # Agregar ruido gaussiano
        noise = np.random.normal(loc=0.0, scale=noise_std, size=sample_len)
        data = composite_signal + noise

        # Redimensionar para consistencia
        data = data.reshape(-1, 1)

        # Escalar los datos sintéticos a [0, 1]
        data_min = data.min()
        data_max = data.max()
        scaled_data_ts = (data - data_min) / (data_max - data_min)

        # Construir los datos con ventanas: [t-4, t-3, t-2, t-1, t, t+1]
        datasize = sample_len - 4 - 1  # desde t=4 hasta t=998
        scaled_data = np.zeros((datasize, 6))
        for t in range(4, sample_len - 1):
            idx = t - 4
            scaled_data[idx, :] = [
                scaled_data_ts[t - 4, 0], scaled_data_ts[t - 3, 0], scaled_data_ts[t - 2, 0],
                scaled_data_ts[t - 1, 0], scaled_data_ts[t, 0], scaled_data_ts[t + 1, 0]
            ]

        # División ejemplo: 600 entrenamiento, 300 prueba
        scaled_data_train = scaled_data[:600]
        scaled_data_test = scaled_data[600:900]

        X_train = scaled_data_train[:, :5]
        Y_train = scaled_data_train[:, 5]
        X_test = scaled_data_test[:, :5]
        Y_test = scaled_data_test[:, 5]
        features = 1

        # Ajustar dimensiones para LSTM
        X_train = np.expand_dims(X_train, axis=2)  # (samples, 5, 1)
        Y_train = np.expand_dims(Y_train, axis=1)
        X_test = np.expand_dims(X_test, axis=2)
        Y_test = np.expand_dims(Y_test, axis=1)

    elif dataset=='seasons':
        sample_len=1000
        periods=[24,168]  # Two seasonalities
        amps=[1.0,0.5]    # Different amplitudes for each seasonality
        noise_std=0.1 
    
        
        points = 1001
        
        # Time indices
        t = np.arange(points)
        
        # Sum up multiple seasonal components
        # e.g., if periods=[24,168], amps=[1.0,0.5], 
        # we get daily & weekly cycles with different amplitudes.
        signal = np.zeros(points)
        for p in range(len(periods)):
            per=periods[p]
            amp=amps[p]
            print(per)
            print
            signal += amp * np.sin(2 * np.pi * t / per)
        
        # Add Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_std, size=points)
        
        # Combine seasonal signal + noise
        data = signal + noise
        
        # Reshape to (points, 1) for consistent scaling logic
        data = data.reshape(-1, 1)

        # Scale to [0,1]
        data_min = data.min()
        data_max = data.max()
        scaled_data_ts = (data - data_min) / (data_max - data_min)

        # Build array with 5 columns: [t-18, t-12, t-6, t, t+6] 
        datasize = sample_len - 4  # number of valid rows
        scaled_data = np.zeros((datasize, 6))

        for i in range(4, sample_len):
            idx = i - 4
            scaled_data[idx, :] = [
                scaled_data_ts[i - 4, 0],
                scaled_data_ts[i - 3, 0],
                scaled_data_ts[i - 2, 0],
                scaled_data_ts[i - 1, 0],  # <-- here's the missing reference
                scaled_data_ts[i, 0],
                scaled_data_ts[i + 1, 0]
            ]

        # Split into train/test
        # Example: first 500 points for train, last 250 for test (adjust as needed)
        train_end = 500
        test_start = 750

        scaled_data_train = scaled_data[:train_end, :]
        scaled_data_test  = scaled_data[test_start:, :]
        features=1

        X_train = scaled_data_train[:, :4]  # t-18, t-12, t-6, t
        Y_train = scaled_data_train[:, 4]   # t+6
        X_test  = scaled_data_test[:, :4]
        Y_test  = scaled_data_test[:, 4]

        # Reshape to match your code’s requirement
        X_train = np.expand_dims(X_train, axis=2)  # (samples, 4, 1)
        Y_train = np.expand_dims(Y_train, axis=1)  # (samples, 1)
        X_test  = np.expand_dims(X_test, axis=2)   # (samples, 4, 1)
        Y_test  = np.expand_dims(Y_test, axis=1)   # (samples, 1)
  
    
        
    elif dataset=='legendre3':#este dataset es unidimensional
        np.random.seed(42)
        x_values = np.linspace(0, 100, 1000)
        # Calculate the third Legendre polynomial without noise
        legendre_poly = 1/8 * (3 * np.cos(x_values) - 5*np.cos(3*x_values))
        # Add seeded random noise to the polynomial
        noise = np.random.normal(0, 0.1, size=len(x_values))
        legendre_poly_with_noise = legendre_poly + noise
        # Store the points in a NumPy array
        data = np.squeeze(np.column_stack((legendre_poly_with_noise)))
        features=1#int(data.shape[1])
        scaled_data_legendre = (data-np.min(data))/(np.max(data)-np.min(data))#scaler.fit_transform(data.reshape(-1,1))
        scaled_data = np.zeros((996,6))
        for t in range(4,999):
            idx = t-4 # row index. We have 1000 rows (0-999)
            scaled_data[idx,:] = [scaled_data_legendre[t-4],scaled_data_legendre[t-3],scaled_data_legendre[t-2],
                                    scaled_data_legendre[t-1],scaled_data_legendre[t],scaled_data_legendre[t+1]]
        # We use the first 300 points to train the model and the rest 
        # for the testing phase:
        scaled_data_train = scaled_data[0:750,0:6]
        scaled_data_test = scaled_data[750:,0:6]
        X_train = scaled_data_train[:,:5]  
        Y_train = scaled_data_train[:,5]
        X_test = scaled_data_test[:,:5]  
        Y_test = scaled_data_test[:,5]
         #esto que sigue lo tienen todos los unidimensionales
        # Añadir una dimensión extra para el feature
        X_train = np.expand_dims(X_train, axis=2)  
        Y_train = np.expand_dims(Y_train, axis=1) 
        X_test = np.expand_dims(X_test, axis=2)  
        Y_test = np.expand_dims(Y_test, axis=1)

    
        


    #train_data = CustomDataset(X_train, Y_train)
    #test_data = CustomDataset(X_test, Y_test)
    train_data = jdl.ArrayDataset(X_train, Y_train)
    test_data=jdl.ArrayDataset(X_test, Y_test)
    # Dataloaders
    #print('shape de traindata antes de trainloader',train_data.shape)
    trainloader=jdl.DataLoader(
    train_data, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
    backend='jax', # Use 'jax' backend for loading data
    batch_size=16, # Batch size 
    shuffle=False, # Shuffle the dataloader every iteration or not
    drop_last=True, # Drop the last batch or not
    )
    
    testloader=jdl.DataLoader(
    test_data, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
    backend='jax', # Use 'jax' backend for loading data
    batch_size=16, # Batch size 
    shuffle=False, # Shuffle the dataloader every iteration or not
    drop_last=True, # Drop the last batch or not
    )
    #print('un sample de xtest',X_test[0,:,:])
    return X_train,Y_train,X_test,Y_test,trainloader,testloader,data,features
    
    