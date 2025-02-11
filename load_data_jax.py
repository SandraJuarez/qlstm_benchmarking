
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import jax
#from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import collections
import jax.numpy as jnp
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
    if dataset=='sp500_2d': #este dataset es multidimensional
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
        
        scaled_data = np.zeros((3760,features,6))
        #en esta matriz vamos a ir guardando nuestras subsecuencias con el siguiente ciclo
        for t in range(4,3764):
            idx = t-4 # row index. We have 1000 rows (0-999)
            #el np.transpose se necesita para todos los multidimensionales
            scaled_data[idx,:,:] =np.transpose([scaled_data_sp500[t-4],scaled_data_sp500[t-3],scaled_data_sp500[t-2],
                                    scaled_data_sp500[t-1],scaled_data_sp500[t],scaled_data_sp500[t+1]])
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
        X_train = scaled_data_train[:,:,:5]  
        Y_train = scaled_data_train[:,:,5]
        X_test = scaled_data_test[:,:,:5]  
        Y_test = scaled_data_test[:,:,5]
    elif dataset=='sp500_3d': #este dataset es multidimensional
        #cargamos el csv
        data=np.genfromtxt('sp500.csv', delimiter=',',skip_header=1)
        #elegimos cuáles columnas son las que nos interesan
        data=data[0:3780,1:4]
        #data=np.transpose(data)
        #normalizamos los datos (dejarlos en valores entre 0 y 1)
        scaled_data_sp500 = (data - data.min(axis=0,keepdims=True)) / (data.max(axis=0,keepdims=True) - data.min(axis=0,keepdims=True))
        
        features=int(data.shape[1])
        print(features)
        #abrimos una matriz de tamaño números de muestras, features, tamaño de subsecuencia +1 
        # (el +1 es el dato que vamos a predecir, ese lo vamos a guardar después en otra variable)
        
        scaled_data = np.zeros((3760,features,6))
        #en esta matriz vamos a ir guardando nuestras subsecuencias con el siguiente ciclo
        for t in range(4,3764):
            idx = t-4 # row index. We have 1000 rows (0-999)
            #el np.transpose se necesita para todos los multidimensionales
            scaled_data[idx,:,:] =np.transpose([scaled_data_sp500[t-4],scaled_data_sp500[t-3],scaled_data_sp500[t-2],
                                    scaled_data_sp500[t-1],scaled_data_sp500[t],scaled_data_sp500[t+1]])
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
        X_train = scaled_data_train[:,:,:5]  
        Y_train = scaled_data_train[:,:,5]
        X_test = scaled_data_test[:,:,:5]  
        Y_test = scaled_data_test[:,:,5]
    elif dataset=='sp500_4d': #este dataset es multidimensional
        #cargamos el csv
        data=np.genfromtxt('sp500.csv', delimiter=',',skip_header=1)
        #elegimos cuáles columnas son las que nos interesan
        data=data[0:3780,1:5]
        #data=np.transpose(data)
        #normalizamos los datos (dejarlos en valores entre 0 y 1)
        scaled_data_sp500 = (data - data.min(axis=0,keepdims=True)) / (data.max(axis=0,keepdims=True) - data.min(axis=0,keepdims=True))
        
        features=int(data.shape[1])
        print(features)
        #abrimos una matriz de tamaño números de muestras, features, tamaño de subsecuencia +1 
        # (el +1 es el dato que vamos a predecir, ese lo vamos a guardar después en otra variable)
        
        scaled_data = np.zeros((3760,features,6))
        #en esta matriz vamos a ir guardando nuestras subsecuencias con el siguiente ciclo
        for t in range(4,3764):
            idx = t-4 # row index. We have 1000 rows (0-999)
            #el np.transpose se necesita para todos los multidimensionales
            scaled_data[idx,:,:] =np.transpose([scaled_data_sp500[t-4],scaled_data_sp500[t-3],scaled_data_sp500[t-2],
                                    scaled_data_sp500[t-1],scaled_data_sp500[t],scaled_data_sp500[t+1]])
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
        X_train = scaled_data_train[:,:,:5]  
        Y_train = scaled_data_train[:,:,5]
        X_test = scaled_data_test[:,:,:5]  
        Y_test = scaled_data_test[:,:,5]
    elif dataset=='sp500_5d': #este dataset es multidimensional
        #cargamos el csv
        data=np.genfromtxt('sp500.csv', delimiter=',',skip_header=1)
        #elegimos cuáles columnas son las que nos interesan
        data=data[0:3780,1:6]
        #data=np.transpose(data)
        #normalizamos los datos (dejarlos en valores entre 0 y 1)
        scaled_data_sp500 = (data - data.min(axis=0,keepdims=True)) / (data.max(axis=0,keepdims=True) - data.min(axis=0,keepdims=True))
        
        features=int(data.shape[1])
        print(features)
        #abrimos una matriz de tamaño números de muestras, features, tamaño de subsecuencia +1 
        # (el +1 es el dato que vamos a predecir, ese lo vamos a guardar después en otra variable)
        
        scaled_data = np.zeros((3760,features,6))
        #en esta matriz vamos a ir guardando nuestras subsecuencias con el siguiente ciclo
        for t in range(4,3764):
            idx = t-4 # row index. We have 1000 rows (0-999)
            #el np.transpose se necesita para todos los multidimensionales
            scaled_data[idx,:,:] =np.transpose([scaled_data_sp500[t-4],scaled_data_sp500[t-3],scaled_data_sp500[t-2],
                                    scaled_data_sp500[t-1],scaled_data_sp500[t],scaled_data_sp500[t+1]])
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
        X_train = scaled_data_train[:,:,:5]  
        Y_train = scaled_data_train[:,:,5]
        X_test = scaled_data_test[:,:,:5]  
        Y_test = scaled_data_test[:,:,5]
    elif dataset=='sp500': #este dataset es unidimensional
        data=np.genfromtxt('sp500.csv', delimiter=',',skip_header=1)
        data=data[0:3780,1:2] #solo queremos la columna 1
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
        X_train = np.expand_dims(X_train, axis=1)  
        Y_train = np.expand_dims(Y_train, axis=1) 
        X_test = np.expand_dims(X_test, axis=1)  
        Y_test = np.expand_dims(Y_test, axis=1)
        
    
        
        
    elif dataset=='mackey': #este dataset es unidimensional
        points=sample_len+28
        data = mackey_glass(points,seed=0)[0]
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
    return X_train,Y_train,X_test,Y_test,trainloader,testloader,data
    