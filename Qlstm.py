import pennylane as qml
import jax
import jax.numpy as jnp
import flax.linen as nn
from pennylane.templates import RandomLayers
from pennylane.templates import StronglyEntanglingLayers
from pennylane.templates import BasicEntanglerLayers
import jax_dataloader as jdl
import numpy as np

class QLSTM(nn.Module):
    seq_lenght:int
    n_qlayers:int
    n_qubits:int
    hidden_size:int
    target_size:int
    return_all_hidden:bool
    
    def setup(self):
        self.weightsf=self.param('weightsf',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        self.weightsi=self.param('weightsi',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        self.weightsu=self.param('weightsu',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        
        self.weightso=self.param('weightso',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        #self.weightsf=self.param('weightsf',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
        #self.weightsi=self.param('weightsi',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
        #self.weightsu=self.param('weightsu',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
        #self.weightso=self.param('weightso',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
    def circuit_forget(self,inputs,weights):
        device = qml.device('default.qubit')
        wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        #self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        #self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        #self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]
        @qml.qnode(device=device)
        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=wires_forget)
            pesos=weights[0,:]
            pesos=pesos.reshape(1,-1)
            qml.templates.BasicEntanglerLayers(pesos, wires=wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires_forget]
        _circuit_forget=jax.jit(_circuit_forget)
        out=_circuit_forget(inputs,weights)
        return out
    
    def circuit_input(self,inputs,weights):
        wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        device = qml.device('default.qubit')
        @qml.qnode(device=device)
        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=wires_input)
            pesos=weights[0,:]
            pesos=pesos.reshape(1,-1)
            qml.templates.BasicEntanglerLayers(pesos, wires=wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires_input]
        _circuit_input=jax.jit(_circuit_input)
        out=_circuit_input(inputs,weights)
        return out
    
    def circuit_update(self,inputs,weights):
        wires_update = [f"wire_input_{i}" for i in range(self.n_qubits)]
        device = qml.device('default.qubit')
        @qml.qnode(device=device)
        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=wires_update)
            pesos=weights[0,:]
            pesos=pesos.reshape(1,-1)
            qml.templates.BasicEntanglerLayers(pesos, wires=wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires_update]
        _circuit_update=jax.jit(_circuit_update)
        out=_circuit_update(inputs,weights)
        return out
    def circuit_output(self,inputs,weights):
        wires_output = [f"wire_input_{i}" for i in range(self.n_qubits)]
        device = qml.device('default.qubit')
        @qml.qnode(device=device)
        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=wires_output)
            pesos=weights[0,:]
            pesos=pesos.reshape(1,-1)
            qml.templates.BasicEntanglerLayers(pesos, wires=wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires_output]
        _circuit_output=jax.jit(_circuit_output)
        out=_circuit_output(inputs,weights)
        return out
    
    @nn.compact
    def __call__(self, x, init_states=None,return_all_hidden=False):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
       
        hidden_seq = []
        batch_size=x.shape[0]
        if init_states is None:
            h_t = jnp.zeros((batch_size, self.hidden_size))  # hidden state (output)
            c_t = jnp.zeros((batch_size, self.hidden_size)) # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(self.seq_lenght):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :] #x has shape (batch,seq_len,features)
          
            # Concatenate input and hidden state
            v_t = jnp.concatenate((h_t, x_t), axis=1)
            # match qubit dimension
            y_t = nn.Dense(self.n_qubits)(v_t) #Dense gives an output of n_qubits
            f_t=self.circuit_forget(y_t,self.weightsf)
            f_t=jnp.asarray(f_t)
            f_t = nn.sigmoid(f_t)  # forget block
            f_t=jnp.transpose(f_t)
            f_t=nn.Dense(self.hidden_size)(f_t)
            i_t = self.circuit_input(y_t,self.weightsi)  # input block
            i_t=jnp.asarray(i_t)
            i_t=nn.sigmoid(i_t)
            i_t=jnp.transpose(i_t)
            i_t=nn.Dense(self.hidden_size)(i_t)
            g_t = self.circuit_update(y_t,self.weightsu) # update block
            g_t=jnp.asarray(g_t)
            g_t=jnp.tanh(g_t)
            g_t=jnp.transpose(g_t)
            g_t=nn.Dense(self.hidden_size)(g_t)
            o_t = self.circuit_output(y_t,self.weightso)# output block
            
            o_t=jnp.asarray(o_t)
            o_t=nn.sigmoid(o_t)
            o_t=jnp.transpose(o_t)
            o_t=nn.Dense(self.hidden_size)(o_t)
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * nn.tanh(c_t) #it has size (batch_size, hidden)
            hidden_seq.append(jnp.expand_dims(h_t, axis=0))#we will end with a number of sequences of the size of the window of time 
                                 
        hidden_seq = jnp.concatenate(hidden_seq, axis=0) #(window, batch_size,hidden)
        hidden_seq = hidden_seq.transpose(1, 0, 2)  #(batch_size,window,hidden)
        final_hidden_seq=hidden_seq[:, -1, :]
        target=nn.Dense(self.target_size)(final_hidden_seq)
        if return_all_hidden:
            return target, hidden_seq
        else:
            return target