import pennylane as qml
import jax
import jax.numpy as jnp
import flax.linen as nn
from pennylane.templates import RandomLayers
from pennylane.templates import StronglyEntanglingLayers
from pennylane.templates import BasicEntanglerLayers
import jax_dataloader as jdl
import numpy as np



class ClassicalLSTM(nn.Module):
    seq_length: int
    hidden_size: int
    target_size: int

    def setup(self):
        self.W_f = self.param('W_f', nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size + self.target_size))
        self.b_f = self.param('b_f', nn.initializers.zeros, (self.hidden_size,))
        self.W_i = self.param('W_i', nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size + self.target_size))
        self.b_i = self.param('b_i', nn.initializers.zeros, (self.hidden_size,))
        self.W_c = self.param('W_c', nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size + self.target_size))
        self.b_c = self.param('b_c', nn.initializers.zeros, (self.hidden_size,))
        self.W_o = self.param('W_o', nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size + self.target_size))
        self.b_o = self.param('b_o', nn.initializers.zeros, (self.hidden_size,))
        self.fc = nn.Dense(self.target_size)
    
    @nn.compact
    def __call__(self, x, init_states=None):
        batch_size = x.shape[0]
        hidden_seq = []

        if init_states is None:
            h_t = jnp.zeros((batch_size, self.hidden_size))
            c_t = jnp.zeros((batch_size, self.hidden_size))
        else:
            h_t, c_t = init_states

        for t in range(self.seq_length):
            x_t = x[:, t, :]
            v_t = jnp.concatenate((h_t, x_t), axis=1)
            
            f_t = nn.sigmoid(jnp.dot(v_t, self.W_f.T) + self.b_f)
            i_t = nn.sigmoid(jnp.dot(v_t, self.W_i.T) + self.b_i)
            g_t = jnp.tanh(jnp.dot(v_t, self.W_c.T) + self.b_c)
            o_t = nn.sigmoid(jnp.dot(v_t, self.W_o.T) + self.b_o)
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * jnp.tanh(c_t)
            
            hidden_seq.append(jnp.expand_dims(h_t, axis=0))
        
        hidden_seq = jnp.concatenate(hidden_seq, axis=0).transpose(1, 0, 2)
        hidden_seq = hidden_seq[:, -1, :]
        target = self.fc(hidden_seq)
        return target
