#!/usr/bin/env python
# coding: utf-8

# ## _*Quantum Fourier Transform for Machine Learning*_ 
# 
# ***
# ### Contributors:
# **ML Use Case**: 2020 - Nicole Barberis
# 
# **Template**: 2018 - Anna Phan

# ## Overview
# 
# #### 1) Use QFT to create informative features for a classical machine learning task.  Will some nuance be revealed?
# 
# #### 2) Output the state vectors for us in a classical machine learning task.
# 
# #### 3) Show that the QFT features add value.

# # Implementation <a id='implementation'></a>

# In[1]:


# Importing QISKit
import math
from qiskit import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute

# Import basic plotting tools
from qiskit.tools.visualization import plot_histogram
import qiskit.tools.jupyter  # This is the where the magic happens (literally).


# ## Setup API Services

# In[2]:


IBMQ.load_account();


# In[3]:


provider = IBMQ.get_provider(group='open')


# In[4]:


provider.backends()


# In[5]:


for backend in provider.backends():
    print( backend.status() )


# In[6]:


#set backend

backend = provider.get_backend('ibmq_essex')
#backend = provider.get_backend('ibmqx2')
#backend = Aer.get_backend('ibmq_qasm_simulator')
#backend = provider.get_backend('ibmqx2')
#backend = Aer.get_backend('qasm_simulator')

#Run the quantum circuit on a statevector simulator backend
backend2 = Aer.get_backend('statevector_simulator')


# ## First let's define the QFT function, as well as a function that creates a state from which a QFT will return 1:

# In[ ]:


#circuit.initialize(desired_vector, [q[0],q[1],q[2]])


# In[7]:


def input_state(circ, q, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(j)
        circ.u1(math.pi/float(2**(j)), j).inverse()


# In[8]:


def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), j, k)
        circ.h(j)


# Let's now implement a QFT on a prepared three qubit input state that should return $001$:

# ## Using custom functions

# In[9]:


#create circuit
set_size = 12
q = QuantumRegister(set_size)
c = ClassicalRegister(set_size)
qc = QuantumCircuit(q, c)


# In[10]:


#input
input_state(qc, q, set_size)


# In[11]:


#QFT
qft(qc, q, set_size)


# In[12]:


#measurement
for i in range(set_size):
    qc.measure(q[i], c[i])

print(qc.qasm())


# In[13]:


qc.draw()


# ## There are 2 circuits:  qc and nc.  The nc circuit is the transpiled one.

# In[15]:


#transpile: set couplings based on the configration of the backend
#new_circuits = transpile(qc, backend, optimization_level=1)

#nc = transpile(qc, backend, optimization_level=1)


# In[ ]:


nc.draw()


# ## Execute Jobs

# In[17]:


circ = qc
#state vectors
#job = execute(circ, backend, shots=1024)
job = execute(circ, backend2)
result = job.result()

print("QFT Results: ", result)
#print(result.get_counts(circ))


# In[19]:


#ONLY for state vector simulation
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)


# In[ ]:


#ONLY for state vector simulation
from qiskit.visualization import plot_state_city
plot_state_city(outputstate)


# In[ ]:


#real QPU
#job_sim = execute(qc, backend)
job = execute(qc, backend, shots=1024)
result = job.result()

print("QFT Results: ", result)
print(result.get_counts(qc))


# In[ ]:


counts = result.get_counts(qc)
print(counts)


# In[ ]:


#Real QPU - After transpiling
job = execute(nc, backend, shots=1024)
result = job.result()

print("QFT Results: ", result)
print(result.get_counts(nc))


# In[ ]:


counts2 = result.get_counts(nc)
print(counts2)


# In[ ]:


#job_sim2 = execute(["qft3"], backend, coupling_map=ibmqx2_coupling, shots=1024, max_credits=3, wait=10, timeout=240)


# In[ ]:


#plot_histogram(counts)
plot_histogram(counts)


# In[ ]:


plot_histogram(counts2)


# In[ ]:


qc.width()


# In[ ]:


qc.n_qubits


# In[ ]:


qc.count_ops()


# In[ ]:


# We can also get just the raw count of operations by computing the circuits size:
qc.size()


# ## Notes and Tests

# In[ ]:


#QFT circuit
circ = QuantumCircuit(3)


# In[ ]:


circ.h(0)
circ.u1(-3.141592653589793,0)
circ.h(1)
circ.u1(-1.570796326794897, 1)
circ.h(2)
circ.u1(-0.785398163397448,2)
circ.h(0)
circ.cu1(1.570796326794897, 1,0)
circ.h(1)
circ.cu1(0.785398163397448,2,0)
circ.cu1(1.570796326794897,2,1)
circ.h(1)

meas = QuantumCircuit(3, 3)
meas.measure(range(3),range(3))
qc = circ+meas


# In[ ]:


from qiskit.visualization import plot_histogram
plot_histogram(counts)


# In[ ]:


#input state on the QFT
input_state(qft3, q, 3)


# In[ ]:


#transformation
qft(qft3, q, 3)


# In[ ]:


#measurement
for i in range(3):
    qft3.measure(q[i], c[i])
print(qft3.qasm())


# In[ ]:


simulate = Q_program.execute(["qft3"], backend="local_qasm_simulator", shots=1024)
simulate.get_counts("qft3")


# In[ ]:


#redo this in more current Qiskit version language
q = Q_program.create_quantum_register("q", 3)
c = Q_program.create_classical_register("c", 3)
qft3 = Q_program.create_circuit("qft3", [q], [c])

input_state(qft3, q, 3)
qft(qft3, q, 3)

for i in range(3):
    qft3.measure(q[i], c[i])
print(qft3.qasm())

simulate = Q_program.execute(["qft3"], backend="local_qasm_simulator", shots=1024)
simulate.get_counts("qft3")


# We indeed see that the outcome is always $001$ when we execute the code on the simulator.
# 
# Note that as written, it is not possible to run the code on either `ibmqx2` or `ibmqx3`, as the qubit couplings used don't exist. So we'll need to get the `ibmqx2` coupling map and use that.

# In[ ]:


ibmqx2_backend = Q_program.get_backend_configuration('ibmqx2')
ibmqx2_coupling = ibmqx2_backend['coupling_map']

run = Q_program.execute(["qft3"], backend="ibmqx2", coupling_map=ibmqx2_coupling, shots=1024, max_credits=3, wait=10, timeout=240)
plot_histogram(run.get_counts("qft3"))


# We see that the highest probability outcome $(00)001$ when we execute the code on `ibmqx2`.

# In[ ]:


get_ipython().run_line_magic('run', '"../version.ipynb"')


# ## Backup Code

# In[ ]:


#https://qiskit-staging.mybluemix.net/documentation/release_notes.html?highlight=quantumprogram

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute

q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)

qc.h(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)

backend = get_backend('qasm_simulator')

job_sim = execute(qc, backend)
sim_result = job_sim.result()

print("simulation: ", sim_result)
print(sim_result.get_counts(qc))


# # Education

# In this tutorial, we [introduce](#introduction) the quantum fourier transform (QFT), [derive](#circuit) the circuit, QASM and QISKit code, before [implementing](#implementation) it using the simulator and five qubit device.
# 
# The latest version of this notebook is available on https://github.com/QISKit/qiskit-tutorial.

# ## Introduction <a id='introduction'></a>
# 
# The Fourier transform occurs in many different versions throughout classical computing, in areas ranging from signal processing to data compression to complexity theory. The quantum Fourier transform (QFT) is the quantum implementation of the discrete Fourier transform over the amplitudes of a wavefunction. It is part of many quantum algorithms, most notably Shor's factoring algorithm and quantum phase estimation. 

# The discrete Fourier transform acts on a vector $(x_0, ..., x_{N-1})$ and maps it to the vector $(y_0, ..., y_{N-1})$ according to the formula
# $$y_k = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1}x_j\omega_N^{jk}$$
# where $\omega_N^{jk} = e^{2\pi i \frac{jk}{N}}$.
# 
# Similarly, the quantum Fourier transform acts on a quantum state $\sum_{i=0}^{N-1} x_i \vert i \rangle$ and maps it to the quantum state $\sum_{i=0}^{N-1} y_i \vert i \rangle$ according to the formula
# $$y_k = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1}x_j\omega_N^{jk}$$
# with $\omega_N^{jk}$ defined as above. Note that only the amplitudes of the state were affected by this transformation.
# 
# This can also be expressed as the map:
# $$\vert x \rangle \mapsto \frac{1}{\sqrt{N}}\sum_{y=0}^{N-1}\omega_N^{xy} \vert y \rangle$$
# 
# Or the unitary matrix:
# $$ U_{QFT} = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} \omega_N^{xy} \vert y \rangle \langle x \vert$$

# ## Circuit and Code <a id='circuit'></a>
# 
# We've actually already seen the quantum Fourier transform for when $N = 2$, it is the Hadamard operator ($H$):
# $$H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$
# Suppose we have the single qubit state $\alpha \vert 0 \rangle + \beta \vert 1 \rangle$, if we apply the $H$ operator to this state, we obtain the new state:
# $$\frac{1}{\sqrt{2}}(\alpha + \beta) \vert 0 \rangle + \frac{1}{\sqrt{2}}(\alpha - \beta)  \vert 1 \rangle 
# \equiv \tilde{\alpha}\vert 0 \rangle + \tilde{\beta}\vert 1 \rangle$$
# Notice how the Hadamard gate performs the discrete Fourier transform for $N = 2$ on the amplitudes of the state. 

# So what does the quantum Fourier transform look like for larger N? Let's derive a circuit for $N=2^n$, $QFT_N$ acting on the state $\vert x \rangle = \vert x_1...x_n \rangle$ where $x_1$ is the most significant bit.
# 
# \begin{aligned}
# QFT_N\vert x \rangle & = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1}\omega_N^{xy} \vert y \rangle \\
# & = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{2 \pi i xy / 2^n} \vert y \rangle \:\text{since}\: \omega_N^{xy} = e^{2\pi i \frac{xy}{N}} \:\text{and}\: N = 2^n\\
# & = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{2 \pi i \left(\sum_{k=1}^n y_k/2^k\right) x} \vert y_1 ... y_n \rangle \:\text{rewriting in fractional binary notation}\: y = y_1...y_k, y/2^n = \sum_{k=1}^n y_k/2^k \\
# & = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} \prod_{k=0}^n e^{2 \pi i x y_k/2^k } \vert y_1 ... y_n \rangle \:\text{after expanding the exponential of a sum to a product of exponentials} \\
# & = \frac{1}{\sqrt{N}} \bigotimes_{k=1}^n  \left(\vert0\rangle + e^{2 \pi i x /2^k } \vert1\rangle \right) \:\text{after rearranging the sum and products, and expanding} \\
# & = \frac{1}{\sqrt{N}} \left(\vert0\rangle + e^{2 \pi i[0.x_n]} \vert1\rangle\right) \otimes...\otimes  \left(\vert0\rangle + e^{2 \pi i[0.x_1.x_2...x_{n-1}.x_n]} \vert1\rangle\right) \:\text{as}\: e^{2 \pi i x/2^k} = e^{2 \pi i[0.x_k...x_n]} 
# \end{aligned}
# 
# This is a very useful form of the QFT for $N=2^n$ as only the last qubit depends on the the
# values of all the other input qubits and each further bit depends less and less on the input qubits. Furthermore, note that $e^{2 \pi i.0.x_n}$ is either $+1$ or $-1$, which resembles the Hadamard transform.
# 
# For the QFT circuit, together with the Hadamard gate, we will also need the controlled phase rotation gate, as defined in [OpenQASM](https://github.com/QISKit/openqasm), to implement the dependencies between the bits:
# $$CU_1(\theta) =
# \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & e^{i\theta}\end{bmatrix}$$

# Before we create the circuit code for general $N=2^n$, let's look at $N=8,n=3$:
# $$QFT_8\vert x_1x_2x_3\rangle = \frac{1}{\sqrt{8}} \left(\vert0\rangle + e^{2 \pi i[0.x_3]} \vert1\rangle\right) \otimes \left(\vert0\rangle + e^{2 \pi i[0.x_2.x_3]} \vert1\rangle\right) \otimes  \left(\vert0\rangle + e^{2 \pi i[0.x_1.x_2.x_3]} \vert1\rangle\right) $$
# 
# The steps to creating the circuit for $\vert y_1y_2x_3\rangle = QFT_8\vert x_1x_2x_3\rangle$ would be:
# 1. Apply a Hadamard to $\vert x_3 \rangle$, giving the state $\frac{1}{\sqrt{2}}\left(\vert0\rangle + e^{2 \pi i.0.x_3} \vert1\rangle\right) = \frac{1}{\sqrt{2}}\left(\vert0\rangle + (-1)^{x_3} \vert1\rangle\right)$
# 2. Apply a Hadamard to $\vert x_2 \rangle$, then depending on $k_3$ (before the Hadamard gate) a $CU_1(\frac{\pi}{2})$, giving the state $\frac{1}{\sqrt{2}}\left(\vert0\rangle + e^{2 \pi i[0.x_2.x_3]} \vert1\rangle\right)$
# 3. Apply a Hadamard to $\vert x_1 \rangle$, then $CU_1(\frac{\pi}{2})$ depending on $k_2$, and $CU_1(\frac{\pi}{4})$ depending on $k_3$.
# 4. Measure the bits in reverse order, that is $y_3 = x_1, y_2 = x_2, y_1 = y_3$.
# 
# In the Quantum Experience composer (if controlled phase rotation gates were available) this circuit would look like:
# <img src="../images/qft3.png" alt="Note: In order for images to show up in this jupyter notebook you need to select File => Trusted Notebook" width="400 px" align="center">
# 
# In QASM, it is:
# ```
# qreg q[3];
# creg c[3];
# h q[0];
# cu1(pi/2) q[1],q[0];
# h q[1];
# cu1(pi/4) q[2],q[0];
# cu1(pi/2) q[2],q[1];
# h q[2];
# ```
# 
# In QISKit, it is:
# ```
# q = Q_program.create_quantum_register("q", 3)
# c = Q_program.create_classical_register("c", 3)
# 
# qft3 = Q_program.create_circuit("qft3", [q], [c])
# qft3.h(q[0])
# qft3.cu1(math.pi/2.0, q[1], q[0])
# qft3.h(q[1])
# qft3.cu1(math.pi/4.0, q[2], q[0])
# qft3.cu1(math.pi/2.0, q[2], q[1])
# qft3.h(q[2])
# ```
# 
# For $N=2^n$, this can be generalised, as in the `qft` function in [tools.qi](https://github.com/QISKit/qiskit-sdk-py/blob/master/tools/qi/qi.py):
# ```
# def qft(circ, q, n):
#     """n-qubit QFT on q in circ."""
#     for j in range(n):
#         for k in range(j):
#             circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
#         circ.h(q[j])
# ```

# In[ ]:




