#!/usr/bin/env python3

## Initialization
P = 3*5
digits = "{:06b}"
size = 3
vars = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']#, 'p6', 'p7']#, 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15']#, 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31']
first = ['a0', 'a1', 'a2']#, 'a3']#, 'a4', 'a5', 'a6', 'a7']#, 'a8']#, 'a9', 'a10', 'a11']#, 'a12', 'a13', 'a14', 'a15']
second = ['b0', 'b1', 'b2']#, 'b3']#, 'b4', 'b5', 'b6', 'b7']#, 'b8']#, 'b9', 'b10', 'b11']#, 'b12', 'b13', 'b14', 'b15']
first.reverse()
second.reverse()
reads_count = 100

print(f"Factoring: {P} using {size}x{size} Multiplier:\n\n")

## Step 1: Express Factoring as Multiplication Circuit

# A binary representation of P 
bP = digits.format(P)
print(f"Representing {P} as {digits} : {bP}")

import dwavebinarycsp as dbc
print("\n Getting multiplication circuit...")
csp = dbc.factories.multiplication_circuit(size)


## Step 2: Convert CSP to a BQM

print("\n Converting it to BQM...")
bqm = dbc.stitch(csp, min_classical_gap=.1)
print("A sample coefficient, p0: ", bqm.linear['p0'])

# from helpers import draw
# draw.circuit_from(bqm)

# Our multiplication_circuit() creates these variables
p_vars = vars

# Convert P from decimal to binary
fixed_variables = dict(zip(reversed(p_vars), digits.format(P)))
fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}

# Fix product variables
for var, value in fixed_variables.items():
    bqm.fix_variable(var, value)
    
# Confirm that a P variable has been removed from the BQM, for example, "p0"
print("Variable p0 in BQM: ", 'p0' in bqm)
print("Variable a0 in BQM: ", 'a0' in bqm)



## Step 3: Submit to the Quantum Computer

#from helpers.solvers import default_solver
#my_solver, my_token = default_solver()
my_solver = 'DW_2000Q_2_1'
my_token = 'DEV-ee8c6930af28fcac5c695e94253b79e6aa8cda96'

from dwave.system.samplers import DWaveSampler
# Using D-Wave system as the sampler
sampler = DWaveSampler(solver=my_solver, token=my_token)
_, target_edgelist, target_adjacency = sampler.structure

import dimod
import minorminer
# Setting minor-embeding
print("\n Embedding BQM...")
embedding = minorminer.find_embedding(bqm.quadratic, target_edgelist)
bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, 3.0)

# Confirm mapping of variables from a0, b0, etc to indexed qubits 
# print("Variable a0 in embedded BQM: ", 'a0' in bqm_embedded)
# print("First five nodes in QPU graph: ", sampler.structure.nodelist[:5])

# Return num_reads solutions (responses are in the D-Wave's graph of indexed qubits)
kwargs = {}
if 'num_reads' in sampler.parameters:
    kwargs['num_reads'] = reads_count
if 'answer_mode' in sampler.parameters:
    kwargs['answer_mode'] = 'histogram'

kwargs['programming_thermalization'] = 1000
kwargs['reduce_intersample_correlation'] = True

print("\n Sampling on Quantum Computer...")
response = sampler.sample(bqm_embedded, **kwargs)

# print("A solution indexed by qubits: \n", next(response.data(fields=['sample'])))

print("\n Mapping Response back to BQM...")
response = dimod.unembed_response(response, embedding, source_bqm=bqm)

#print("\nThe solution in problem variables: \n",next(response.data(fields=['sample'])))

print("\n Selecting correct samples:")
correct = 0
length = -1
samples = iter(response.samples())
for sample in samples:
    a=""
    b=""

    for val in list(first):
        a+=str(sample[val])

    if(length == -1):
        length = len(sample)
        print("Q-bits Used: ",length)

    if (P%int(a,2)!=0):
        for val in list(second):
                b+=str(sample[val])
        if(P%int(b,2)!=0):
            continue

    if(correct == 0):
        print(f"\n First factor is : {a} ->",int(a, 2))
        print(f" Second factor is : {b} ->",int(b, 2))

    correct = correct+1

print("Correct ones:",correct)
# The cycle repeats for some number of samples specified by the user in the num_reads parameter, and returns one solution per sample. The total time to complete the requested number of samples is returned by the QPU as qpu_sampling_time.
total_time = response._info["timing"]["qpu_sampling_time"]
print(f"Total Time taken : {total_time} ns.")
print(f"Average Time for {reads_count} samples: ", total_time/reads_count, " ns.")
