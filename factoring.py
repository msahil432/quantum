import dwavebinarycsp as dbc

# Use itertools to produce all possible 3-bit binary combinations for x1, x2, x3
import itertools

# Use a dimod test sampler that gives the BQM value for all values of its variables
from dimod import ExactSolver
sampler = ExactSolver()

## Step 1: Express Factoring as Multiplication Circuit
P = 15
digits = "{:06b}"
size = 3
vars = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
reads_count = 50

from helpers.solvers import default_solver
my_solver, my_token = default_solver()

# my_solver = 'paste your solver in here'
# my_token = 'paste your token in here'

# A binary representation of P ("{:06b}" formats for 6-bit binary)
bP = digits.format(P)
print(bP)

csp = dbc.factories.multiplication_circuit(size)
# Print one of the CSP's constraints, the gates that constitute 3-bit binary multiplication
print(next(iter(csp.constraints)))

## Step 2: Convert to a BQM

# Convert the CSP into BQM bqm
bqm = dbc.stitch(csp, min_classical_gap=.1)
# Print a sample coefficient (one of the programable inputs to a D-Wave system)
print("p0: ", bqm.linear['p0'])

# To see helper functions, select Jupyter File Explorer View from the Online Learning page
from helpers import draw
draw.circuit_from(bqm)

# Our multiplication_circuit() creates these variables
p_vars = vars

# Convert P from decimal to binary
fixed_variables = dict(zip(reversed(p_vars), digits.format(P)))   #change_here
fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}

# Fix product variables
for var, value in fixed_variables.items():
    bqm.fix_variable(var, value)
    
# Confirm that a P variable has been removed from the BQM, for example, "p0"
print("Variable p0 in BQM: ", 'p0' in bqm)
print("Variable a0 in BQM: ", 'a0' in bqm)

## Step 3: Submit to the Quantum Computer

from dwave.system.samplers import DWaveSampler
# Use a D-Wave system as the sampler
sampler = DWaveSampler(solver=my_solver, token=my_token)
_, target_edgelist, target_adjacency = sampler.structure

import dimod
from helpers.embedding import embeddings

# Set a pre-calculated minor-embeding
embedding = embeddings[sampler.solver.id]
bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, 3.0)

# Confirm mapping of variables from a0, b0, etc to indexed qubits 
print("Variable a0 in embedded BQM: ", 'a0' in bqm_embedded)
print("First five nodes in QPU graph: ", sampler.structure.nodelist[:5])

# Return num_reads solutions (responses are in the D-Wave's graph of indexed qubits)
kwargs = {}
if 'num_reads' in sampler.parameters:
    kwargs['num_reads'] = reads_count
if 'answer_mode' in sampler.parameters:
    kwargs['answer_mode'] = 'histogram'
response = sampler.sample(bqm_embedded, **kwargs)
print("A solution indexed by qubits: \n", next(response.data(fields=['sample'])))

# Map back to the BQM's graph (nodes labeled "a0", "b0" etc,)
response = dimod.unembed_response(response, embedding, source_bqm=bqm)
print("\nThe solution in problem variables: \n",next(response.data(fields=['sample'])))


from helpers.convert import to_base_ten
# Select just just the first sample. 
sample = next(response.samples(n=1))
dict(sample)
print(sample)