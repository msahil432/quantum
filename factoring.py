
# coding: utf-8

# # Factoring with the D-Wave System
# 
# In the [factoring demo](https://cloud.dwavesys.com/leap/demos/factoring/) you saw how the D-Wave system factored an integer by running a multiplication circuit in reverse.
# 
# This notebook demonstrates how you can solve a constraint satisfaction problem (CSP) on a quantum computer with the example of factoring. 
# 
# 1. [Factoring as a Constraint Satisfaction Problem](#Factoring-as-a-Constraint-Satisfaction-Problem) describes the factoring problem as an example CSP.
# 2. [Formulating the Problem for a D-Wave System](#Formulating-the-Problem-for-a-D-Wave-System) shows how such CSPs can be formulated for solution on a quantum computer.
# 3. [A Simple Example](#A-Simple-Example) codes a small CSP to clarify the solution technique.
# 4. [Factoring on the Quantum Computer](#Factoring-on-the-Quantum-Computer) codes a factoring problem for solution on the D-Wave system.
# 5. [Further Information](#Further-Information) details some points touched on in previous sections and examines more closely the results returned from the quantum computer. 
# 
# This notebook  should help you understand both the techniques and [Ocean software](https://github.com/dwavesystems) tools used for solving CSPs on D-Wave quantum computers.

# **New to Jupyter Notebooks?** JNs are divided into text or code cells. Pressing the **Run** button in the menu bar moves to the next cell. Code cells are marked by an "In: \[\]" to the left; when run, an asterisk displays until code completion: "In: \[\*\]".

# # Factoring as a Constraint Satisfaction Problem
# The complexity class for classical integer factoring is believed to be between P and NP-hard.  Although research has yielded algorithms that perform faster than the intuitive trial division, including Fermat's algorithm, Pollard's two algorithms, and sieve algorithms, it's still an open question whether a classical algorithm exists that can factor in polynomial time. For quantum computing, Shor's algorithm runs in polynomial time (the D-Wave system does not run this algorithm).  
# 
# This notebook solves factoring on a D-Wave quantum computer by formulating it as a *constraint satisfaction problem*. CSPs require that all a problem's variables be assigned values that result in the satisfying of all constraints. For factoring, the problem's constraints are that the two variables representing factors, $a$ and $b$, be assigned only natural numbers and that their multiplication be equal to the factored number, $P$. 
# 
# Among CSPs are hard problems well suited to solution on quantum computers. For example, the map-coloring problem is to color all regions of a map such that any two regions sharing a border have different colors (see a D-Wave system solve a four-color map-coloring problem here: [Ocean software examples](https://docs.ocean.dwavesys.com/en/latest/getting_started.html#examples)). The job-shop scheduling problem is to schedule multiple jobs done on several machines with constraints on the machines' execution of tasks. You can apply the solution technique shown here to many CSPs. 

# # Formulating the Problem for a D-Wave System
# How can we formulate the factoring problem in a way that a D-Wave quantum computer can understand? 
# 
# D-Wave systems solve binary quadratic models (BQM), the Ising model traditionally used in statistical mechanics and its computer-science equivalent, the quadratic unconstrained binary optimization (QUBO) problem. Given $N$ variables $x_1,...,x_N$, where each variable $x_i$ can have binary values $0$ or $1$, the system finds assignments of values that minimize
# 
# $\sum_i^N q_ix_i + \sum_{i<j}^N q_{i,j}x_i  x_j$,
# 
# where $q_i$ and $q_{i,j}$ are configurable (linear and quadratic) coefficients. To formulate a problem for the D-Wave system is to program $q_i$ and $q_{i,j}$ so that assignments of $x_1,...,x_N$ also represent solutions to the problem.
# 
# There are different ways to formulate the factoring problem as a BQM. Let's start with an intuitive one. 
# 
# ## Formulation A
# Feel comfortable skipping this subsection if you prefer to move on to what we'll be coding. Or try implementing it yourself as an exercise, although this is not our recommended formulation. 
# 
# You can solve an equation, say $x+1=2$, by minimizing the square of the subtraction of one side from another, $\min_x[2-(x+1)]^2$. Intuitively such a minimization seeks the shortest distance between the sides, which occurs at equality, with the square eliminating negative distance. 
# 
# For a factored number, $P$, to be equal to its factors, $a, b$, you can solve  $P=ab$ with the minimization
# 
# $\min_{a,b}(P-ab)^2$.
# 
# To solve this minimization on a quantum computer, we would represent the integers with equivalent binary numbers; for example, integer $a$ is represented as $a_0 + 2a_1 + 4a_2 +... +2^ma_m$, where variables $a_i$ can have binary values $0$ or $1$. 
# 
# The D-Wave system solves binary *quadratic* models so our now-binary minimization should not have terms with higher order than $a_ib_j$. However, squaring $(P-ab)$ introduces non-quadratic terms for problems of any decent size. We would therefore use order reduction techniques on all higher terms. For example, by introducing a new variable $x_k=a_0b_2$, we can replace a non-quadratic term such as $8b_0b_2a_0^2$ with quadratic term $8b_0x_k$ (notice that $a_0^2=a_0$), and do so repeatedly until all terms are linear or quadratic. 
# 
# We would now have a BQM. Programming its coefficients on a D-Wave system would solve the factoring problem. 
# 
# ## Formulation B
# 
# Formulation A above produces a BQM in three steps: (1) state equation $P=ab$ as a minimization, (2) represent integers as binary numbers, and (3) reduce to quadratic.
#    
# An alternative is to express the constraints of the problem with Boolean logic. This technique is very versatile: modern computing is built on Boolean gates, the scope of addressable problems is immense. Our implementation below follows these steps:
# 
# 1. Express $P=ab$ as a CSP with a binary multiplication circuit.
# 2. Convert to a BQM.
# 3. Program the quantum computer with the BQM's coefficients.
# 
# Below we'll note some differences between the two formulations.

# # A Simple Example
# This section illustrates the steps of Formulation B above on a very simple problem: a circuit with two switches that turns on a light. 
# 
# <img src="images/example_light_and_switches.png" width=300x/>

# ## Step 1: Express as a CSP with Boolean Logic
# 
# We can express the problem of how to turn on the light as a CSP with a single constraint: for light $L$ to shine, both switches $SW_1$ and $SW_2$ must be on. In logic, we express this constraint as the conjunction $L = SW_1 \wedge SW_2$. 
# 
# Unlike real-world CSPs, which can have thousands of constraints to satisfy simultaneously, the one constraint of this trivial CSP also expresses its solution. Although we forgo a surprise ending, it's instructive to run through the steps needed to "solve the problem".
# 
# First, we express the CSP with binary variables:
# 
# * Switches $SW_1$ and $SW_2$ are represented by binary variables $x_1, x_2$ with values 1 when on and 0 when off.
# * Light $L$ is represented by binary variable $x_3$ with value 1 if it's shining and 0 if not.
# * Our logical conjunction, $L = SW_1 \wedge SW_2$, is expressed in binary format as $x_3 = x_1 \wedge x_2$. 
# 
# The graphic below shows an AND gate and its truth table, which gives the gate's output, $x_3$, for all combinations of inputs $x_1, x_2$. 
# 
# <img src="images/AND_TruthTableandGate.png" width=300x/>
# 
# It's clear from the table that our problem's constraint, $L = SW_1 \wedge SW_2$, and the AND gate's operation, $x_3 = x_1x_2$, are equivalent. We can express our constraint as an AND gate. 

# Ocean's [dwavebinarycsp](https://docs.ocean.dwavesys.com/projects/binarycsp/en/latest/) binary CSP tool provides factories for useful constraints such as logic gates. Run the cell below (by pressing the **Run** button with your mouse in the cell) to create a CSP with a constraint representing an AND gate.

# In[ ]:


import dwavebinarycsp as dbc
# Add an AND gate as a constraint to CSP and_csp defined for binary variables 
and_gate = dbc.factories.and_gate(["x1", "x2", "x3"])
and_csp = dbc.ConstraintSatisfactionProblem('BINARY')
and_csp.add_constraint(and_gate)

# Test that for input x1,x2=1,1 the output is x3=1 (both switches on and light shining)
and_csp.check({"x1": 1, "x2": 1, "x3": 1})


# ## Step 2: Convert to a BQM 
# The quantum computer solves binary quadratic models. Let's express our light-circuit CSP as a BQM.
# 
# An advantage of Formulation B is that BQMs are known for logic gates (you can find BQMs for gates in the D-Wave system documentation and see examples here: [Ocean software examples](https://docs.ocean.dwavesys.com/en/latest/getting_started.html#examples)). More than one BQM can represent our AND gate; it's just a polynomial of binary variables, with only linear and quadratic terms, that has lowest value for variables that match rows of the AND truth table. Ocean tools can do the math for you, but here let's first write out a BQM for our AND gate:  $3x_3 + x_1x_2 - 2x_1x_3 - 2x_2x_3$.
# 
# To see that this BQM represents the AND gate, you can set its variables to the values of the AND truth table, for example $x_1, x_2, x_3=0,0,0$, and to non-valid values, such as $ x_1, x_2, x_3=0,0,1$. All the former should produce lower values than any of the latter. The code cell below does so for all possible configurations. 
# 
# Run the next cell. In the printed output, the left column (under "E") is the BQM's value for the combinations of variables to the right (under "x1, x2, x3").

# In[ ]:


# Use itertools to produce all possible 3-bit binary combinations for x1, x2, x3
import itertools
configurations = []
for (x1, x2, x3) in  list(itertools.product([0, 1], repeat=3)):
     E = 3*x3+x1*x2-2*x1*x3-2*x2*x3
     configurations.append((E, x1, x2, x3))
# Sort from lowest to highest value of the BQM
configurations.sort()
# Print BQM value under "E" and all configurations under "x1, x2, x3"
print("E, x1, x2, x3")
configurations


# Now let's use Ocean's [dwavebinarycsp](https://docs.ocean.dwavesys.com/projects/binarycsp/en/latest/)  to convert our binary CSP to a BQM for us. The code cell below does so and prints out the BQM's coefficients, the inputs used to program the D-Wave system. As noted, More than one BQM can represent our AND gate, so the BQM generated here does not have to match the one we wrote ourselves.

# In[ ]:


# Convert the CSP into BQM and_bqm
and_bqm = dbc.stitch(and_csp)
and_bqm.remove_offset()
# Print the linear and quadratic coefficients. These are the programable inputs to a D-Wave system
print(and_bqm.linear)
print(and_bqm.quadratic)


# ## Step 3: Solve By Minimization 
# Lastly, we solve the problem by finding variable values that produce the BQM's lowest values. For real-world problems, with large numbers of variables and constraints, minimizing a BQM is hard: this is where a quantum computer comes in handy. 
# 
# The next section, which solves a factoring problem, uses the D-Wave system. For this trivial example, instead of using the D-Wave system as the *sampler* (the component used to minimize a BQM), we'll use one of Ocean software's test samplers. Ocean's [dimod](https://docs.ocean.dwavesys.com/projects/dimod/en/latest/) provides one that simply returns the BQM's value for every possible assignment of variable values.

# In[ ]:


# Use a dimod test sampler that gives the BQM value for all values of its variables
from dimod import ExactSolver
sampler = ExactSolver()


# The next cell prints the BQM's values ("energy") in ascending order. Note that they are indeed lowest for valid assignments (values under "x1, x2, x3" match rows of our AND truth table) and higher for non-valid ones.  

# In[ ]:


# Solve the BQM
solution = sampler.sample(and_bqm)
list(solution.data())


# Note: to understand the examples of this Jupyter Notebook, it's enough to understand that samplers such as the D-Wave system find solutions that minimize a BQM. If you want further details on that minimization (the "energy" here and "E" previously), see below under [Further Information](#Further-Information).

# # Factoring on the Quantum Computer
# This section solves a factoring problem as a CSP, following the same steps we used for the simple problem of the light circuit:
# 
# 1. Express factoring as a CSP using Boolean logic operations.
# 2. Convert to a BQM.
# 3. Minimize the BQM.

# ## Step 1: Express Factoring as Multiplication Circuit
# We again start by expressing the problem with Boolean logic gates, in this case a multiplication circuit.
# 
# This example factors integer 21 which we represent as a 6-bit binary number. To express our factoring equation, $P = ab$, in Boolean logic we use a simple 3-bit multiplier (a circuit that takes two 3-bit binary numbers and outputs their 6-bit binary product).  
# 
# Note: Binary multipliers are made with logic gates like the AND gate we used above. Understanding the factoring example and its application to solving CSPs does not require an understanding of binary multiplication. If you do want to know more, see below under [Further Information](#Further-Information).

# In[ ]:


# Set an integer to factor
P = 21

# A binary representation of P ("{:06b}" formats for 6-bit binary)
bP = "{:06b}".format(P)
print(bP)


# The cell below obtains a multiplication circuit from Ocean's [dwavebinarycsp](https://docs.ocean.dwavesys.com/projects/binarycsp/en/latest/) binary CSP tool as we did before for the AND gate. Run the cell to create a CSP for a 3-bit multiplication circuit.

# In[ ]:


csp = dbc.factories.multiplication_circuit(3)
# Print one of the CSP's constraints, the gates that constitute 3-bit binary multiplication
print(next(iter(csp.constraints)))


# ## Step 2: Convert to a BQM
# 
# Next we express our multiplication circuit as a BQM that provides the coefficients used to program the problem on a D-Wave system.

# In[ ]:


# Convert the CSP into BQM bqm
bqm = dbc.stitch(csp, min_classical_gap=.1)
# Print a sample coefficient (one of the programable inputs to a D-Wave system)
print("p0: ", bqm.linear['p0'])


# Running the next cell just creates a nice visualization of the BQM. Each node of the graph represents a variable; these include P and its factors as binary numbers, and some internal variables of the multiplication circuit.

# In[ ]:


# To see helper functions, select Jupyter File Explorer View from the Online Learning page
from helpers import draw
draw.circuit_from(bqm)


# As in the [factoring demo](https://cloud.dwavesys.com/leap/demos/factoring/), the D-Wave system factors our integer by running a multiplication circuit in reverse. Below, we fix the variables of the multiplication circuit's BQM to the binary digits of the factored number P. This modifies the BQM by removing the known variables and updating neighboring values accordingly.

# In[ ]:


# Our multiplication_circuit() creates these variables
p_vars = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']

# Convert P from decimal to binary
fixed_variables = dict(zip(reversed(p_vars), "{:06b}".format(P)))
fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}

# Fix product variables
for var, value in fixed_variables.items():
    bqm.fix_variable(var, value)
    
# Confirm that a P variable has been removed from the BQM, for example, "p0"
print("Variable p0 in BQM: ", 'p0' in bqm)
print("Variable a0 in BQM: ", 'a0' in bqm)


# ## Step 3: Submit to the Quantum Computer
# Lastly, we solve the BQM by finding variable assignments that produce its lowest values. Here we use a D-Wave system. 

# ### Setting Up a Solver
# Typically when working with a D-Wave system, you configure a default solver and its SAPI URL with your API token and that configuration is used implicitly. Occasionally you may wish to override a default and specify particular solver settings.
# 
# The cell below displays your default solver configuration. The next cell allows you to explicitly select a solver and API token.

# In[ ]:


from helpers.solvers import default_solver
my_solver, my_token = default_solver()


# In[ ]:


# Uncomment if you need to paste in a solver and/or token
# my_solver = 'paste your solver in here'
# my_token = 'paste your token in here'


# Your default or manually set solver and token are used in the next cell, where *DWaveSampler()* from Ocean software's [dwave-system](https://docs.ocean.dwavesys.com/projects/system/en/latest/) tool handles the connection to a D-Wave system. 

# In[ ]:


from dwave.system.samplers import DWaveSampler
# Use a D-Wave system as the sampler
sampler = DWaveSampler(solver=my_solver, token=my_token)
_, target_edgelist, target_adjacency = sampler.structure


# Mapping between the graph of our problem&mdash;the multiplication circuit's graph with nodes labeled "a0", "b0" etc.&mdash;to the D-Wave QPU's numerically indexed qubits, is known as *minor-embedding*. A problem can be minor embedded onto the QPU in a variety of ways and this affects solution quality and performance. Ocean software provides tools suited for different types of problems; for example, [dwave-system](https://docs.ocean.dwavesys.com/projects/system/en/latest/) *EmbeddingComposite()* has a heuristic for automatic embedding. 
# 
# This example uses a pre-calculated minor-embedding (see [below](#Further-Information) for details).

# In[ ]:


import dimod
from helpers.embedding import embeddings

# Set a pre-calculated minor-embeding
embedding = embeddings[sampler.solver.id]
bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, 3.0)

# Confirm mapping of variables from a0, b0, etc to indexed qubits 
print("Variable a0 in embedded BQM: ", 'a0' in bqm_embedded)
print("First five nodes in QPU graph: ", sampler.structure.nodelist[:5])


# When the D‑Wave quantum computer solves a problem, it uses quantum phenomena such as superposition and tunneling to explore all possible solutions simultaneously and find a set of the best ones. Because the sampled solution is probabilistic, returned solutions may differ between runs. Typically, when submitting a problem to the system, we ask for many samples, not just one. This way, we see multiple “best” answers and reduce the probability of settling on a suboptimal answer.
# 
# In the code below, *num_reads* should provide enough samples to make it likely a valid answer is among them.

# In[ ]:


# Return num_reads solutions (responses are in the D-Wave's graph of indexed qubits)
kwargs = {}
if 'num_reads' in sampler.parameters:
    kwargs['num_reads'] = 50
if 'answer_mode' in sampler.parameters:
    kwargs['answer_mode'] = 'histogram'
response = sampler.sample(bqm_embedded, **kwargs)
print("A solution indexed by qubits: \n", next(response.data(fields=['sample'])))

# Map back to the BQM's graph (nodes labeled "a0", "b0" etc,)
response = dimod.unembed_response(response, embedding, source_bqm=bqm)
print("\nThe solution in problem variables: \n",next(response.data(fields=['sample'])))


# ### Viewing the Solution
# We need to convert back from binary numbers to integers. Because quantum computing is probabilistic, there is a slight chance that in many executions of this example, your execution might return an incorrect example. Rerunning the previous cell will most likely produce a correct answer. 

# In[ ]:


from helpers.convert import to_base_ten
# Select just just the first sample. 
sample = next(response.samples(n=1))
dict(sample)
a, b = to_base_ten(sample)

print("Given integer P={}, found factors a={} and b={}".format(P, a, b))


# # Summary
# 
# This Jupyter Notebook showed how you can formulate a constraint satisfaction problem for solution on a quantum computer using Ocean software. We solved a factoring problem as an example of one proposed solution technique.   
# 
# We considered two ways of formulating the factoring problem. Formulation A is intuitive and direct, but conversion of large integers to binary introduces (a) increasing weights per bit, $2^ma_m$, and (b) in the squaring of $(P-ab)$, terms of higher order that need to be reduced to quadratic. These affect performance. Formulation B, using binary gates, is a useful technique in general. The "modularity" of binary gates provides some benefits for minor-embedding: repeated small units that can be "tiled" onto the QPU's topology.       

# # Further Information
# 
# This section provides more information on binary multiplication, minimizing BQMs, sampling for solutions, and minor-embedding.  

# ## Binary Multiplication
# 
# Binary number multiplication works the same way that multiplication is taught in school. That is, for two 3-bit numbers $a_2a_1a_0$ and $b_2b_1b_0$, the multiplication is written as
# 
# \begin{array}{rrrrrr}
# &  &  &  &  & \\
# &  &  & a_{2} & a_{1} & a_{0}\\
# \times &  &  & b_{2} & b_{1} & b_{0}\\
# \hline 
# & 0 & 0 & b_{0}a_{2} & b_{0}a_{1} & b_{0}a_{0}\\
# & 0 & b_{1}a_{2} & b_{1}a_{1} & b_{1}a_{0} & 0\\
# & b_{2}a_{2} & b_{2}a_{1} & b_{2}a_{0} & 0 & 0\\
# \hline 
# p_{5}  & p_{4} & p_{3} & p_{2} & p_{1} & p_{0}\\
# \end{array}
# 
# where each $p_i$ is the sum of the values in the $i$-th column; for example, $p_2 = b_{0}a_{2} + b_{1}a_{1} + b_{2}a_{0}$.
# 
# A binary multiplication circuit represents each of our nine products $b_ia_j$ as an AND gate and each of the three summations of 5-bit partial products as an adder.
# 
# <img src="images/BinaryMultiplicationCircuit_gates.png" width=400x/>
# 
# The simple implementation used by our example does not support two's complement, carry lookahead, or any other nifty features. If you're curious [this Wikipedia article](https://en.wikipedia.org/wiki/Binary_multiplier) is a great place to start.
# 
# In the circuit below, given inputs 011 (3) and 111 (7), the circuit outputs 010101 (21). 
# 
# <img src="images/21.jpg" width=400x/>

# ## Minimizing BQMs
# 
# A fundamental rule of physics is that everything tends to seek a minimum energy state. Objects slide down hills; hot things cool down over time. This behavior is also true in the world of quantum physics. To solve optimization problems on a D-Wave system, we frame them as energy-minimization problems, and solve them through the physics of the QPU, which finds low-energy states.
# 
# D-Wave systems solve a subset of binary quadratic models, the Ising and quadratic unconstrained binary optimization (QUBO) problems, by finding assignments of variables that correspond to minimum energy.
# 
# For the Ising model, $N$ variables $\bf s=[s_1,...,s_N]$ correspond to physical Ising spins, where $h_i$ are the biases and $J_{i,j}$ the couplings (interactions) between spins.
# 
# $\text{Ising:} \qquad
#   E(\bf{s}|\bf{h},\bf{J})
#   = \left\{ \sum_{i=1}^N h_i s_i +
#   \sum_{i<j}^N J_{i,j} s_i s_j  \right\}
#   \qquad\qquad s_i\in\{-1,+1\}$
# 
# For the QUBO model, $N$ binary variables represented as an upper-diagonal matrix $Q$, where diagonal terms are the linear coefficients and the nonzero off-diagonal terms the quadratic coefficients.
# 
# $\text{QUBO:} \qquad E(\bf{x}| \bf{Q})
#     =  \sum_{i\le j}^N x_i Q_{i,j} x_j
#     \qquad\qquad x_i\in \{0,1\}$
# 
# The BQM we formulate to express our problem sets up qubits representing our binary variables on the QPU. The QPU finds low-energy states of the qubits. In most cases, the lower the energy, the better the solution. 

# ## Samples and Solutions
# 
# Samplers are processes that sample from low energy states of an objective function, which is a mathematical expression of the energy of a system. A binary quadratic model (BQM) sampler samples from low energy states in models such as those defined by an Ising model traditionally used in statistical mechanics or its computer-science equivalent, the QUBO, and returns an iterable of samples, in order of increasing energy.
# 
# The D-Wave returns a [`dimod.Response`](https://docs.ocean.dwavesys.com/projects/dimod/en/latest/reference/response.html) object that contains all of the information about the generated samples and has some access methods. For instance, `response.samples()` returns mappings between the variables of the BQM and the values they take in each sample and `response.data(['energy'])` contains the energy value associated with the sample. 

# The next cell views the energy of the samples, using a `dict` mapping pairs `(a, b)` to information about them.

# In[ ]:


from collections import OrderedDict

# Function for converting the response to a dict of integer values
def response_to_dict(response):
    results_dict = OrderedDict()
    for sample, energy in response.data(['sample', 'energy']):
        # Convert A and B from binary to decimal
        a, b = to_base_ten(sample)
        # Aggregate results by unique A and B values (ignoring internal circuit variables)
        if (a, b) not in results_dict:
            results_dict[(a, b)] = energy
            
    return results_dict

# Convert the dimod.Response object to a dict and display it
results = response_to_dict(response)
results


# We can create a scatter plot of the samples and their energies, showing that the lowest energy states correspond to correct answers.

# In[ ]:


draw.energy_of(results)


# ## Minor-Embedding
# 
# The pre-calculated minor-embedding we used was selected for its good performance from many candidate embeddings generated by a heuristic algorithm from *dwave-ocean-sdk* called [minorminer](https://docs.ocean.dwavesys.com/projects/minorminer/en/latest/) ([GitHub repo](https://github.com/dwavesystems/minorminer)). Alternatively, we can use this tool here to generate a new embedding that may or may not have as good performance. Here we will ask for more samples from the QPU by raising the *num_reads* parameter from 50 to 1000 to increase our chances of getting a correct answer. Once a problem has been submitted to the QPU, the difference in processing time for 50 or 1000 runs is small. 

# In[ ]:


import minorminer

# Find an embedding
embedding = minorminer.find_embedding(bqm.quadratic, target_edgelist)
if bqm and not embedding:
    raise ValueError("no embedding found")

# Apply the embedding to the factoring problem to map it to the QPU
bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, 3.0)

# Request num_reads samples
kwargs['num_reads'] = 1000
response = sampler.sample(bqm_embedded, **kwargs)

# Convert back to the problem graph
response = dimod.unembed_response(response, embedding, source_bqm=bqm)


# Convert the D-Wave system's response to a dict of integer values and display.

# In[ ]:


results = response_to_dict(response)
results


# Create a scatter plot of the samples and their energies.

# In[ ]:


draw.energy_of(results)

