#!/usr/bin/env python3

def factorize(P, leng, reads_count):
    ## Initialization
    # P = 3*7
    digits = "{:0"+str(leng)+"b}"
    size = int(leng/2)
    vs = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31']
    fir = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    sec = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

    temp = 0
    vars = []
    while(temp<leng):
        vars.append(vs[temp])
        temp = temp+1

    temp = 0
    first = []
    second = []
    while(temp<size):
        first.append(fir[temp])
        second.append(sec[temp])
        temp=temp+1

    first.reverse()
    second.reverse()

    #reads_count = 1000

    # print(f"Factoring: {P} using {size}x{size} Multiplier:\n\n")

    ## Step 1: Express Factoring as Multiplication Circuit

    # A binary representation of P 
    bP = digits.format(P)
    # print(f"Representing {P} as {digits} : {bP}")

    import dwavebinarycsp as dbc
    # print("\n Getting multiplication circuit...")
    csp = dbc.factories.multiplication_circuit(size)


    ## Step 2: Convert CSP to a BQM

    # print("\n Converting it to BQM...")
    bqm = dbc.stitch(csp, min_classical_gap=.1)
    # print("A sample coefficient, p0: ", bqm.linear['p0'])

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
    # print("Variable p0 in BQM: ", 'p0' in bqm)
    # print("Variable a0 in BQM: ", 'a0' in bqm)



    ## Step 3: Submit to the Quantum Computer

    #from helpers.solvers import default_solver
    #my_solver, my_token = default_solver()
    my_solver = 'DW_2000Q_2_1'
    my_token = 'DEV-4d7a6f09914ed65fc27f512d0499d0ca2b344dcc'

    from dwave.system.samplers import DWaveSampler
    # Using D-Wave system as the sampler
    sampler = DWaveSampler(solver=my_solver, token=my_token)
    _, target_edgelist, target_adjacency = sampler.structure

    import dimod
    import minorminer
    # Setting minor-embeding
    # print("\n Embedding BQM...")
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

    # print("\n Sampling on Quantum Computer...")
    response = sampler.sample(bqm_embedded, **kwargs)

    # print("A solution indexed by qubits: \n", next(response.data(fields=['sample'])))

    # print("\n Mapping Response back to BQM...")
    response = dimod.unembed_response(response, embedding, source_bqm=bqm)

    #print("\nThe solution in problem variables: \n",next(response.data(fields=['sample'])))

    # print("\n Selecting correct sample:")
    i = 1
    correct = 0
    while (i<=reads_count):
        sample = next(response.samples(n=1))
        i=i+1

        a=""
        b=""

        for val in list(first):
            a+=str(sample[val])
        
        if (P%int(a,2)!=0):
            continue
        
        # print(f"\n First factor is : {a} ->",int(a, 2))

        for val in list(second):
            b+=str(sample[val])
        # print(f" Second factor is : {b} ->",int(b, 2))
        
        #break
        correct = correct+1


    # The cycle repeats for some number of samples specified by the user in the num_reads parameter, and returns one solution per sample. The total time to complete the requested number of samples is returned by the QPU as qpu_sampling_time.
    s_time = response._info["timing"]["qpu_sampling_time"]
    p_time = response._info["timing"]["qpu_programming_time"]
    # print(f"Total Time taken : {total_time} ns.")
    # print(f"Average Time for {reads_count} samples: ", total_time/reads_count, " ns.")

    import time

    temp = 0
    classic = 0
    while(temp<reads_count):
        before = time.time()
        i = 2
        while (i < (P/2)):
            if (P%i==0):
                break
            i=i+1

        # print("Factors are: ",i, P/i, "\n")
        after = time.time()
        classic = classic + ((after-before)*1000000000)
        temp = temp+1
        # print("Before: ", "%.20f" % before)
        # print("After:  ","%.20f" % after)
        # print("Time Taken: ", "%.15f" % classic, " ns.")

    print(fixed_variables.items())

    print(leng, "\t", 
    i,"\t", 
    P/i,"\t",
    reads_count, "\t", 
    len(fixed_variables.items()),"\t",
    p_time, "\t",
    s_time, "\t",
    correct, "\t\t",
    classic)

print("""Bits \t First \t Sec \t reads\t q_bits\t p_time\t s_time\t correct \t classic""")
factorize(13*17, 10, 1000)
# factorize(5*7, 6, 10)
# factorize(7*11, 8, 10)
# factorize(11*13, 8, 10)

