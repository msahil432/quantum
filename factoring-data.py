#!/usr/bin/env python3

def qsample(P, embedding, bqm, bqm_embedded, sampler, reads_count, thermal):
    
    kwargs = {}
    if 'num_reads' in sampler.parameters:
        kwargs['num_reads'] = reads_count
    if 'answer_mode' in sampler.parameters:
        kwargs['answer_mode'] = 'histogram'

    kwargs['programming_thermalization'] = thermal
    kwargs['reduce_intersample_correlation'] = True

    response = sampler.sample(bqm_embedded, **kwargs)
    
    import dimod
    response = dimod.unembed_response(response, embedding, source_bqm=bqm)

    correct = 0
    qbits = -1
    samples = iter(response.samples())
    for sample in samples:
        a=""
        b=""
        try:
            if(qbits==-1):
                qbits = len(sample)

            for val in list(first):
                a+=str(sample[val])
            
            if (P%int(a,2)!=0):
                for val in list(second):
                    b+=str(sample[val])
                
                if(P%int(b,2)!=0):
                    continue
            
            correct = correct+1
        except:
            a = ""


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

    #print(fixed_variables.items())

    print(leng, "\t", 
    i,"\t", 
    P/i,"\t",
    reads_count, "\t", 
    qbits,"\t",
    p_time, "\t",
    thermal,"\t",
    s_time, "\t",
    correct, "\t",
    classic)

# -----------------------------------------
leng = 6
P = 7*7

digits = "{:0"+str(leng)+"b}"
size = int(leng/2)
vs = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31']
fir = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
sec = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

temp = 0
p_vars = []
while(temp<leng):
    p_vars.append(vs[temp])
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

bP = digits.format(P)

import dwavebinarycsp as dbc
csp = dbc.factories.multiplication_circuit(size)
bqm = dbc.stitch(csp, min_classical_gap=.1)

fixed_variables = dict(zip(reversed(p_vars), digits.format(P)))
fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}

for var, value in fixed_variables.items():
    bqm.fix_variable(var, value)
    
#from helpers.solvers import default_solver
#my_solver, my_token = default_solver()
my_solver = 'DW_2000Q_2_1'
# my_token = 'DEV-05a577c056c4609844ab17c5fdee534a323ec8fe'
my_token = 'DEV-4d7a6f09914ed65fc27f512d0499d0ca2b344dcc'

from dwave.system.samplers import DWaveSampler

sampler = DWaveSampler(solver=my_solver, token=my_token)
_, target_edgelist, target_adjacency = sampler.structure

import dimod
import minorminer
embedding = minorminer.find_embedding(bqm.quadratic, target_edgelist)
bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, 3.0)

print("""Bits \t First \t Sec \t reads\t q_bits\t p_time\tthermal\t s_time\t correct \t classic""")

qsample(P, embedding, bqm, bqm_embedded, sampler, 100, 1)
qsample(P, embedding, bqm, bqm_embedded, sampler, 100, 100)
qsample(P, embedding, bqm, bqm_embedded, sampler, 100, 1000)
qsample(P, embedding, bqm, bqm_embedded, sampler, 10, 10)
qsample(P, embedding, bqm, bqm_embedded, sampler, 100, 10)
qsample(P, embedding, bqm, bqm_embedded, sampler, 1000, 10)
