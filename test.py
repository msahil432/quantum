#!/usr/bin/env python3

P = 13*11

print(f"Factoring {P} using While Loop:\n\n")

import time
before = time.time()

i = 2
while (i < (P/2)):
    if (P%i==0):
        break
    i=i+1

print("Factors are: ",i, P/i, "\n")
after = time.time()
print("Before: ", "%.20f" % before)
print("After:  ","%.20f" % after)
print("Time Taken: ", "%.15f" % ((after-before)*1000000000), " ns.")