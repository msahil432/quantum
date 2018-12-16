#!/usr/bin/env python3

P = 227*211

print(f"Factoring {P} with While Loop: ")

import time
temp = 0
before = time.time()
i = 2
while (i < (P/2)):
    if (P%i==0):
        break
    i=i+1

# print("Factors are: ",i, P/i, "\n")
after = time.time()
classic = ((after-before)*1000000000)
temp = temp+1
# print("Before: ", "%.20f" % before)
# print("After:  ","%.20f" % after)
print("Time:", "%.13f" % classic)