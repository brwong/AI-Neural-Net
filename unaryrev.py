#!/usr/bin/python

from sys import argv

if len(argv) < 3:
	print "not enough arguments."
	print "unaryrev.py unaryfile.csv decimalfile.csv"
	exit()

f1 = open(argv[1], 'r')

vals = [[int(v) for v in i.strip().split(',')] for i in f1.readlines()]

f1.close()

default = 0
if len(argv) > 3:
	try:
		default = int(argv[3])
	except:
		pass

conv = [default for i in vals]

def un(v):
     n = 0
     for i in v:
         if i==1:
             return n
         n += 1
     return -1

for i in range(len(vals)):
     s = sum(vals[i])
     if s == 1:
         conv[i] = un(vals[i])

f2 = open(argv[2], 'w')
for n in conv:
	f2.write(str(n) + "\n")
f2.close()

