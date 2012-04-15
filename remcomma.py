#!/usr/bin/python

from sys import argv

if len(argv) < 3:
	exit()

f1 = open(argv[1], 'r')
f2 = open(argv[2], 'w')

l = f1.readline()
while l:
	f2.write(l.replace(',', ' '))
	l = f1.readline()

f2.close()
f1.close()

print "done."

