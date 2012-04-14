#!/usr/bin/python

from sys import argv

if len(argv) < 3:
	print "not enough arguments."
	print "checker.py original.txt filetocheck.txt"
	exit()

original = open(argv[1], 'r')
compare = open(argv[2], 'r')

total = 0
correct = 0

for l in original.readlines():
	total += 1
	if l == compare.readline():
		correct += 1

print str(correct) + " out of " + str(total) + "."
print str((float(correct)/float(total))*100.0) + '%'

