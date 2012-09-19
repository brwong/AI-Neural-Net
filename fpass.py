#!/usr/bin/python


def genFib():
	vals = [0, 1]
	yield 0
	yield 1
	while True:
		vals.append(vals[-2] + vals[-1])
		yield vals[-1]

