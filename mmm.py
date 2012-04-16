#!/usr/bin/python

from math import exp
from sys import stdout

## loading functions

def loadtrainx():
	f = open('mmm.txt', 'r')
	vals = [[float(d) for d in l.strip().split(' ')] for l in f.readlines()]
	f.close()
	return vals

def loadtestx():
	f = open('mmmtest.txt', 'r')
	vals = [[float(d) for d in l.strip().split(' ')] for l in f.readlines()]
	f.close()
	return vals

def loadtrainy():
	f = open('trainy.csv', 'r')
	vals = [float(d.strip()) for d in f.readlines()]
	f.close()
	return vals

def loadtrainyun():
	f = open('trainyun.csv', 'r')
	vals = [[int(d) for d in l.strip().split(' ')] for l in f.readlines()]
	f.close()
	return vals

## vector operations

def scalar(a, v):
	return [a * n for n in v]

def times(a, b):
	return sum(map((lambda x,y:x*y), a, b))

## sigmoid functions

def sig(z):
	if -z > 709:
		return 0
	return float(1)/(1 + exp(-z))

def sigmoid(w, x):
	return sig(times(w,x))

# primary node class
# the neural network is composed of these
# it passes data forward and propagates backward
class node():
	def __init__(self, name = ""):
		#self.weights = []
		self.biasweight = 0.2
		self.inputs = []
		#self.invites = []
		self.invites = {}
		# inside is an object:
		# weight, delta
		self.output = None
		self.history = []
		self.delta = None
		self.destinations = []
		self.name = name
		#self.mytype = mytype
	#def enter(self, who, val):
	#	if who in self.invites:
	#		self.inputs[self.invites[who]] = val
	def connectTo(self, other, w):
		self.destinations.append(other)
		other.invites[self] = {}
		other.invites[self]['weight'] = w
	def clearHistory(self):
		self.history = []
		self.output = None
	def feed(self, val):
		self.output = val
		self.history.append(val)
		if len(self.history) > 3:
			self.history.pop(0)
	def forward(self):
		self.output = None
		inputs = [1]
		weights = [self.biasweight]
		for i in self.invites:
			if i.output == None:
				print 'no output for node '+i.name
				return False
			inputs.append(i.output)
			weights.append(self.invites[i]['weight'])
		self.output = sigmoid(inputs, weights)
		self.history.append(self.output)
		if len(self.history) > 3:
			self.history.pop(0)
		return True
	def back(self, val):
		out = self.history[-1]
		self.delta = out * (1 - out) * (val - out)
	def backward(self):
		out = self.history[-1]
		total = 0
		for d in self.destinations:
			total += (d.invites[self]['weight'])*d.delta
		self.delta = out * (1 - out) * total
	def update(self, learnrate):
		self.biasweight = self.biasweight + learnrate * self.delta * 1
		for i in self.invites:
			self.invites[i]['weight'] = self.invites[i]['weight'] + learnrate * self.delta * i.output


# convert unary value arrays to decimal numbers
# it does this by taking the highest number and returning its position
def untodec(un):
	dec = []
	for u in un:
		for i in range(len(u)):
			found = -1
			if round(u[i]) == 1:
				if found != -1:
					found = -1
					break
				found = i
		if found == -1:
			m = max(u)
			for j in range(len(u)):
				if u[j] == m:
					found = j
					break
		dec.append(found)
	return dec

# pass several tuples through the network
def setthrough(inp,hid,out,xs):
	ys = []
	for x in xs:
		ys.append(through(inp,hid,out,x))
	return ys

# pass one tuple into the network, return output
def through(inp,hid,out,x):
	for v in range(len(x)):
		inp[v].feed(x[v])
	for h in hid:
		h.forward()
	for o in out:
		o.forward()
	return [o.output for o in out]

# retrieve and return training data;
# create and return neural network layers
def loader():
	#load data
	print "loading x..."
	trainx = loadtrainx()
	print "got x."
	print "loading y..."
	trainy = loadtrainyun()
	print "got y."
	#defined layers and create nodes
	inp = [node() for i in range(75)]
	hid = [node() for i in range(3)]
	out = [node() for i in range(5)]
	for n in range(len(inp)):
		for h in hid:
			inp[n].connectTo(h, 0.2)
		##volume
		#if n <= 2:
		#	if n == 2:
		#		inp[n].connectTo(hid[0], 0.2)
		#	else:
		#		inp[n].connectTo(hid[0], 0.1)
		##pitch
		#elif n <= 36:
		#	if n % 3 == 2:
		#		inp[n].connectTo(hid[1], 0.2)
		#	else:
		#		inp[n].connectTo(hid[1], 0.1)
		##timbre
		#else:
		#	if n % 3 == 2:
		#		inp[n].connectTo(hid[1], 0.2)
		#	else:
		#		inp[n].connectTo(hid[1], 0.1)
	for h in hid:
		h.biasweight = 0.1
		for o in out:
			h.connectTo(o, 0.1)
			o.biasweight = 0.1
	return inp, hid, out, trainx, trainy

# experimentation area (different number of training cycles, etc.)
def process(inp, hid, out, xs, ys):
	print "got "+str(len(inp))+" inputs"
	print "got "+str(len(hid))+" hidden"
	print "got "+str(len(out))+" output"
	print "GOOD LUCK!!"
	#inp, hid, out = train(inp, hid, out, xs, ys, 1)
	#"""
	testx = loadtestx()
	totpasses = 0
	for i in range(15):
		totpasses += 1
		inp, hid, out = train(inp, hid, out, xs, ys, 1)
		yss = untodec([through(inp,hid,out,x) for x in xs])
		f = open('f'+str(totpasses)+'.csv', 'w')
		for y in yss:
			f.write(str(y)+"\n")
		f.close()
		yss = untodec([through(inp,hid,out,x) for x in testx])
		f = open('testyf'+str(totpasses)+'.csv', 'w')
		for y in yss:
			f.write(str(y)+"\n")
		f.close()
		print "did "+str(totpasses)+" f passes."
	#"""
	return inp, hid, out, xs, ys

#feed numbers in, back propagate, update weights
def train(inp, hid, out, xs, ys, passes):
	for epoch in range(passes):
		print ""
		print "epoch", epoch
		for song in range(len(xs)):
			stdout.write(str(epoch).rjust(3,' ')+":"+str(song).ljust(4,' ')+"... ")
			if not song%10:
				print ""
			#input values, feed forward
			for d in range(len(xs[song])):
				inp[d].feed(xs[song][d])
			for h in hid:
				h.forward()
			for o in out:
				o.forward()
			#input targets, back prop
			for targval in range(len(ys[song])):
				out[targval].back(ys[song][targval])
			for h in hid:
				h.backward()
			for h in hid:
				h.update(0.1)
			for o in out:
				o.update(0.1)
		print [o.output for o in out], ys[song]
	return inp, hid, out

def main():
	return process(*loader())

if __name__=='__main__':
	inp, hid, out, trainx, trainy = main()
	#pass

## this is a demonstration of the neural network
## featured in homework 7, and its operation
def exampleFromHw7(passes = 1):
	#define layers and create nodes
	inp = [node(i) for i in ["n1", "n2"]]
	hid = [node(i) for i in ["n3", "n4"]]
	out = [node(i) for i in ["n5", "n6", "n7"]]
	for n in inp:
		for h in hid:
			n.connectTo(h, 0.2)
	for h in hid:
		for o in out:
			h.connectTo(o, 0.2)
	#input values, feed forward
	for jj in range(passes):
		inp[0].feed(0)
		inp[1].feed(1)
		for h in hid:
			h.forward()
		for o in out:
			o.forward()
		#input targets, back prop
		out[0].back(1)
		out[1].back(0)
		out[2].back(0)
		for h in hid:
			h.backward()
		for h in hid:
			h.update(0.4)
		for o in out:
			o.update(0.4)
	#test the trained network
	inp[0].feed(0)
	inp[1].feed(1)
	for h in hid:
		h.forward()
	for o in out:
		o.forward()
	print "number of passes: "+str(passes)+" results:"
	print [o.output for o in out]

