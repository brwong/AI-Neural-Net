#!/usr/bin/python

from math import exp
from sys import stdout

def loadtrainx():
	f = open('normx.txt', 'r')
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

def scalar(a, v):
	return [a * n for n in v]

def times(a, b):
	return sum(map((lambda x,y:x*y), a, b))

def sig(z):
	if -z > 709:
		return 0
	return float(1)/(1 + exp(-z))

def sigmoid(w, x):
	return sig(times(w,x))

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


def through(inp,hid,out,x):
	for v in range(len(x)):
		inp[v].feed(x[v])
	for h in hid:
		h.forward()
	for o in out:
		o.forward()
	return [o.output for o in out]

def loader():
	#load data
	print "loading x..."
	trainx = loadtrainx()
	print "got x."
	print "loading y..."
	trainy = loadtrainyun()
	print "got y."
	#defined layers and create nodes
	inp = [node() for i in range(5000)]
	hid = [node() for i in range(3)]
	out = [node() for i in range(5)]
	for n in range(len(inp)):
		if not n % 25:
			inp[n].connectTo(hid[0], 0.2)
		elif n % 25 <= 12:
			inp[n].connectTo(hid[1], 0.2)
		else:
			inp[n].connectTo(hid[2], 0.2)
	for h in hid:
		for o in out:
			h.connectTo(o, 0.2)
	return inp, hid, out, trainx, trainy

def process(inp, hid, out, xs, ys):
	print "got "+str(len(inp))+" inputs"
	print "got "+str(len(hid))+" hidden"
	print "got "+str(len(out))+" output"
	inp, hid, out = train(inp, hid, out, xs[:1500], ys, 5)
	return inp, hid, out, xs, ys

def train(inp, hid, out, xs, ys, passes):
	for epoch in range(passes):
		print ""
		print "epoch", epoch
		for song in range(len(xs)):
			stdout.write(str(epoch).rjust(3,' ')+":"+str(song).ljust(4,' ')+"... ")
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
				h.update(0.4)
			for o in out:
				o.update(0.4)
		print [o.output for o in out], ys[song]
	return inp, hid, out

def main():
	return process(*loader())

if __name__=='__main__':
	inp, hid, out, trainx, trainy = main()
	#pass

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

