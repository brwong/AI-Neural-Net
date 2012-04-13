%-- 12-04-13 04:23:48 PM --%
matlabpool
load trainx.txt
load trainy.csv
load testx.txt
firstsegtrain = trainx(:,1:25);
net = network;
net.numInputs = 1;
net.inputs{1}.size = 25;
net.numLayers = 2;
net.layers{1}.size = 3;
net.layers{2}.size = 1;
net.inputConnect(1) = 1;
net.layerConnect(2,1) = 1;
net.outputConnect(2) = 1;
net.targetConnect(2) = 1;
net.layers{1}.transferFcn = 'logsig';
net.biasConnect = [1;1];
net.layers{1}.initFcn
net.inputWeights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net.biases{2}.initFcn = 'rands';
net.layerWeights{2,1}.initFcn = 'rands';
net.performFcn = 'mae';
net.trainFcn = 'trainlm';
net = init(net);
[net,tr] = train(net, firstsegtrain', trainy');
checky = net(firstsegtrain');
verif = [trainy checky');
