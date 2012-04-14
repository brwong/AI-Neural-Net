%-- 12-04-13 04:23:48 PM --%
load trainx.txt
load trainy.csv
load trainyun.csv
load testx.txt
firstsegtrain = trainx(:,1:25);
first4segtrain = trainx(:,1:100);
net = network;
net.numInputs = 1;
%net.inputs{1}.size = 25;
net.inputs{1}.size = 100;
net.numLayers = 3;
net.layers{1}.size = 12;
net.layers{2}.size = 8;
net.layers{3}.size = 5;
net.inputConnect(1) = 1;
net.layerConnect(2,1) = 1;
net.layerConnect(3,2) = 1;
net.outputConnect(3) = 1;
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';
net.biasConnect = [1;1;1];
net.inputWeights{1,1}.initFcn = 'rands';
net.inputWeights{2,1}.initFcn = 'rands';
net.inputWeights{3,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net.biases{2}.initFcn = 'rands';
net.biases{3}.initFcn = 'rands';
net.layerWeights{2,1}.initFcn = 'rands';
net.layerWeights{3,2}.initFcn = 'rands';
%net.performFcn = 'mae';
net.performFcn = 'mse';
net.trainFcn = 'trainlm';
net = init(net);
%[net,tr] = train(net, firstsegtrain', trainyun');
[net,tr] = train(net, first4segtrain', trainyun');
checky = net(first4segtrain');
verif = [trainy checky'];
