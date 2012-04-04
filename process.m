load trainx.txt
load trainy.csv
load testx.txt
knn = knnclassify(testx, trainx, trainy);
csvwrite('testy.txt', knn)
