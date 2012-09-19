%load data
load testx.txt
%create matrix to fill
normalizedtest = zeros(2500,5000);
%loop through each song
for i = 1:2500
	%create a temporary matrix for segments
	temp = zeros(200,25);
	%load segments into the temp matrix
	for j = 1:200
		temp(j,:) = testx(i,25*(j-1)+1:25*(j-1)+25);
	end;
	temp = smooth(temp)';
	for j = 1:200
		normalizedtest(i,25*(j-1)+1:25*(j-1)+25) = temp(j,:);
	end;
end;
csvwrite('norm.txt', normalizedtest);
