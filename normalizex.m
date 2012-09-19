%load data
load trainx.txt
%create matrix to fill
normalized = zeros(2500,5000);
%loop through each song
for i = 1:2500
	%create a temporary matrix for segments
	temp = zeros(200,25);
	temp2 = zeros(200,25);
	%load segments into the temp matrix
	for j = 1:200
		temp(j,:) = trainx(i,25*(j-1)+1:25*(j-1)+25);
	end;
	%temp = normc(temp);
	temp = smooth(temp);
	for j = 1:25
		temp2(:,j) = temp(200*(j-1)+1:200*(j-1)+200,:);
	end;
	for j = 1:200
		normalized(i,25*(j-1)+1:25*(j-1)+25) = temp2(j,:);
	end;
end;
csvwrite('normx.txt', normalized);
