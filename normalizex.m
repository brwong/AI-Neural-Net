load testx.txt
normalized = zeros(2500,5000);
for i = 1:2500
temp = zeros(200,25);
for j = 1:200
temp(j,:) = testx(i,j:j+24);
end;
temp = normc(temp);
for j = 1:200
normalized(i,j:j+24) = temp(j,:);
end;
end;
csvwrite('normx.txt', normalized);
