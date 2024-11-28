function A = back_normalize(K)
n = size(K,1);


shape = [191 -255];
A = K .* repmat(shape,n,1);
acc = [0 255];
A = repmat(acc, n,1) + A;
end