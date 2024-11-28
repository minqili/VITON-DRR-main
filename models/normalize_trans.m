function A = normalize_trans(K)
n = size(K,1);

acc = [0 255];
shape = [191 -255];
A_1 = K .* repmat(shape,n,1);
A = A_1 + repmat(acc, n, 1);
end