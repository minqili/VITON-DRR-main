function A = back_normalize_x(normal, K)
n = size(K,1);
A = K*normal.xscale;
A = A+repmat(normal.xd,n,1);
shape = [191 255];
A = A .* repmat(shape,n,1);