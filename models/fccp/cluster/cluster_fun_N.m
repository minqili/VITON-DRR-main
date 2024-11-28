function index=cluster_fun_N(X,Nc)
%%X: 2 x N  , index: N x 1
N=size(X,2);
C=X(:,1);
% Nc=1;
% sd0=d0^2;
% sd=sd0+1;
% while sd>sd0
for ii=1:Nc
    dif=repmat(X,[1,1,ii])-repmat(permute(C,[1,3,2]),[1,N,1]);
    sqdist=permute(sum(dif.^2),[2,3,1]);
    mndist=min(sqdist,[],2);
    sd=max(mndist);
    newind=find(mndist==sd);
    C=[C X(:,newind(1))];
%     Nc=Nc+1;
end

% dif=repmat(X,[1,1,Nc])-repmat(permute(C,[1,3,2]),[1,N,1]);
% sqdist=permute(sum(dif.^2),[2,3,1]);
% mndist=min(sqdist,[],2);
index=zeros(N,1);
for ii=1:N
    tmp=find(sqdist(ii,:)==mndist(ii));
    index(ii)=tmp(1);
end

