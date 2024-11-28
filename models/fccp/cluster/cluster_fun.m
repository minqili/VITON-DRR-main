function index=cluster_fun(X,d0)
%%% clustering of D dimensional point set, d0: the maximum width of clusters
%%% X: D x N  , here D is dimension, index: N x 1, the serial number of a cluster a point belong to
N=size(X,2); %%% set cardinality
C=X(:,1); %%% first cluster center
Nc=1; %%% number of clusters
sd0=d0^2; %%% cluster width uplimit
sd=sd0+1; %%% initial cluster width
%%%generate cluster centers
while sd>sd0
    dif=repmat(X,[1,1,Nc])-repmat(permute(C,[1,3,2]),[1,N,1]);
    sqdist=permute(sum(dif.^2),[2,3,1]); %%% squared distance between points and cluster centers, N x Nc
    mndist=min(sqdist,[],2); %% N x 1 ,distance of points to nearest cluster center
    sd=max(mndist); %% the max of above distance
    newind=find(mndist==sd); %% the index of new cluster center
    C=[C X(:,newind(1))];  %% append the new cluster center
    Nc=Nc+1; %% increment the number of clusters
end

dif=repmat(X,[1,1,Nc])-repmat(permute(C,[1,3,2]),[1,N,1]);%%%redundant
sqdist=permute(sum(dif.^2),[2,3,1]);%%%redundant %% squared distance between point and cluster center, N x Nc
mndist=min(sqdist,[],2);%%%redundant %% N x 1
index=zeros(N,1);
for ii=1:N
    tmp=find(sqdist(ii,:)==mndist(ii));
    index(ii)=tmp(1);
end

