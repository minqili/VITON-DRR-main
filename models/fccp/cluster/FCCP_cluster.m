function Xin=FCCP_cluster(data,method)
% clustering of D dimensional point set, d0: the maximum width of clusters
% X: D x N  , here D is dimension, index: N x 1, the serial number of a cluster a point belong to
% FCCP_cluster(X(:,1:D)','neighbor',width_X);
X=data.X;
N=size(X,2);
switch method          
    case 'neighbor'
        %nearest neighbor clustering 
        width=data.width_X;
        Xin=cluster_fun(X,width);              
    case 'neighborN'
        rate=4*data.width_X;
        Xin=cluster_fun_N(X,floor(rate*N)); %%%Dim=2 here
    case 'Kmean'
        rate=4*data.width_X;
        Xin= kmeans(X',floor(rate*N));%%returning the cluster indices of each point
    case 'Cmean'
        rate=data.width_X;
        [center,U,obj_fcn]=fcm(X(:,1:2),floor(rate*N));
        maxU = max(U);
        Xin=zeros(N,1);
        for ii=1:N
            tmp=find(U(:,ii)==maxU(ii));
            Xin(ii)=tmp(1);
        end
end

test0=1

