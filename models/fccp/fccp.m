function  [P, C, W, iter, T1] =fccp(Xo, Yo, beta, lambda, max_it, tol, viz, outliers, corresp,sigma2, t, nsc, normal)
X=Xo;Y=Yo;
[N, D]=size(X);  
%%%%%%%%%%%%%%%%%%%%%     begin clustering
Dist_X = pdist2(X,X);
if D==2
  wid=0.1; %0.18 %0.1 for 2D fish
else
  wid=0.12;
    wid=0.1;
%     wid=0.15;
%     wid=0.08;
%     wid=0.18;
%     wid=0.07;
%     wid=0.065;
%     wid=0.06;
%     wid=0.055;
end
width_X=wid*mean(mean(Dist_X )); %  width=0.18 %18 %.071

Dist_Y = pdist2(Y,Y);
width_Y=wid*mean(mean(Dist_Y )); %  width=0.18 %18 %.071

cluster_method = 'neighbor' %'neighborN' % % 'Kmean'     
data.X=X(:,1:D)';
data.width_X=width_X;
Xin0 = FCCP_cluster(data,cluster_method);  %Kmean neighbor neighborN

Xc=[];X0=[];Xin=[];
for i=1: max(Xin0)
     ct=find(Xin0==i);
     if size(ct,1)==1
        Xc=[Xc;X(ct,:)];
        X0=[X0;X(ct,:)];
        Xin=[Xin;i];
     else
       Xc=[Xc;mean(X(ct,:))];
       X0=[X0;X(ct,:)];
       Xin=[Xin;(i*ones(1,size(ct,1)))'];
     end
end

Yc=[];Y0=[];Yin=[];       
data.X=Y(:,1:D)';
data.width_X=width_Y;
Yin0 = FCCP_cluster(data,cluster_method);  %Kmean neighbor neighborN

for i=1: max(Yin0)
     ct=find(Yin0==i);
     if size(ct,1)==1
        Yc=[Yc;Y(ct,:)];
        Y0=[Y0;Y(ct,:)];
        Yin=[Yin;i];
     else
         Yc=[Yc;mean(Y(ct,:))];
         Y0=[Y0;Y(ct,:)];
         Yin=[Yin;(i*ones(1,size(ct,1)))'];
     end
end

%%%%%%%%%%%%  tmp
X0=Xo;
Y0=Yo;

Xin=Xin0;
Yin=Yin0;

%%%%%%%%%%%%  end tmp




X=Xc; 
Y=Yc;   
[N, D]=size(Xc); [M, D]=size(Yc);

%% Show cluster centers
if size(Xc,2)==3
    figure;
    view=[0,0,800,800];MarkerSize=8;
    my_plot3d(Xc,Yc,view,2,MarkerSize);
    axis off;
    axis equal;
    grid off;
    title('Clusters before registration');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  end cluster
K = size(Yc,1);   % 25;
% Initialization
T=Y; 
iter=0;  ntol=tol+10; W=zeros(K,D);
if ~exist('sigma2','var') || isempty(sigma2) || (sigma2==0), 
    sigma2=(M*trace(X'*X)+N*trace(Y'*Y)-2*sum(X)*sum(Y)')/(M*N*D);
end

tmp_Y = unique(Y, 'rows'); idx = randperm(size(tmp_Y,1)); 
idx = idx(1:min(K,size(tmp_Y,1)));
ctrl_pts=Yc;  %use templated points  lim
% Construct affinity matrix G
G = cpd_G(ctrl_pts,ctrl_pts,beta);
U = cpd_G(Y0, ctrl_pts, beta);

% configuration of SC
mean_dist_global=[]; 
nbins_theta=12;
nbins_r=5;
nsamp1=size(X,1);
nsamp2=size(Y,1);
ndum1=max(0, nsamp2-nsamp1);
ndum2=max(0, nsamp1-nsamp2);
eps_dum=0.00000001;%0.15;
r_inner=1/8;
r_outer=2;
% n_iter=30;
out_vec_1=zeros(1,nsamp1); 
out_vec_2=zeros(1,nsamp2);

c1=mean(X);
t1=atan2(X(:,2)-c1(2), X(:,1)-c1(1));
[BH1,mean_dist_1]=sc_compute(X',t1',mean_dist_global,nbins_theta,...
    nbins_r,r_inner,r_outer,out_vec_1);
iter=0; 
ntol=tol+10; 
L=1;

while (iter<max_it) && (ntol > tol) && (sigma2 > 1e-8) 
    if mod(iter, nsc)==0 % START SC
        c2=mean(T);
        t2=atan2(T(:,2)-c2(2), T(:,1)-c2(1));
        [BH2,mean_dist_2]=sc_compute(T',t2',mean_dist_1,nbins_theta,...
            nbins_r,r_inner,r_outer,out_vec_2);
        % compute pairwise cost between all shape contexts
        costmat=hist_cost_2(BH1,BH2);
        % pad the cost matrix with costs for dummies
        nptsd=nsamp1+ndum1;
        costmat2=eps_dum*ones(nptsd,nptsd);
        costmat2(1:nsamp1,1:nsamp2)=costmat;
        cvec=hungarian(costmat2);
        [Nc, D]=size(X); [Mc, D]=size(Y);    
        P_prior = zeros(Nc, Mc);
        if Nc > Mc
            for i = 1:Mc
                P_prior(cvec(i),i) = t;
            end
        else
            for i = 1:Nc
                P_prior(i,cvec(i)) = t;
            end
        end
        P_prior = P_prior + (1-t)/Mc;
        idx = sum(P_prior,2) < 1;
        P_prior(idx,:) = 1/size(Yin,1);     
    end       
    L_old=L;

%% E-Step
% compute_Pc-P: Projection between cluter Pc and point P.
    [P, P1, Pt1, PX, Pc, L]=compute_Pc_P(X,T, sigma2 ,outliers, P_prior, Xin, Yin, X0, Y0); st='';
%% M-Step   
%     disp([' FC nonrigid ' st ' : dL= ' num2str(ntol) ', iter= ' num2str(iter) ' sigma2= ' num2str(sigma2)]);  
    L=L+lambda/2*trace(W'*G*W);
    ntol=abs((L-L_old)/L);

    % start cluster
    [N, D]=size(X0); [M, D]=size(Y0);
    P1 = max(P1, 1e-3);
    dP=spdiags(P1,0,M,M); % precompute diag(P)

    W = (U'*dP*U + lambda*sigma2*G+0.0001*eye(size(G,1)))\(U'*PX-U'*dP*Y0);
    % update Y postions
    T1=Y0+U*W;
    Np=sum(P1);
    sigma2save=sigma2;  
    sigma2=0.5*abs((sum(sum(X0.^2.*repmat(Pt1,1,D)))+sum(sum(T1.^2.*repmat(P1,1,D))) -2*trace(PX'*T1)) /(Np*D));    
    %Point cluster Projection   
    
    [T, Yc] = Pc_proj(Yin,T1);  
         
%     figure(3);
%     clf;   
%     C1=T;
%     plot3(Xc(:,1),Xc(:,2),Xc(:,3),'b.','MarkerSize', 10);
%     hold on;
%     plot3(Yc(:,1),Yc(:,2),Yc(:,3),'r.','MarkerSize', 10);
%     plot3(C1(Yc_position,1),C1(Yc_position,2),C1(Yc_position,3),'r.','MarkerSize', 10);
    axis off;
    axis equal;
    % end cluster   
    outliers = 1 - Np/N; 
    if outliers>0.99
        outliers = 0.99;
    elseif outliers<0.01
        outliers = 0.01;
    end    
    iter=iter+1
end

% Show cluster centers
if size(Xc,2)==3
    figure;
    view=[0,0,800,800];MarkerSize=8;  
    my_plot3d(X,Yc,view,2,MarkerSize);
    axis off;
    axis equal;
    grid off;
    title('Clusters after registration');
end
%================= fine adaption for small dataset 
fine=1; 
if fine==1 && size(X0,1)<1000
iter0=iter+9;   
iter2=iter
Y=T1; T=T1; X=Xo;  P_prior=P;
W=zeros(M,D); 
G=cpd_G(Y,Y,beta);
mean_dist_global=[]; % use [] to estimate scale from the data
nbins_theta=12;
nbins_r=5;
nsamp1=size(X,1);
nsamp2=size(Y,1);
ndum1=max(0, nsamp2-nsamp1);
ndum2=max(0, nsamp1-nsamp2);
eps_dum=0.15;
r_inner=1/8;
r_outer=2;
n_iter=30;
out_vec_1=zeros(1,nsamp1); 
out_vec_2=zeros(1,nsamp2);

[BH1,mean_dist_1]=sc_compute(X',zeros(1,nsamp1),mean_dist_global,nbins_theta,...
    nbins_r,r_inner,r_outer,out_vec_1);

    while (iter<iter0) && (ntol > 0.1*tol) && (sigma2 > 1e-8) 
        if iter0-iter==9 
            [BH2,mean_dist_2]=sc_compute(T',zeros(1,nsamp2),mean_dist_1,nbins_theta,...
                nbins_r,r_inner,r_outer,out_vec_2);
            % compute pairwise cost between all shape contexts
            costmat=hist_cost_2(BH1,BH2);
            % pad the cost matrix with costs for dummies
            nptsd=nsamp1+ndum1;
            costmat2=eps_dum*ones(nptsd,nptsd);
            costmat2(1:nsamp1,1:nsamp2)=costmat;
            cvec=hungarian(costmat2);

            P_prior = zeros(N, M);

            if N > M
                for i = 1:M
                    P_prior(cvec(i),i) = t;
                end
            else
                for i = 1:N
                    P_prior(i,cvec(i)) = t;
                end
            end
            P_prior = P_prior + (1-t)/M;
            idx = sum(P_prior,2) < 1;
            P_prior(idx,:) = 1/M;
        end
        L_old=L;
        [P, P1, Pt1, PX, L]=compute_P(X,T, sigma2 ,outliers, P_prior); st='';
        L=L+lambda/2*trace(W'*G*W);
        ntol=abs((L-L_old)/L);
        dP=spdiags(P1,0,M,M); % precompute diag(P)
        W=(dP*G+lambda*sigma2*eye(M))\(PX-dP*Y);
        % update Y postions
        
        T=Y+G*W;

%         Y_1 = back_normalize(normal, Y);
%         T_1= back_normalize(normal,T);
%         X_1 = back_normalize_x(normal,X);
%         Trans_final = T_1-Y_1;
       
        Np=sum(P1);sigma2save=sigma2;
        sigma2=abs((sum(sum(X.^2.*repmat(Pt1,1,D)))+sum(sum(T.^2.*repmat(P1,1,D))) -2*trace(PX'*T)) /(Np*D));

        outliers = 1 - Np/N; 
        if outliers>0.99
            outliers = 0.99;
        elseif outliers<0.01
            outliers = 0.01;
        end
        T1=T;
        % Plot the result on current iteration
        
        if viz, figure(2); cpd_plot_iter(X, T); end;
        iter=iter+1;
    end
end

% disp([' FCCP nonrigid ' st ' : dL= ' num2str(ntol) ', iter= ' num2str(iter) ' sigma2= ' num2str(sigma2)]);
% disp('FCCP registration succesfully completed.');
if corresp, C=cpd_Pcorrespondence(X,T1,sigma2save,outliers); else C=0; end;

