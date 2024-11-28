function [P, P1, Pt1, PX, Pc, L]=compute_Pc_P(X,T, sigma2 ,outlier, s_matrix, Xin, Yin, X0, Y0)
% s_matrix= P_prior
[N, D] = size(X);
M = size(T, 1);
% s_matrix = ones(N, M) / M;
if D == 2
    a = (max(X(:,1)) - min(X(:,1)))*(max(X(:,2)) - min(X(:,2)));
    b = (max(T(:,1)) - min(T(:,1)))*(max(T(:,2)) - min(T(:,2)));
else
    a = (max(X(:,1)) - min(X(:,1)))*(max(X(:,2)) - min(X(:,2)))*(max(X(:,3)) - min(X(:,3)));
    b = (max(T(:,1)) - min(T(:,1)))*(max(T(:,2)) - min(T(:,2)))*(max(T(:,3)) - min(T(:,3)));
end
a = max(a, b);
ksig = -2.0 * sigma2;   % BY LIM
outlier_tmp=outlier*(-ksig*3.14159265358979)^(0.5*D)/((1-outlier)*a);
% outlier_tmp=(1-outlier)*(-ksig*3.14159265358979)^(0.5*D)/(outlier*a);  %by LIM

P = repmat(X,[1 1 M])-permute(repmat(T,[1 1 N]),[3 2 1]);
P = squeeze(sum(P.^2,2));
P = P/ksig;
P = exp(P).*s_matrix;

%%%clusters to points conversion
Pc=P;
P=Pc(Xin,Yin); 
%%%%%%%%%% LIM TEST IMPORTANT
sum_tmp = sum(Pc,2) + outlier_tmp;  %by LIM
Pc=Pc./repmat(sum_tmp,1,size(T,1)); % normalization such that each column sums to 1
% Pc=Pc';
%%%%%%%%%%

[N, D]=size(X0); [M, D]=size(Y0);
% s=sum(P,2); 
% P=P./repmat(s,1,M); % normalization such that each row sums to 1
% sum_tmp=sum(P,2);  
sum_tmp = sum(P,2) + outlier_tmp;  %by LIM
P=P./repmat(sum_tmp,1,M); % normalization such that each column sums to 1
P=P';
%         s=sum(P,2); 
%         P=P./repmat(s,1,N); % normalization such that each row sums to 1
P1 = P*ones(N,1);
% s_tmp=sum(Pc,2); 
% sum_tt = sum(Pc,2) + outlier_tmp;  %by LIM
% Pc=Pc./repmat(sum_tt,1,size(T,1));
% Pc=Pc';
% P1_tmp=Pc*ones(size(X,1),1);
Pt1 = P'*ones(M,1);
PX = P*X0;

% P1=P1/max(P1); % BY LIM
% Pt1 =Pt1/max(Pt1); %by LIM
% sum_tmp = sum(P,2) + outlier_tmp;
% P = P./repmat(sum_tmp, 1, M);
% P = P';
% P1 = P*ones(N,1);
% Pt1 = P'*ones(M,1);
% PX = P*X;
% L = sum(log(sum_tmp)) + D*N*log(sigma2)/2;      
               
%%%%%%%%%%% end cluster
    
    
% sum_tmp = sum(P,2) + outlier_tmp;
% P = P./repmat(sum_tmp, 1, M);
% P = P';
% P1 = P*ones(N,1);
% Pt1 = P'*ones(M,1);
% % PX = P*X;
% PX = P*X0;

L = sum(log(sum_tmp)) + D*N*log(sigma2)/2;