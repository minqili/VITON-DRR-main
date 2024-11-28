%============================================
% This demo shows 2D points registration results by the algorithm from the paper:
% "Fast non-rigid points registration with cluster correspondences projection".
% Verison 0: 14/4/2019
%============================================
function t1 = FCCP_demo2D()
clear;
clc;
close all;
addpath('./datamat');
addpath('./results');
addpath(genpath('fccp'))
%%======================================================================
% load data
%----------Generate data----
load_data=2;

Ti=1; Te=1;

if load_data==2
for k1=Ti:1
    for k2=Te:1%20
%       tmp_name=['save_fish_outlier_',num2str(k1),'_',num2str(k2),'.mat'];
     % tmp_name=['save_fish_occlusion_',num2str(k1),'_',num2str(k2),'.mat'];        
%       tmp_name=['save_fish_noise_',num2str(k1),'_',num2str(k2),'.mat'];        
%       tmp_name=['save_fish_def_',num2str(k1),'_',num2str(k2),'.mat'];


      load ("data1.mat");
      X = x1; Y = y2a;  

    rand_correspondence=0
    if rand_correspondence==1
        Y0=Y;
        p = randperm(size(Y,1),size(Y,1));    
        Y = Y0(p,:);
    end

    add_noise=0
    if add_noise==1
        noise = sqrt(j-1)*randn(size(Y));  
        Y=Y+noise;
    end  

    figure(1), 
    cpd_plot_iter(X, Y);   % axis off;

    Dim=size(Y,2); %%%dimension
    N=size(X,1);M=size(Y,1); %%% cardinality 

    t1a=clock;
    opt.viz = 1;
    opt.outliers = 0
    opt.t = 1-  size(X,1)/size(Y,1);%0.9;
    if opt.t>0
      opt.outliers = opt.t; % 0.5; 
    end
    opt.sparse = 1;
    opt.nsc = 5;

    t0=cputime;
    [Transform, C, normal]=fccp_register(Y, X, opt);
    t1=cputime-t0;
     Transform.X=Transform.Y;
     V = Transform.X;
     t1b=clock;
    
     V_final = normalize_trans(V);
     X_final = normalize_trans(X);
     X_1 = back_normalize(X);
     final_trans = cat(2,X_1,V_final);
     writematrix(final_trans, ".\\txtdata\\data01.txt");
     figure(2),cpd_plot_iter(Transform.X, Y); axis off; title('After registering Y to X');
   end
end

end


