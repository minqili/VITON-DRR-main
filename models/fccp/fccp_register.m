function [Transform, C, normal]=fccp_register(X, Y, opt)
[M,D]=size(Y); [N, D2]=size(X);
% Check the input options and set the defaults
if nargin<3, error('Not enough input parameters.'); end;
if ~isfield(opt,'normalize') || isempty(opt.normalize), opt.normalize = 1; end;
if ~isfield(opt,'max_it') || isempty(opt.max_it), opt.max_it = 120; end;  %100
if ~isfield(opt,'tol') || isempty(opt.tol), opt.tol = 1e-5; end; %1e-5
if ~isfield(opt,'viz') || isempty(opt.viz), opt.viz = 0; end;
if ~isfield(opt,'corresp') || isempty(opt.corresp), opt.corresp = 1; end;
if ~isfield(opt,'outliers') || isempty(opt.outliers), opt.outliers = 0.1; end;
if ~isfield(opt,'sigma2') || isempty(opt.sigma2), opt.sigma2 = 0; end;
if ~isfield(opt,'sparse') || isempty(opt.sparse), opt.sparse = 0; end;
if ~isfield(opt,'beta') || isempty(opt.beta), opt.beta = 2; end;
if ~isfield(opt,'lambda') || isempty(opt.lambda), opt.lambda = 10; end;  %20>10>1  % BY LIM ????
if ~isfield(opt,'nsc') || isempty(opt.nsc), opt.nsc = 10; end;

% checking for the possible errors
if D~=D2, error('The dimension of point-sets is not the same.'); end;
if (D>M)||(D>N), disp('The dimensionality is larger than the number of points. Possibly the wrong orientation of X and Y.'); end;
if (D<=1) || (D>3), opt.viz=0; end;

% Convert to double type, save Y
X=double(X);  
Y=double(Y); Yorig=Y; 

% default mean and scaling
normal.xd=0; normal.yd=0;
normal.xscale=1; normal.yscale=1;

% Normalize to zero mean and unit variance
if opt.normalize, 
    [X,Y,normal]=cpd_normalize(X,Y); 
end;

opt.lambda=opt.lambda*(1+2*opt.outliers);
if opt.sparse
  [P, C, W, iter, T] = fccp(X, Y, opt.beta, opt.lambda, opt.max_it, opt.tol, opt.viz, opt.outliers, opt.corresp, opt.sigma2, opt.t, opt.nsc, normal);  
end

Transform.iter=iter;
Transform.Y=T;
Transform.normal=normal;
Transform.P=P;

Transform.beta=opt.beta;
Transform.W=W;
Transform.Yorig=Yorig;
Transform.s=1;
Transform.t=zeros(D,1);
if opt.normalize,
     Transform.Y=T*normal.xscale+repmat(normal.xd,size(T,1),1);
end 






