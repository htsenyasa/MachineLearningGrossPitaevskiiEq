function res = runmatlab(inter, alpha)

%%% This file is an example of how to use GPELab (FFT version)

%% Ground state of a Gross-Pitaevskii equation with quadratic potential and cubic nonlinearity in 1D

addpath(genpath("/home/user/Study/Src/APPL/Src/GPELab"));
inter = str2double(inter);
alpha = str2double(alpha);


%-----------------------------------------------------------
% Setting the data
%-----------------------------------------------------------

%% Setting the method and geometry
Computation = 'Ground';
Ncomponents = 1;
Type = 'BESP';
Deltat = 1e-2;
Stop_time = [];
Stop_crit = {'Energy',1e-12};
Method = Method_Var1d(Computation, Ncomponents, Type, Deltat, Stop_time, Stop_crit);
xmin = -10;
xmax = 10;
Nx = 128 + 1;
Geometry1D = Geometry1D_Var1d(xmin,xmax,Nx);

%% Setting the physical problem
Delta = alpha;
Beta = inter;
Physics1D = Physics1D_Var1d(Method, Delta, Beta);
Physics1D = Dispersion_Var1d(Method, Physics1D);
%Physics1D = Potential_Var1d(Method, Physics1D, @(X)X.^2/2 + 25*sin(X*pi/4).^2);
Physics1D = Potential_Var1d(Method, Physics1D, @(X)mypotential(X));
Physics1D = Nonlinearity_Var1d(Method, Physics1D);

%% Setting the initial data
InitialData_Choice = 1;
Phi_0 = InitialData_Var1d(Method, Geometry1D, Physics1D, InitialData_Choice);

%% Setting informations and outputs
save = 0;
Outputs = OutputsINI_Var1d(Method, save);
Printing = 1;
Evo = 15;
Draw = 1;
Print = Print_Var1d(Printing,Evo,Draw);

%-----------------------------------------------------------
% Launching simulation
%-----------------------------------------------------------
[Phi, Outputs] = GPELab1d(Phi_0,Method,Geometry1D,Physics1D,Outputs,[])%,Print);
phi = Phi{1};
dens = phi .* phi;
en = Outputs.Energy{1};
en = en(length(en));

file_name = "/home/user/Study/Src/APPL/data/nonlinearSE/generic_dataset/matlab/";
dlmwrite(file_name + "dens.txt", dens);
dlmwrite(file_name + "energy.txt", en);
exit;

end

