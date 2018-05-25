function pot = mypotential(x)

%x = hdf5read('/home/user/Study/Src/APPL/Src/xmds2/func.h5', 'x');
N = 256;
index = x * N/20 + (N/2 + 1);
data = hdf5read('/home/user/Study/Src/APPL/Src/xmds2/func.h5', 'func');

pot = data(index);
end

