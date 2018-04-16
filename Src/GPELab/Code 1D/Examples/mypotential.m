function pot = mypotential(x)

%x = hdf5read('/home/user/Study/Src/APPL/Src/xmds2/func.h5', 'x');
index = x * 128/20 + 65;
data = hdf5read('/home/user/Study/Src/APPL/Src/xmds2/func.h5', 'func');

pot = data(index);
end

