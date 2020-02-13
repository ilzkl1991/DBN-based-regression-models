function BPout=rbf1(x)
%global net; 
 load('outputps.mat')
 load('inputps.mat')
 
% y=mapminmax('apply',x,inputps);
load ('-mat','RBF1');

%网络预测输出
an=sim(net,x);

%网络输出反归一化
BPout=mapminmax('reverse',an',outputps);
 end


