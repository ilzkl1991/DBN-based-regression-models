function AA = nnff1(nn, x)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

%     n = nn.n;    
    n=3;
    m = size(x, 1);       %m为训练数据的组数，一次性训练完后进行MSE评价
    
    x = [ones(m,1) x];    %ones(m,1)生成m*1的全1阵
    nn.a{1} = x;          %可直接向结构体中添加元素项，此处为对nn.a的初始化（nn.a{1}为n*2的矩阵）  %a1为输入层，直接为样本输入，为计算下层输入做准备

    %feedforward pass
%     for i = 2 : n-1
%         switch nn.activation_function 
%             case 'sigm'
% %                 nn.a{i - 1}
% %                  nn.W{i - 1}'
%                 % Calculate the unit's outputs (including the bias term)
%                 nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');   %此处为隐含层输出，sigmoid作为激活函数
% %                case 'linear'
% %                 nn.a{i} = nn.a{i - 1} * nn.W{i - 1}';
%             case 'tanh_opt'
%                 nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
%         end
nn.a{2} = sigm(nn.a{1} * nn.W{ 1}');
        
        %Add the bias term  添加偏置量
        nn.a{2} = [ones(m,1) nn.a{2}];


AA=sigm(nn.a{2} * nn.W{2}');    
    
    

