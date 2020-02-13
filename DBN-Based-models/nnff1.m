function AA = nnff1(nn, x)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

%     n = nn.n;    
    n=3;
    m = size(x, 1);       %mΪѵ�����ݵ�������һ����ѵ��������MSE����
    
    x = [ones(m,1) x];    %ones(m,1)����m*1��ȫ1��
    nn.a{1} = x;          %��ֱ����ṹ�������Ԫ����˴�Ϊ��nn.a�ĳ�ʼ����nn.a{1}Ϊn*2�ľ���  %a1Ϊ����㣬ֱ��Ϊ�������룬Ϊ�����²�������׼��

    %feedforward pass
%     for i = 2 : n-1
%         switch nn.activation_function 
%             case 'sigm'
% %                 nn.a{i - 1}
% %                  nn.W{i - 1}'
%                 % Calculate the unit's outputs (including the bias term)
%                 nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');   %�˴�Ϊ�����������sigmoid��Ϊ�����
% %                case 'linear'
% %                 nn.a{i} = nn.a{i - 1} * nn.W{i - 1}';
%             case 'tanh_opt'
%                 nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
%         end
nn.a{2} = sigm(nn.a{1} * nn.W{ 1}');
        
        %Add the bias term  ���ƫ����
        nn.a{2} = [ones(m,1) nn.a{2}];


AA=sigm(nn.a{2} * nn.W{2}');    
    
    

