function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;             %n=3？？
    m = size(x, 1);       %m为训练数据的组数，一次性训练完后进行MSE评价
    
    x = [ones(m,1) x];    %ones(m,1)生成m*1的全1阵
    nn.a{1} = x;          %可直接向结构体中添加元素项，此处为对nn.a的初始化（nn.a{1}为n*2的矩阵）  %a1为输入层，直接为样本输入，为计算下层输入做准备

    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
%                 nn.a{i - 1}
%                  nn.W{i - 1}'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');   %此处为隐含层输出，sigmoid作为激活函数
%                case 'linear'
%                 nn.a{i} = nn.a{i - 1} * nn.W{i - 1}';
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term  添加偏置量
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    switch nn.output   %输出层激活函数？？
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');  %最后一个输出节点的输出值
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
%             nn.L = sum(nn.e .^ 2)/ m 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
