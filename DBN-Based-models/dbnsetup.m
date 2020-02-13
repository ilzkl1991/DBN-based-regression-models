function dbn = dbnsetup(dbn, x, opts)
% dbnsetup(dbn, train_x, opts);      主函数中的调用方法
    n = size(x, 2);                  %训练时输入数据的个数
    dbn.sizes = [n, dbn.sizes];      %dbn.sizes为RBM隐层个数

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

%         dbn.rbm{u}.W  = rand(dbn.sizes(u + 1), dbn.sizes(u));
%         dbn.rbm{u}.vW = rand(dbn.sizes(u + 1), dbn.sizes(u));
% 
%         dbn.rbm{u}.b  = rand(dbn.sizes(u), 1);
%         dbn.rbm{u}.vb = rand(dbn.sizes(u), 1);
% 
%         dbn.rbm{u}.c  = rand(dbn.sizes(u + 1), 1);
%         dbn.rbm{u}.vc = rand(dbn.sizes(u + 1), 1);
%         
        
        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
