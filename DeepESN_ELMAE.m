classdef DeepESN_ELMAE < handle
    %DeepESN: 集成ELM-AE的深度回声状态网络
    properties
        % 原有属性
        Nr, Nu, Ny, Nl               
        spectral_radius, input_scaling, inter_scaling, leaking_rate  
        input_scaling_mode, f, bias, washout       
        readout_regularization        
        Win, Wil, W, Wout             
        run_states, initial_state, state 
        
        % 新增ELM-AE属性
        elm_mappings      % 各层ELM-AE映射矩阵（cell数组）       % MODIFIED
        dim_encoder       % 各层降维维度（例如[50,30]表示第1层降至50维，第2层30维） % MODIFIED
    end
    
    methods (Access = public)
        function self = default(self)
            % 原有默认值
            self.Nr = 10;    self.Nu = 1;     self.Ny = 1;    self.Nl = 10;   
            self.spectral_radius = 0.9;  self.input_scaling = 1;   self.inter_scaling = 1;  
            self.leaking_rate = 1;        self.input_scaling_mode = 'bynorm';  
            self.f = @tanh;  self.bias = 1;   self.washout = 1000;  
            self.readout_regularization = 0;  
            self.Win = [];   self.Wil = cell(self.Nl,1);  self.W = cell(self.Nl,1);  
            self.Wout = [];  self.run_states = cell(self.Nl,1);  
            self.initial_state = [];  self.state = cell(self.Nl,1);  
            
            % 新增ELM-AE默认值                                      % MODIFIED
            self.dim_encoder = repmat(10, 1, self.Nl-1);  % 除最后一层外，每层降维至10维
            self.elm_mappings = cell(self.Nl-1, 1);       % 每层ELM-AE映射存储
        end
        
        function self = DeepESN_ELMAE()
            self.default();
        end
        
        function init_state(self)
            % 原有状态初始化
            for layer = 1:self.Nl
                self.state{layer} = self.initial_state;
                self.run_states{layer} = [];
            end
        end
        
        function initialize(self)
            % 初始化储备池（修正Wil矩阵维度）                     % FIXED
            assert(length(self.dim_encoder) == self.Nl-1, 'dim_encoder长度必须为Nl-1');
            
            % 原有权重初始化逻辑（修改Wil部分）
            self.Win = 2*rand(self.Nr,self.Nu+1)-1;
            switch self.input_scaling_mode
                case 'bynorm'
                    self.Win = self.input_scaling * self.Win / norm(self.Win);
                case 'byrange'
                    self.Win = self.Win * self.input_scaling;
            end
            
            for i = 2:self.Nl
                % 根据前一层降维后的维度调整Wil矩阵大小
                prev_dim = self.dim_encoder(i-1); % 前一层降维后的维度
                self.Wil{i} = 2*rand(self.Nr, prev_dim + 1) - 1; % (Nr, prev_dim+1)
                switch self.input_scaling_mode
                    case 'bynorm'
                        self.Wil{i} = self.inter_scaling * self.Wil{i}/norm(self.Wil{i});
                    case 'byrange'
                        self.Wil{i} = self.Wil{i} * self.inter_scaling;
                end
            end
            
            for i = 1:self.Nl
                self.W{i} = 2*sprandn(self.Nr,self.Nr, 0.1)-1;
                I = eye(self.Nr);
                Wt = (1-self.leaking_rate)*I + self.leaking_rate*self.W{i};
                Wt = self.spectral_radius * Wt / max(abs(eig(Wt)));
                self.W{i} = (Wt - (1-self.leaking_rate)*I)/self.leaking_rate;
            end
            
            self.initial_state = zeros(self.Nr,1);
            self.init_state();
        end
        
        function states = run(self, input, is_training)
            % 运行储备池（新增ELM-AE处理和模式开关）                    % MODIFIED
            if nargin < 3
                is_training = true; % 默认训练模式
            end
            
            Nt = size(input,2);
            for layer = 1:self.Nl
                self.run_states{layer} = zeros(self.Nr,Nt);
            end
            
            old_state = self.state;
            for t = 1:Nt
                for layer = 1:self.Nl
                    x = old_state{layer};
                    if layer == 1
                        u = input(:,t);
                        input_part = self.Win * [u;self.bias];
                    else
                        % 应用ELM-AE降维                                % MODIFIED
                        if is_training
                            prev_states = self.run_states{layer-1}(:,1:t);
                        else
                            prev_states = self.run_states{layer-1};
                        end
                        u = self.apply_elm_mapping(prev_states, layer-1, t, is_training);
                        input_part = self.Wil{layer} * [u;self.bias];
                    end
                    self.state{layer} = (1-self.leaking_rate)*x + self.leaking_rate*self.f(input_part + self.W{layer}*x);
                    self.run_states{layer}(:,t) = self.state{layer};
                    old_state{layer} = self.state{layer};
                end
            end
            states = self.run_states;
        end
        
        function self = train_readout(self, target)
            X = DeepESN_ELMAE.shallow_states(self.run_states, self.Nl, self.Nr);
            X = X(:,self.washout+1:end);
            target = target(:,self.washout+1:end);
            X = [X;self.bias * ones(1,size(X,2))];

            self.Ny = size(self.Wout,1); % 显式设置输出维度
            if self.readout_regularization == 0
                self.Wout = target * pinv(X);
            else
                self.Wout = target * X' / (X*X' + self.readout_regularization*eye(size(X,1)));
            end
        end
        
        function output = compute_output(self, states, remove_washout)
            % 计算输出（保持原有逻辑）
            states = DeepESN_ELMAE.shallow_states(states, self.Nl, self.Nr); 
            if remove_washout
                states = states(:,self.washout+1:end);
            end
            output = self.Wout * [states;self.bias * ones(1,size(states,2))];
        end
        
        function [outputTR, outputTS] = train_test(self, input, target, training, test)
            % 训练测试（新增ELM-AE映射训练）                           % MODIFIED
            self.init_state();
            
            % 训练阶段
            training_input = input(:,training);
            training_target = target(:,training);
            training_states = self.run(training_input, true); % 训练模式生成ELM映射
            self.train_elm_mappings(training_states);          % 训练各层ELM-AE       % MODIFIED
            self.train_readout(training_target);
            outputTR = self.compute_output(training_states, true);
            
            % 测试阶段
            test_input = input(:,test);
            test_states = self.run(test_input, false); % 测试模式应用ELM映射
            outputTS = self.compute_output(test_states, false);
        end
        
        % 新增ELM-AE相关方法                                        % MODIFIED
        function self = train_elm_mappings(self, states)
            % 训练各层ELM-AE映射
            for layer = 1:self.Nl-1
                layer_states = states{layer}(:,self.washout+1:end)';
                [~, self.elm_mappings{layer}] = compute_mapping_elmae(layer_states, self.dim_encoder(layer));
            end
        end
        
        function u = apply_elm_mapping(self, prev_states, layer_idx, t, is_training)
            % 应用ELM-AE映射
            if is_training
                % 训练阶段在线计算
                mapped = compute_mapping_elmae(prev_states', self.dim_encoder(layer_idx));
            else
                % 测试阶段使用预存映射
                mapped = out_of_sample_elmae(prev_states', self.elm_mappings{layer_idx});
            end
            u = mapped(t,:)'; % 提取当前时间步的降维状态
        end
    end
    
    methods (Static)
        % 保持原有静态方法
        function perf = MSE(target, output)
            perf = mean((target-output).^2);
        end
        
        function [perf,d] = MCscore(target, output)
            delays = size(target,1);
            for delay = 1:delays
                c = corrcoef(target(delay,:),output(delay,:));
                d(delay) = c(1,2)^2;
            end
            perf = sum(d);
        end
        
        function X = shallow_states(states, Nl, Nr)
            Nt = size(states{1},2);
            X = zeros(Nl * Nr, Nt);
            for t = 1:Nt
                for layer = 1:Nl
                    idx = (layer-1)*Nr + 1 : layer*Nr;
                    X(idx,t) = states{layer}(:,t);
                end
            end
        end
    end
end