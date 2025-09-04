classdef DeepESN_PCA < DeepESN
    % PCADeepESN - 带层间PCA降维的DeepESN子类
    % 实现方式：上层状态x(t)经PCA降维后，加偏置传入下层
    
    properties
        % PCA相关参数
        pca_components   % 各层PCA组件（cell array）
        pca_means        % 各层数据均值（cell array）
        pca_dims         % 各层保留维度 [Nl-1 x 1]
        pca_variance     % 保留方差比例（默认0.95）
    end
    
    methods
        function self = DeepESN_PCA()
            % 构造函数
            self = self@DeepESN();  % 调用父类构造函数
            self.pca_variance = 0.5;
            self.pca_dims = [];
            self.initialize();
        end
        
        function init_pca(self, num_layers)
            % 初始化PCA存储结构
            self.pca_components = cell(num_layers-1, 1);
            self.pca_means = cell(num_layers-1, 1);
            self.pca_dims = zeros(1, self.Nl-1);
        end
        
        function prepare_pca(self, training_input)
            % 准备PCA变换（需在训练前调用）
            % 步骤1：收集各层状态数据
            self.init_state();
            Nt = size(training_input, 2);
            layer_states = cell(self.Nl, 1);
            
            % 临时运行网络收集状态
            for l = 1:self.Nl
                layer_states{l} = zeros(self.Nr, Nt);
            end
            current_state = self.state;
            
            for t = 1:Nt
                for l = 1:self.Nl
                    % 原始状态更新逻辑
                    x = current_state{l};
                    if l == 1
                        u = training_input(:, t);
                        input_part = self.Win * [u; self.bias];
                    else
                        u_prev = layer_states{l-1}(:, t);
                        input_part = self.Wil{l} * [u_prev; self.bias];
                    end
                    new_state = (1-self.leaking_rate)*x + ...
                        self.leaking_rate*self.f(input_part + self.W{l}*x);
                    layer_states{l}(:, t) = new_state;
                    current_state{l} = new_state;
                end
            end
            
            % 步骤2：训练各层PCA模型
            self.init_pca(self.Nl);  % 初始化存储结构
            for l = 1:self.Nl-1
                % 收集有效数据
                raw_data = layer_states{l}';
                
                % 计算PCA
                [coeff, ~, ~, ~, explained] = my_pca(raw_data);
                cum_var = cumsum(explained);
                k = find(cum_var >= self.pca_variance*100, 1);
                
                % 存储PCA参数
                self.pca_components{l} = coeff(:, 1:k);
                self.pca_means{l} = mean(raw_data, 1);


                self.pca_dims(l) = k;
                
                % 调整下层输入权重维度
                self.Wil{l+1} = self.initialize_pca_weights(l+1, k);
            end
        end
        
        function W = initialize_pca_weights(self, layer, input_dim)
            % 初始化带PCA的层间权重
            W = 2*rand(self.Nr, input_dim + 1) - 1;  % +1 for bias
            switch self.input_scaling_mode
                case 'bynorm'
                    W = self.inter_scaling * W / norm(W);
                case 'byrange'
                    W = W * self.inter_scaling;
            end
        end
        
        function states = run(self, input)
            % 重写run方法，加入PCA处理
            Nt = size(input, 2);
            states = cell(self.Nl, 1);
            for l = 1:self.Nl
                states{l} = zeros(self.Nr, Nt);
            end
            
            current_state = self.state;
            for t = 1:Nt
                for l = 1:self.Nl
                    % 状态更新逻辑
                    x = current_state{l};
                    if l == 1
                        % 第一层处理
                        u = input(:, t);
                        input_part = self.Win * [u; self.bias];
                    else
                        % 上层状态PCA变换
                        prev_state = states{l-1}(:, t);
                        proj_state = self.apply_pca(prev_state, l-1);
                        
                        % 组合PCA结果与偏置
                        pca_input = [proj_state; self.bias];
                        input_part = self.Wil{l} * pca_input;
                    end
                    
                    % 更新状态
                    new_state = (1-self.leaking_rate)*x + ...
                        self.leaking_rate*self.f(input_part + self.W{l}*x);
                    states{l}(:, t) = new_state;
                    current_state{l} = new_state;
                end
            end
            self.run_states = states;
            self.state = current_state;
        end
        
        function projected = apply_pca(self, state, layer)
            % 应用PCA变换
            if layer <= length(self.pca_components) && ~isempty(self.pca_components{layer})
                centered = state' - self.pca_means{layer};
                projected = (centered * self.pca_components{layer})';
            else
                projected = state;  % 未启用PCA时保持原样
            end
        end
    end
end