clc;
clear;
rng(2025);  % Set random seed for reproducibility

%% Load weather data and preprocess (preserve original column names)
T = readtable('weather.csv', 'VariableNamingRule', 'preserve');

% Record total number of rows before date removal
fprintf('Total rows in original data (including date column): %d\n', size(T, 1));

% Remove date column (keep only numerical features)
T(:, 'date') = [];
fprintf('Total rows after removing date column: %d\n', size(T, 1));
fprintf('%s\n', repmat('=', 1, 60));  % Separator line

% Separate feature columns (all columns as input features) and target column (OT column for next time step prediction)
feature_cols = T;  % Features: all meteorological indicators (including OT)
target_col = T.OT;  % Target: OT column (actual value to predict for next time step)

% Convert to numerical matrices (rows = time steps, columns = features)
features = table2array(feature_cols);  % Dimensions: [total_time_steps, num_features]
targets = target_col;                  % Dimensions: [total_time_steps, 1]

%% Outlier detection and handling for OT column only (to avoid prediction bias)
target_mean = mean(targets);
target_std = std(targets);
% 3σ criterion: values deviating more than 3 standard deviations from mean are outliers
outliers_target = abs(targets - target_mean) > 3 * target_std;

% Display OT column outlier information
fprintf('【OT Column (Target Data) Outlier Information】\n');
fprintf('Outlier criterion: Deviation from OT mean exceeds 3 standard deviations\n');
num_target_outliers = sum(outliers_target);
fprintf('Total rows with OT column outliers: %d\n', num_target_outliers);
if num_target_outliers > 0
    fprintf('Row indices of OT outliers (after date removal): ');
    fprintf('%d ', find(outliers_target));
    fprintf('\n');
else
    fprintf('No outliers detected in OT column\n');
end
fprintf('%s\n', repmat('=', 1, 60));  % Separator line

% Remove rows with OT outliers, ensuring feature-target time alignment
features = features(~outliers_target, :);  % Dimensions: [valid_time_steps, num_features]
targets = targets(~outliers_target);        % Dimensions: [valid_time_steps, 1]
fprintf('Remaining data rows (time steps) after removing OT outliers: %d\n', size(features, 1));
fprintf('%s\n', repmat('=', 1, 60));  % Separator line

%% Min-max normalization (retain only normalization)
% Feature normalization (column-wise, each feature scaled independently to [0,1])
min_vals = min(features, [], 1);  % Minimum value for each feature
max_vals = max(features, [], 1);  % Maximum value for each feature
features_normalized = (features - min_vals) ./ (max_vals - min_vals + eps);  % Add eps to avoid division by zero

% Target normalization (scale to [0,1] for direct evaluation on this scale)
min_target = min(targets);
max_target = max(targets);
targets_normalized = (targets - min_target) ./ (max_target - min_target + eps);

%% Sliding window processing (core: create "historical window → next time step" training pairs)
window_size = 24;    % Sliding window size (use previous 24 time steps to predict 25th)
stride = 1;          % Sliding step (move 1 time step each slide)
[windowed_input, windowed_target] = create_sliding_windows(features_normalized, targets_normalized, window_size, stride);

% Display windowed data dimensions (verify input-target alignment)
fprintf('【Sliding Window Processing Results (Next Time Step Prediction)】\n');
fprintf('Windowed input dimensions: Features × Window size × Number of windows = %d × %d × %d\n', size(windowed_input));
fprintf('Windowed target dimensions: 1 × Number of windows (each window corresponds to next time step actual value) = %d × %d\n', size(windowed_target));
fprintf('Window meaning: Use features from previous %d time steps to predict OT value at time step %d (normalized)\n', window_size, window_size+1);
fprintf('%s\n', repmat('=', 1, 60));  % Separator line

%% Reshape data format (adapt to DeepESN input requirements: [features×window_size, number_of_windows])
num_features = size(windowed_input, 1);    % Total number of features
window_size_actual = size(windowed_input, 2);  % Actual window size (=24)
num_windows = size(windowed_input, 3);     % Total number of windows (= valid_time_steps - window_size)

% Input reshaping: flatten 3D window (features×window×count) to 2D (features×window_size, number_of_windows)
input_data_reshaped = reshape(windowed_input, num_features * window_size_actual, num_windows);
% Target remains: 1×number_of_windows (each window corresponds to 1 next time step target, normalized)
target_data = windowed_target;

% Display reshaped data dimensions
fprintf('【Data Reshaping Results】\n');
fprintf('Reshaped input data dimensions: (Number of features × Window size) × Number of windows = %d × %d\n', size(input_data_reshaped));
fprintf('Target data dimensions: 1 × Number of windows (next time step actual value, normalized) = %d × %d\n', size(target_data));
fprintf('%s\n', repmat('=', 1, 60));  % Separator line

%% Split into training and testing sets (critical: time-ordered split, training first, testing later)
train_ratio = 0.8;
% Split by number of windows (windows generated in time order, first 80% for training, last 20% for testing)
train_window_num = floor(num_windows * train_ratio);
test_window_num = num_windows - train_window_num;

% Window indices (ensure temporal continuity: training windows 1~train_window_num, testing windows train_window_num+1~num_windows)
train_range = 1:train_window_num;
test_range1 = train_window_num :( num_windows - 1);
test_range = (train_window_num + 1):num_windows;

% Split training/testing data (input-target strictly aligned, both normalized)
train_input = input_data_reshaped(:, train_range);  % Training input: first 80% of windows
train_target = target_data(:, train_range);        % Training target: corresponding next time step actual values for first 80% windows
test_input = input_data_reshaped(:, test_range1);    % Testing input: last 20% of windows (historical data)
test_target = target_data(:, test_range);          % Testing target: corresponding next time step actual values for last 20% windows (to be verified)

% Display split results (verify temporal continuity)
fprintf('【Training and Testing Set Split Results (Next Time Step Prediction)】\n');
fprintf('Training window range: Window 1 ~ Window %d (corresponding to original time steps 1 ~ %d)\n', ...
    train_window_num, train_window_num + window_size);
fprintf('Testing window range: Window %d ~ Window %d (corresponding to original time steps %d ~ %d)\n', ...
    train_window_num+1, num_windows, train_window_num + window_size + 1, num_windows + window_size);
fprintf('Training set input dimensions: %d × %d\n', size(train_input));
fprintf('Training set target dimensions: 1 × %d (next time step actual values, normalized)\n', size(train_target, 2));
fprintf('Testing set input dimensions: %d × %d\n', size(test_input));
fprintf('Testing set target dimensions: 1 × %d (next time step actual values, normalized)\n', size(test_target, 2));
fprintf('Number of training windows: %d, Number of testing windows: %d\n', train_window_num, test_window_num);
fprintf('%s\n', repmat('=', 1, 60));  % Separator line

%% Sliding window function definition (core: generate "window input → next time step target")
function [windowed_input, windowed_target] = create_sliding_windows(input_series, target_series, window_size, stride)
    num_time_steps = size(input_series, 1);  % Total number of time steps (rows = time steps)
    num_features = size(input_series, 2);    % Number of features (columns = features)
    
    % Calculate total number of windows (ensure each window has corresponding next time step target)
    num_windows = floor((num_time_steps - window_size) / stride);
    
    % Preallocate memory (input: features×window_size×number_of_windows; target: 1×number_of_windows)
    windowed_input = zeros(num_features, window_size, num_windows);
    windowed_target = zeros(1, num_windows);
    
    % Generate windows by sliding
    for i = 1:num_windows
        % Time step indices for window (current window: start_idx ~ end_idx)
        start_idx = (i-1)*stride + 1;
        end_idx = start_idx + window_size - 1;
        
        % Window input: extract features for current window (transpose to "features×window_size")
        windowed_input(:, :, i) = input_series(start_idx:end_idx, :)';
        
        % Window target: actual value at "next time step" after current window (end_idx + 1, normalized)
        windowed_target(i) = target_series(end_idx + 1);
    end
end

%% Parameter optimization settings (PSO hyperparameter search range, adapted for next time step prediction)
param_ranges = {
    {'Nr',  'integer',    [50, 200]},      % Number of neurons per reservoir layer (avoid overfitting with large values)
    {'Nl',  'integer',    [2, 8]},         % Number of reservoir layers (DeepESN depth)
    {'spectral_radius', 'continuous', [0.6, 0.95]}, % Spectral radius (<1 ensures stable dynamics)
    {'leaking_rate',    'continuous', [0.2, 0.8]},  % Leaking rate (controls state update speed)
    {'compression_ratio', 'continuous', [0.3, 0.8]} % Compression ratio (if ICSDESN has feature compression)
};

n_particles = 8;     % Number of particles (balance between speed and accuracy)
max_iter = 5;        % Number of iterations (avoid excessive searching)

%% PSO optimization core function (adapted for normalized data, next time step prediction)
function [best_params, best_mse] = pso_optimize_deepcsesn(param_ranges, data, target, train_range, test_range, n_particles, max_iter)
    % Initialize particle swarm
    particles = initialize_particles(param_ranges, n_particles);
    
    % Initialize global best
    global_best_mse = Inf;
    global_best_params = [];
    
    % Progress bar
    h = waitbar(0,'PSO optimization progress...');
    
    for iter = 1:max_iter
        % Changed from parfor to for loop for sequential processing
        for i = 1:n_particles  
            % Decode particle parameters
            params = decode_particle(particles(i), param_ranges);
            
            % Evaluate model (MSE for next time step prediction on normalized data)
            current_mse = evaluate_deepcsesn(params, data, target, train_range, test_range);
            
            % Update personal best
            if current_mse < particles(i).best_mse
                particles(i).best_mse = current_mse;
                particles(i).best_position = particles(i).position;
            end
        end
        
        % Update global best
        [current_best_mse, idx] = min([particles.best_mse]);
        if current_best_mse < global_best_mse
            global_best_mse = current_best_mse;
            global_best_params = decode_particle(particles(idx), param_ranges);
        end
        
        % Update particle positions and velocities
        particles = update_particles(particles, global_best_params, param_ranges);
        
        % Update progress bar
        waitbar(iter/max_iter, h, sprintf('Iteration %d/%d, Current best MSE (normalized): %.6f', iter, max_iter, global_best_mse));
    end
    close(h);
    best_params = global_best_params;
    best_mse = global_best_mse;
end

%% Helper functions (particle initialization, decoding, updating)
function particles = initialize_particles(param_ranges, n_particles)
    template = struct('position', [], 'velocity', [], 'best_position', [], 'best_mse', Inf);
    particles = repmat(template, 1, n_particles);
    
    for i = 1:n_particles
        position = zeros(1, length(param_ranges));
        for p = 1:length(param_ranges)
            param = param_ranges{p};
            switch param{2}
                case 'integer'  % Integer parameters (e.g., Nr, Nl)
                    position(p) = randi(param{3});
                case 'continuous'  % Continuous parameters (e.g., spectral radius, leaking rate)
                    position(p) = param{3}(1) + rand()*(param{3}(2)-param{3}(1));
            end
        end
        particles(i).position = position;
        particles(i).velocity = zeros(1, length(param_ranges));
        particles(i).best_position = position;
    end
end

function params = decode_particle(particle, param_ranges)
    params = struct();
    for p = 1:length(param_ranges)
        param = param_ranges{p};
        params.(param{1}) = particle.position(p);
    end
end

function particles = update_particles(particles, global_best, param_ranges)
    inertia_weight = 0.5;    % Inertia weight (controls influence of historical velocity)
    cognitive_weight = 1.5;  % Cognitive weight (influence of personal best)
    social_weight = 1.5;     % Social weight (influence of global best)
    
    % Construct global best parameter vector (strictly matches param_ranges order)
    global_best_vector = [
        global_best.Nr,
        global_best.Nl,
        global_best.spectral_radius,
        global_best.leaking_rate,
        global_best.compression_ratio
    ];
    
    for i = 1:length(particles)
        % Velocity update
        r1 = rand(1, length(param_ranges));
        r2 = rand(1, length(param_ranges));
        particles(i).velocity = inertia_weight * particles(i).velocity + ...
            cognitive_weight * r1 .* (particles(i).best_position - particles(i).position) + ...
            social_weight * r2 .* (global_best_vector - particles(i).position);
        
        % Position update
        particles(i).position = particles(i).position + particles(i).velocity;
        
        % Boundary constraints (ensure parameters stay within reasonable ranges)
        for p = 1:length(param_ranges)
            param = param_ranges{p};
            if strcmp(param{2}, 'integer')
                particles(i).position(p) = round(particles(i).position(p));  % Round integer parameters
            end
            particles(i).position(p) = max(particles(i).position(p), param{3}(1));  % Lower bound
            particles(i).position(p) = min(particles(i).position(p), param{3}(2));  % Upper bound
        end
    end
end

%% Model evaluation function (based on normalized data, next time step prediction)
function mse = evaluate_deepcsesn(params, data, target, train_range, test_range)
    % Initialize ICSDESN model (assuming ICSDESN interface matches DeepESN)
    net = ICSDESN();
    net.Nu = size(data, 1);  % Input dimension = number of features × window size
    net.Nr = params.Nr;      % Number of neurons per layer
    net.Nl = params.Nl;      % Number of layers
    net.spectral_radius = params.spectral_radius;
    net.leaking_rate = params.leaking_rate;
    net.compression_ratio = params.compression_ratio;
    net.washout = 50;        % Transient period (window level, avoid initial unstable states)
    net.readout_regularization = 1e-10;  % Regularization (prevent overfitting)
    net.initialize();

    % Training: learn "window input → next time step" mapping (normalized data)
    train_states = net.run(data(:, train_range(1:end-1)));  % Run training windows
    net.train_readout(target(:, train_range(2:end)));     % Train output layer weights

    % Testing: predict next time step using test windows (normalized data)
    test_states = net.run(data(:, test_range(1:end-1)));    % Run test windows (historical data)
    test_output = net.compute_output(test_states, true);  % Predict output (discard transient period)
    
    % Extract valid test targets (align dimension with predicted output, normalized)
    valid_test_target = target(:, test_range(net.washout+2:end));
    
    % Calculate MSE (based on normalized data)
    mse = mean((valid_test_target - test_output).^2, 'all');
end

% Helper function to calculate R-squared (coefficient of determination)
function r2 = calculate_r2(predicted, actual)
    ss_total = sum((actual - mean(actual, 'all')).^2, 'all');  % Total sum of squares
    ss_residual = sum((actual - predicted).^2, 'all');         % Residual sum of squares
    
    % Avoid division by zero for constant data
    if ss_total < 1e-10
        r2 = 0;
    else
        r2 = 1 - (ss_residual / ss_total);  % R-squared formula
    end
end

%% Run optimization (execute PSO optimization)
fprintf('\n=== Starting PSO Hyperparameter Optimization (Next Time Step Prediction) ===\n');
optimizationStart = tic;

[best_params, best_mse] = pso_optimize_deepcsesn(...
    param_ranges, input_data_reshaped, target_data, train_range, test_range, n_particles, max_iter);

optimizationTime = toc(optimizationStart);
fprintf('PSO optimization time: %.2f minutes\n', optimizationTime/60);

% Output optimal parameters
disp('===== PSO Optimization Results =====');
disp('Optimal hyperparameters:');
disp(best_params);
disp(['Optimal test MSE (normalized): ', num2str(best_mse, '%.6f')]);

%% Train final model with optimized parameters
net_final = ICSDESN();
net_final.Nu = size(input_data_reshaped, 1);
net_final.Nr = best_params.Nr;
net_final.Nl = best_params.Nl;
net_final.spectral_radius = best_params.spectral_radius;
net_final.leaking_rate = best_params.leaking_rate;
net_final.compression_ratio = best_params.compression_ratio;
net_final.washout = 200;  % Transient period (consistent with evaluation function)
net_final.readout_regularization = 1e-10;
net_final.initialize();

% Training phase
fprintf('\n=== Training Final Model with Optimal Parameters ===\n');
train_tic = tic;
train_states = net_final.run(input_data_reshaped(:, train_range(1:end-1)));
net_final.train_readout(target_data(:, train_range(2:end)));
train_time = toc(train_tic);
fprintf('Training time: %.3f seconds\n', train_time);

% Testing phase (next time step prediction, based on normalized data)
test_tic = tic;
test_states = net_final.run(input_data_reshaped(:, test_range(1:end-1)));
test_output = net_final.compute_output(test_states, true);  % Predicted output (normalized, transient period discarded)
test_time = toc(test_tic);

% Extract valid data (ensure predicted output aligns with target dimensions, both normalized)
valid_idx = net_final.washout + 2 : size(test_range, 2);
test_output_valid = test_output;  % Predicted output already has transient period discarded, use directly
test_target_valid = test_target(:, valid_idx);  % Extract valid portion of test targets

% Calculate prediction metrics (based on normalized data)
error = test_target_valid - test_output_valid;
abs_error = abs(error);
squared_error = error.^2;
epsilon = 1e-6;
percentage_error = abs_error ./ (abs(test_target_valid) + epsilon);  % Avoid division by zero

% Core metrics (all based on normalized scale)
final_mse = mean(squared_error, 'all');
final_rmse = sqrt(final_mse);
final_mae = mean(abs_error, 'all');
final_mape = mean(percentage_error, 'all') * 100;
final_r2 = calculate_r2(test_output_valid, test_target_valid);  % Calculate R-squared

% Output final performance (clearly labeled "normalized")
disp('===== Final Model Performance (Next Time Step Prediction, Normalized) =====');
disp(['Test MSE (normalized):     ', num2str(final_mse, '%.6f')]);
disp(['Test RMSE (normalized):    ', num2str(final_rmse, '%.6f')]);
disp(['Test MAE (normalized):     ', num2str(final_mae, '%.6f')]);
disp(['Test MAPE:                ', num2str(final_mape, '%.2f'), '%']);  % MAPE is scale-independent
disp(['Test R²:                  ', num2str(final_r2, '%.6f')]);  % R-squared value
disp(['Testing time:              ', num2str(test_time, '%.3f'), ' seconds']);

%% 100-run statistics (verify model stability, based on normalized data)
num_runs = 100;
mse_list = zeros(1, num_runs);
rmse_list = zeros(1, num_runs);
mae_list = zeros(1, num_runs);
mape_list = zeros(1, num_runs);
r2_list = zeros(1, num_runs);  % Store R-squared values

% Progress bar
h = waitbar(0, 'Running 100 model iterations... 0%');

for i = 1:num_runs
    % Initialize model
    net = ICSDESN();
    net.Nu = size(input_data_reshaped, 1);
    net.Nr = best_params.Nr;
    net.Nl = best_params.Nl;
    net.spectral_radius = best_params.spectral_radius;
    net.leaking_rate = best_params.leaking_rate;
    net.compression_ratio = best_params.compression_ratio;
    net.washout = 50;
    net.readout_regularization = 1e-11;
    net.initialize();

    % Training
    train_states = net.run(input_data_reshaped(:, train_range(1:end-1)));
    net.train_readout(target_data(:, train_range(2:end)));
    
    % Testing (next time step prediction, normalized data)
    test_states = net.run(input_data_reshaped(:, test_range(1:end-1)));
    test_output_norm = net.compute_output(test_states, true);
    
    % Extract valid targets and calculate metrics (normalized)
    valid_test_target = target_data(:, test_range(net.washout+2:end));
    error = valid_test_target - test_output_norm;
    
    % Calculate and store R-squared for this run
    current_r2 = calculate_r2(test_output_norm, valid_test_target);
    
    % Store results
    mse_list(i) = mean(error.^2, 'all');
    rmse_list(i) = sqrt(mse_list(i));
    mae_list(i) = mean(abs(error), 'all');
    mape_list(i) = mean(abs(error)./(abs(valid_test_target)+epsilon), 'all')*100;
    r2_list(i) = current_r2;  % Store R-squared value
    
    % Update progress bar
    waitbar(i/num_runs, h, sprintf('Running 100 model iterations... %.1f%%', i/num_runs*100));
end
close(h);

% Statistical results (based on normalized data)
avg_mse = mean(mse_list);
avg_rmse = mean(rmse_list);
avg_mae = mean(mae_list);
avg_mape = mean(mape_list);
avg_r2 = mean(r2_list);  % Average R-squared

std_mse = std(mse_list);
std_rmse = std(rmse_list);
std_mae = std(mae_list);
std_mape = std(mape_list);
std_r2 = std(r2_list);  % Standard deviation of R-squared

disp('===== 100-Run Statistical Results (Next Time Step Prediction, Normalized) =====');
fprintf('Average test MSE (normalized):  %.6f (Std: %.6f)\n', avg_mse, std_mse);
fprintf('Average test RMSE (normalized): %.6f (Std: %.6f)\n', avg_rmse, std_rmse);
fprintf('Average test MAE (normalized):  %.6f (Std: %.6f)\n', avg_mae, std_mae);
fprintf('Average test MAPE:             %.2f%% (Std: %.2f%%)\n', avg_mape, std_mape);
fprintf('Average test R²:               %.6f (Std: %.6f)\n', avg_r2, std_r2);


