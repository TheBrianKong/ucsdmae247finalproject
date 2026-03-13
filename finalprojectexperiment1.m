% MAE 247: ICF vs CKF vs KCF vs GKCF - Figure 4 Recreation
% Includes CRN (Common Random Numbers) to eliminate jitter s.t. differences only come from algorithms
clear; clc; close all;
if isempty(gcp('nocreate'))
    parpool; % Starts the parallel workers
end
%% Experiment Parameters
N = 15;                 % number of sensors
Delta = 2;              % degree of communication network
SR = 300;               % sensor range
t_max = 40;             % time steps
epsilon = 0.65 / Delta; % Consensus speed parameter
gridsize = 500;

% Monte Carlo
num_envs = 20;           
num_tracks = 20;         
K_max = 20;             

H = [1 0 0 0; 0 1 0 0];
Phi = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];
Q = diag([10, 10, 1, 1]); 
R = 100 * eye(2);
B = inv(R);

%% Communication Graph Setup
Adj = zeros(N, N);
for d = 1:(Delta/2)
    Adj = Adj + diag(ones(N-d, 1), d) + diag(ones(N-d, 1), -d);
    Adj = Adj + diag(ones(d, 1), N-d) + diag(ones(d, 1), -(N-d));
end

%% Monte Carlo Loop
mean_error_icf_K = zeros(1, K_max);  std_icf_K = zeros(1, K_max);
mean_error_ckf_K = zeros(1, K_max);  std_ckf_K = zeros(1, K_max);
mean_error_kcf_K = zeros(1, K_max);  std_kcf_K = zeros(1, K_max);
mean_error_gkcf_K = zeros(1, K_max); std_gkcf_K = zeros(1, K_max);

fprintf('Starting Monte Carlo Simulation...\n');

for K_test = 1:K_max
    fprintf('Testing K = %d...\n', K_test);
    
    total_error_icf = 0;  total_sq_error_icf = 0;
    total_error_ckf = 0;  total_sq_error_ckf = 0;
    total_error_kcf = 0;  total_sq_error_kcf = 0;
    total_error_gkcf = 0; total_sq_error_gkcf = 0;
    num_data_points = 0; 
    
    parfor env = 1:num_envs
        rng(env * 1000); 
        
        sensor_pos = gridsize * rand(2, N);
        sensor_theta = 2 * pi * rand(1, N);
        
        for trk = 1:num_tracks
            rng(env * 1000 + trk); 
            
            % target 
            x_true_init = [gridsize/2; gridsize/2; 2*cos(rand*2*pi); 2*sin(rand*2*pi)]; 
            target_track = zeros(4, t_max);
            current_x = x_true_init;
            for t = 1:t_max
                current_x = Phi * current_x + chol(Q)' * randn(4,1);
                target_track(:, t) = current_x;
            end
            
            % init equal prior (fig 4)
            P_prior_init = diag([100, 100, 10, 10]);
            J_prior_init = inv(P_prior_init);
            
            x_prior_icf = zeros(4, N);
            J_prior_icf = repmat(J_prior_init, 1, 1, N);
            
            shared_prior_noise = chol(P_prior_init)' * randn(4,1);
            for i = 1:N
                x_prior_icf(:, i) = x_true_init + randn(4,1);%shared_prior_noise; 
            end
            x_prior_ckf = x_prior_icf(:, 1);
            J_prior_ckf = J_prior_init;
            
            x_prior_kcf = x_prior_icf;
            J_prior_kcf =J_prior_icf;
            
            x_prior_gkcf = x_prior_icf;
            J_prior_gkcf = J_prior_icf;
            
            % time
            for t= 1:t_max
                x_true = target_track(:, t);
                U_all = zeros(4, 4, N);
                u_all = zeros(4, N);
                
                for i = 1:N
                    vec_to_target = x_true(1:2) - sensor_pos(:, i);
                    dist = norm(vec_to_target);
                    angle_diff = wrapToPi(atan2(vec_to_target(2), vec_to_target(1)) - sensor_theta(i));
                    
                    if dist <= SR && abs(angle_diff) <= pi/6
                        Z_i = H * x_true + chol(R)' * randn(2,1); 
                        U_all(:,:,i) = H' * B * H;
                        u_all(:,i)   = H' * B * Z_i;
                    end
                end
                
                % inclusive neighborhood (S_i and y_i)
                S = zeros(4, 4, N);
                y = zeros(4, N);
                for i = 1:N
                    inclusive_neighborhood = [find(Adj(i, :)), i];
                    for j = inclusive_neighborhood
                        S(:, :, i) = S(:, :, i) + U_all(:,:,j);
                        y(:, i)   = y(:, i)   + u_all(:,j);
                    end
                end
                
                % ckf
                U_total = sum(U_all, 3);
                u_total = sum(u_all, 2);
                J_post_ckf = J_prior_ckf + U_total;
                x_post_ckf = J_post_ckf \ (J_prior_ckf * x_prior_ckf + u_total);
                
                err_ckf = norm(x_true(1:2) - x_post_ckf(1:2));
                total_error_ckf = total_error_ckf + (N * err_ckf);
                total_sq_error_ckf = total_sq_error_ckf + (N * err_ckf^2);
                
                J_prior_ckf = inv(Phi * inv(J_post_ckf) * Phi' + Q);
                x_prior_ckf = Phi * x_post_ckf;
                
                % kcf
                x_kcf_k = x_prior_kcf;
                J_post_kcf = zeros(4, 4, N);
                for i = 1:N
                    J_post_kcf(:,:,i) = J_prior_kcf(:,:,i) + S(:,:,i);
                end
                for k_iter = 1:K_test
                    x_new = x_kcf_k;
                    for i = 1:N
                        neighbors = find(Adj(i, :));
                        sum_x_diff = zeros(4, 1);
                        for j = neighbors
                            sum_x_diff = sum_x_diff + (x_kcf_k(:, j) - x_kcf_k(:, i));
                        end
                        J_inv = inv(J_prior_kcf(:,:,i));
                        consensus_term = (epsilon / (1 + norm(J_inv))) * J_inv * sum_x_diff;
                        
                        if k_iter == 1
                            meas_term = inv(J_post_kcf(:,:,i)) * (y(:,i) - S(:,:,i)*x_prior_kcf(:,i));
                        else
                            meas_term = zeros(4, 1); 
                        end
                        x_new(:, i) = x_kcf_k(:, i) + meas_term + consensus_term;
                    end
                    x_kcf_k = x_new;
                end
                x_post_kcf = x_kcf_k;
                
                % gkcf
                x_gkcf_k = x_prior_gkcf;
                J_post_gkcf = zeros(4, 4, N);
                for i = 1:N
                    neighbors = find(Adj(i, :));
                    sum_J_diff = zeros(4, 4);
                    for j = neighbors
                        sum_J_diff = sum_J_diff + (J_prior_gkcf(:,:,j) - J_prior_gkcf(:,:,i));
                    end
                    J_post_gkcf(:,:,i) = J_prior_gkcf(:,:,i) + epsilon * sum_J_diff + S(:,:,i);
                end
                for k_iter = 1:K_test
                    x_new = x_gkcf_k;
                    for i = 1:N
                        neighbors = find(Adj(i, :));
                        sum_Jx_diff = zeros(4, 1);
                        for j = neighbors
                            sum_Jx_diff = sum_Jx_diff + J_prior_gkcf(:,:,j) * (x_gkcf_k(:, j) - x_gkcf_k(:, i));
                        end
                        consensus_term = inv(J_post_gkcf(:,:,i)) * epsilon * sum_Jx_diff;
                        
                        if k_iter == 1
                            meas_term = inv(J_post_gkcf(:,:,i)) * (y(:,i) - S(:,:,i)*x_prior_gkcf(:,i));
                        else
                            meas_term = zeros(4, 1); 
                        end
                        x_new(:, i) = x_gkcf_k(:, i) + meas_term + consensus_term;
                    end
                    x_gkcf_k = x_new;
                end
                x_post_gkcf = x_gkcf_k;
                
                % icf
                V = zeros(4, 4, N);
                v = zeros(4, N);
                for i = 1:N
                    V(:, :, i) = (1/N) * J_prior_icf(:, :, i) + U_all(:,:,i);
                    v(:, i)    = (1/N) * J_prior_icf(:, :, i) * x_prior_icf(:, i) + u_all(:,i);
                end
                for k_iter = 1:K_test
                    V_new = V;
                    v_new = v;
                    for i = 1:N
                        neighbors = find(Adj(i, :));
                        sum_V_diff = zeros(4, 4);
                        sum_v_diff = zeros(4, 1);
                        for j = neighbors
                            sum_V_diff = sum_V_diff + (V(:, :, j) - V(:, :, i));
                            sum_v_diff = sum_v_diff + (v(:, j) - v(:, i));
                        end
                        V_new(:, :, i) = V(:, :, i) + epsilon * sum_V_diff;
                        v_new(:, i)   = v(:, i)   + epsilon * sum_v_diff;
                    end
                    V = V_new;
                    v = v_new;
                end
                
                for i = 1:N
                    J_post_icf = N * V(:, :, i);
                    x_post_icf = V(:, :, i) \ v(:, i); 
                    
                    % ICF
                    err_icf = norm(x_true(1:2) - x_post_icf(1:2));
                    total_error_icf = total_error_icf + err_icf;
                    total_sq_error_icf = total_sq_error_icf + err_icf^2;
                    
                    % KCF
                    err_kcf = norm(x_true(1:2) - x_post_kcf(1:2, i));
                    total_error_kcf = total_error_kcf + err_kcf;
                    total_sq_error_kcf = total_sq_error_kcf + err_kcf^2;
                    
                    % GKCF
                    err_gkcf = norm(x_true(1:2) - x_post_gkcf(1:2, i));
                    total_error_gkcf = total_error_gkcf + err_gkcf;
                    total_sq_error_gkcf = total_sq_error_gkcf + err_gkcf^2;
                    
                    num_data_points = num_data_points + 1;
                    % predict
                    J_prior_icf(:, :, i) = inv(Phi * inv(J_post_icf) * Phi' + Q);
                    x_prior_icf(:, i)    = Phi * x_post_icf;
                    
                    J_prior_kcf(:, :, i) = inv(Phi * inv(J_post_kcf(:,:,i)) * Phi' + Q);
                    x_prior_kcf(:, i)    = Phi * x_post_kcf(:, i);
                    
                    J_prior_gkcf(:, :, i) = inv(Phi * inv(J_post_gkcf(:,:,i)) * Phi' + Q);
                    x_prior_gkcf(:, i)    = Phi * x_post_gkcf(:, i);
                end
            end
        end
    end
    
    % means
    mean_error_icf_K(K_test) = total_error_icf / num_data_points;
    mean_error_ckf_K(K_test) = total_error_ckf / num_data_points;
    mean_error_kcf_K(K_test) = total_error_kcf / num_data_points;
    mean_error_gkcf_K(K_test) = total_error_gkcf / num_data_points;
    
    % std
    std_icf_K(K_test) = sqrt((total_sq_error_icf / num_data_points) - mean_error_icf_K(K_test)^2);
    std_ckf_K(K_test) = sqrt((total_sq_error_ckf / num_data_points) - mean_error_ckf_K(K_test)^2);
    std_kcf_K(K_test) = sqrt((total_sq_error_kcf / num_data_points) - mean_error_kcf_K(K_test)^2);
    std_gkcf_K(K_test) = sqrt((total_sq_error_gkcf / num_data_points) - mean_error_gkcf_K(K_test)^2);
end

%% Plotting
figure;
hold on;

% CKF
plot(1:K_max, mean_error_ckf_K, '-go', 'LineWidth', 1.5, 'MarkerSize', 6); 
plot(1:K_max, mean_error_ckf_K + 0.2*std_ckf_K, ':g', 'LineWidth', 1);
plot(1:K_max, mean_error_ckf_K - 0.2*std_ckf_K, ':g', 'LineWidth', 1);

% KCF
plot(1:K_max, mean_error_kcf_K, '-k*', 'LineWidth', 1.5, 'MarkerSize', 6);
plot(1:K_max, mean_error_kcf_K + 0.2*std_kcf_K, ':k', 'LineWidth', 1);
plot(1:K_max, mean_error_kcf_K - 0.2*std_kcf_K, ':k', 'LineWidth', 1);

% GKCF
plot(1:K_max, mean_error_gkcf_K, '-bs', 'LineWidth', 1.5, 'MarkerSize', 6);
plot(1:K_max, mean_error_gkcf_K + 0.2*std_gkcf_K, ':b', 'LineWidth', 1);
plot(1:K_max, mean_error_gkcf_K - 0.2*std_gkcf_K, ':b', 'LineWidth', 1);

% ICF
plot(1:K_max, mean_error_icf_K, '-rx', 'LineWidth', 1.5, 'MarkerSize', 6); 
plot(1:K_max, mean_error_icf_K + 0.2*std_icf_K, ':r', 'LineWidth', 1);
plot(1:K_max, mean_error_icf_K - 0.2*std_icf_K, ':r', 'LineWidth', 1);

xlabel('Number of Consensus Iterations, K');
ylabel('Mean Error, e and S.D. \pm 0.2\sigma', 'Interpreter', 'tex');
title('Experiment 1: Varying Consensus Iterations (Equal Priors)');
legend('CKF', '', '', 'KCF', '', '', 'GKCF', '', '', 'ICF', 'Location', 'northeast');
grid on;
xlim([1 K_max]);
hold off;