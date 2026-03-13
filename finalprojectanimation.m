% MAE 247: Algorithm Convergence Animation (Dynamic Tracking + Active Sensors)
% Outputs an MP4 video with a zoomed-in, tracking perspective and LOS arrows.
clear; clc; close all;

%% 1. Experiment Parameters
N = 15;                 % sensors
K = 20;                 % iterations per time step
Delta = 2;              % degree
SR = 300;               % range
t_max = 30;             % times steps for animation
epsilon = 0.65 / Delta; % consensus speed param
gridsize = 500;         % environment size
zoom_window = 100;      % frame size

% sys mats
H = [1 0 0 0; 0 1 0 0];
Phi = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];
Q = diag([10, 10, 1, 1]); 
R = 100 * eye(2);
B = inv(R);

%% 2. Environment & Initializations Setup
rng('shuffle'); % for env

sensor_pos = gridsize * rand(2, N);
sensor_theta = 2 * pi * rand(1, N);

Adj = zeros(N, N);
for d = 1:(Delta/2)
    Adj = Adj + diag(ones(N-d, 1), d) + diag(ones(N-d, 1), -d);
    Adj = Adj + diag(ones(d, 1), N-d) + diag(ones(d, 1), -(N-d));
end

% target trajectory initial state (center start)
x_true = [gridsize/2; gridsize/2; 2*cos(rand*2*pi); 2*sin(rand*2*pi)]; 

% init uncorrelated priors
P_prior_init = diag([200, 200, 10, 10]); 
J_prior_init = inv(P_prior_init);

x_prior_icf = zeros(4, N);
J_prior_icf = repmat(J_prior_init, 1, 1, N);
for i = 1:N
    x_prior_icf(:, i) = x_true + chol(P_prior_init)' * randn(4,1); 
end

x_prior_ckf = x_prior_icf(:, 1); J_prior_ckf = J_prior_init;
x_prior_kcf = x_prior_icf;       J_prior_kcf = J_prior_icf;
x_prior_gkcf = x_prior_icf;      J_prior_gkcf = J_prior_icf;

%% 3. Setup Animation Figure & Video Writer
fig = figure('Name', 'Convergence Animation', 'Position', [100, 100, 800, 700]);
hold on; axis equal;
box on; grid on;
xlabel('X Position'); ylabel('Y Position');

fov_angle = pi/6;
edge_len = SR / cos(fov_angle);
for i = 1:N
    p1 = sensor_pos(:, i);
    p2 = p1 + edge_len * [cos(sensor_theta(i)-fov_angle); sin(sensor_theta(i)-fov_angle)];
    p3 = p1 + edge_len * [cos(sensor_theta(i)+fov_angle); sin(sensor_theta(i)+fov_angle)];
    patch([p1(1) p2(1) p3(1)], [p1(2) p2(2) p3(2)], 'b', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
end

for i = 1:N
    neighbors = find(Adj(i, :));
    for j = neighbors
        if j > i 
            plot([sensor_pos(1, i), sensor_pos(1, j)], [sensor_pos(2, i), sensor_pos(2, j)], 'g--', 'LineWidth', 0.5);
        end
    end
end
quiver(sensor_pos(1,:), sensor_pos(2,:), 20*cos(sensor_theta), 20*sin(sensor_theta), 0, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
scatter(sensor_pos(1,:), sensor_pos(2,:), 30, 'ro', 'filled', 'MarkerEdgeColor', 'k');

h_trail = plot(NaN, NaN, 'k:', 'LineWidth', 1);
h_kcf   = scatter(NaN, NaN, 25, 'magenta', 'd', 'LineWidth', 1);
h_gkcf  = scatter(NaN, NaN, 25, 'b', 's', 'LineWidth', 1);
h_icf   = scatter(NaN, NaN, 40, 'r', 'x', 'LineWidth', 1.5);

h_ckf   = plot(NaN, NaN, 'go', 'MarkerSize', 5, 'LineWidth', 1.5, 'MarkerFaceColor', 'g');
h_true  = plot(NaN, NaN, 'k*', 'MarkerSize', 10, 'LineWidth', 1.5);

legend([h_true, h_ckf, h_kcf, h_gkcf, h_icf], ...
    'True Target', 'CKF (Optimal)', 'KCF Nodes', 'GKCF Nodes', 'ICF Nodes', ...
    'Location', 'northeastoutside');

trail_x = []; trail_y = [];
h_los = []; h_arrows = []; % Arrays to hold dynamic LOS lines and boundary arrows

video_filename = sprintf('Convergence_Tracking_K%d.mp4', K);
vidObj = VideoWriter(video_filename, 'MPEG-4'); 
vidObj.FrameRate = 12; 
open(vidObj);
fprintf('Recording video to: %s\n', video_filename);

%% 4. Main Animation Loop (Time Steps)
for t = 1:t_max
    x_true = Phi * x_true + chol(Q)' * randn(4,1);
    trail_x = [trail_x, x_true(1)];
    trail_y = [trail_y, x_true(2)];
    
    set(h_trail, 'XData', trail_x, 'YData', trail_y);
    set(h_true, 'XData', x_true(1), 'YData', x_true(2));
    
    % camera
    xlim([x_true(1) - zoom_window/2, x_true(1) + zoom_window/2]);
    ylim([x_true(2) - zoom_window/2, x_true(2) + zoom_window/2]);
    
    % line of sight
    U_all = zeros(4, 4, N);
    u_all = zeros(4, N);
    active_cams = []; % track which cameras can see the target this turn
    
    for i = 1:N
        vec_to_target = x_true(1:2) - sensor_pos(:, i);
        dist = norm(vec_to_target);
        angle_diff = wrapToPi(atan2(vec_to_target(2), vec_to_target(1)) - sensor_theta(i));
        
        if dist <= SR && abs(angle_diff) <= pi/6
            Z_i = H * x_true + chol(R)' * randn(2,1); 
            U_all(:,:,i) = H' * B * H;
            u_all(:,i)   = H' * B * Z_i;
            active_cams = [active_cams, i]; % record active camera
        end
    end
    
    % active cams
    if ~isempty(h_los); delete(h_los); h_los = []; end
    if ~isempty(h_arrows); delete(h_arrows); h_arrows = []; end
    
    W = zoom_window / 2; % dist from edge to screen
    for i = active_cams
        xc = sensor_pos(1, i); yc = sensor_pos(2, i);
        xt = x_true(1); yt = x_true(2);
        
        % line of sight
        h_los(end+1) = plot([xt, xc], [yt, yc], 'Color', [1, 0.5, 0, 0.4], 'LineStyle', '-', 'LineWidth', 1.5,'DisplayName',sprintf("LoS to Cam %d",i));
    end
    
    % inclusive sums (KCF/GKCF)
    S = zeros(4, 4, N); y = zeros(4, N);
    for i = 1:N
        inc_neigh = [find(Adj(i, :)), i];
        for j = inc_neigh
            S(:,:,i) = S(:,:,i) + U_all(:,:,j);
            y(:,i)   = y(:,i)   + u_all(:,j);
        end
    end
    
    % CKF (Instantaneous Update)
    U_total = sum(U_all, 3); u_total = sum(u_all, 2);
    J_post_ckf = J_prior_ckf + U_total;
    x_post_ckf = J_post_ckf \ (J_prior_ckf * x_prior_ckf + u_total);
    set(h_ckf, 'XData', x_post_ckf(1), 'YData', x_post_ckf(2));
    
    % Initialize Distributed Filters for Consensus
    x_kcf_k = x_prior_kcf;
    J_post_kcf = zeros(4, 4, N);
    for i = 1:N; J_post_kcf(:,:,i) = J_prior_kcf(:,:,i) + S(:,:,i); end
    
    x_gkcf_k = x_prior_gkcf;
    J_post_gkcf = zeros(4, 4, N);
    for i = 1:N
        sum_J_diff = sum(J_prior_gkcf(:,:,find(Adj(i,:))), 3) - sum(Adj(i,:))*J_prior_gkcf(:,:,i);
        J_post_gkcf(:,:,i) = J_prior_gkcf(:,:,i) + epsilon * sum_J_diff + S(:,:,i);
    end
    
    V = zeros(4, 4, N); v = zeros(4, N);
    for i = 1:N
        V(:,:,i) = (1/N) * J_prior_icf(:,:,i) + U_all(:,:,i);
        v(:,i)   = (1/N) * J_prior_icf(:,:,i) * x_prior_icf(:,i) + u_all(:,i);
    end
% Convergence loop
    for k_iter = 1:K
        
        title(sprintf('Time Step: %d | Consensus Iteration: %d / %d', t, k_iter, K), 'FontSize', 12);
        
        x_new_kcf = x_kcf_k;
        for i = 1:N
            sum_x_diff = sum(x_kcf_k(:, find(Adj(i,:))) - x_kcf_k(:, i), 2);
            J_inv = inv(J_prior_kcf(:,:,i));
            consensus_term = (epsilon / (1 + norm(J_inv))) * J_inv * sum_x_diff;
            meas_term = (k_iter==1) * (inv(J_post_kcf(:,:,i)) * (y(:,i) - S(:,:,i)*x_prior_kcf(:,i)));
            x_new_kcf(:, i) = x_kcf_k(:, i) + meas_term + consensus_term;
        end
        x_kcf_k = x_new_kcf;
        
        
        x_new_gkcf = x_gkcf_k;
        for i = 1:N
            neighbors = find(Adj(i, :));
            sum_Jx_diff = zeros(4, 1);
            for j = neighbors
                sum_Jx_diff = sum_Jx_diff + J_prior_gkcf(:,:,j) * (x_gkcf_k(:, j) - x_gkcf_k(:, i));
            end
            consensus_term = inv(J_post_gkcf(:,:,i)) * epsilon * sum_Jx_diff;
            meas_term = (k_iter==1) * (inv(J_post_gkcf(:,:,i)) * (y(:,i) - S(:,:,i)*x_prior_gkcf(:,i)));
            x_new_gkcf(:, i) = x_gkcf_k(:, i) + meas_term + consensus_term;
        end
        x_gkcf_k = x_new_gkcf;
       
        V_new = V; v_new = v;
        x_icf_intermediate = zeros(4, N);
        for i = 1:N
            neighbors = find(Adj(i, :));
            sum_V_diff = sum(V(:,:,neighbors), 3) - sum(Adj(i,:))*V(:,:,i);
            sum_v_diff = sum(v(:,neighbors), 2) - sum(Adj(i,:))*v(:,i);
            V_new(:,:,i) = V(:,:,i) + epsilon * sum_V_diff;
            v_new(:,i)   = v(:,i)   + epsilon * sum_v_diff;
            x_icf_intermediate(:, i) = V_new(:,:,i) \ v_new(:,i);
        end
        V = V_new; v = v_new;

        set(h_kcf, 'XData', x_kcf_k(1,:), 'YData', x_kcf_k(2,:));
        set(h_gkcf, 'XData', x_gkcf_k(1,:), 'YData', x_gkcf_k(2,:));
        set(h_icf, 'XData', x_icf_intermediate(1,:), 'YData', x_icf_intermediate(2,:));
        
        drawnow;
        

        frame = getframe(fig);
        writeVideo(vidObj, frame);
        
    end
    
    % predict
    x_post_icf = zeros(4, N);
    for i = 1:N
        J_post_icf = N * V(:, :, i);
        x_post_icf(:, i) = V(:, :, i) \ v(:, i); 
        J_prior_icf(:, :, i) = inv(Phi * inv(J_post_icf) * Phi' + Q);
        x_prior_icf(:, i)    = Phi * x_post_icf(:, i);
        
        J_prior_kcf(:, :, i) = inv(Phi * inv(J_post_kcf(:,:,i)) * Phi' + Q);
        x_prior_kcf(:, i)    = Phi * x_kcf_k(:, i);
        
        J_prior_gkcf(:, :, i) = inv(Phi * inv(J_post_gkcf(:,:,i)) * Phi' + Q);
        x_prior_gkcf(:, i)    = Phi * x_gkcf_k(:, i);
    end
    J_prior_ckf = inv(Phi * inv(J_post_ckf) * Phi' + Q);
    x_prior_ckf = Phi * x_post_ckf;
end

close(vidObj);
fprintf('Animation Complete! File saved as: %s\n', video_filename);