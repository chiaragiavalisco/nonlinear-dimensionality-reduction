% =========================================================================
% Title:       Nonlinear Dimensionality Reduction: Concentric Circles
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Evaluates MDS, Kernel PCA (RBF kernel), and Isomap on a 
%              2D synthetic dataset of two concentric rings with noise.
% =========================================================================

clear; clc; close all;

% --- Data Generation & Parameters ---
M = 600;                        % Number of points per ring
t1 = 2*pi*rand(1, M);           % Random angles

% Radii definitions
inR1 = 5;                       % Inner radius of circle 1
outR1 = 9;                      % Outer radius of circle 1
inR2 = 20;                      % Inner radius of circle 2
outR2 = 25;                     % Outer radius of circle 2

noise = 2;                      % Magnitude of Gaussian noise

% Radii generation with noise
r1 = sqrt((outR1^2 - inR1^2)*rand(1, M) + inR1^2) + noise*randn(1, M);
r2 = sqrt((outR2^2 - inR2^2)*rand(1, M) + inR2^2) + noise*randn(1, M);

% Polar to Cartesian conversion
x1 = r1 .* cos(t1);
y1 = r1 .* sin(t1);
x2 = r2 .* cos(t1);
y2 = r2 .* sin(t1);

% Plot Ground Truth
figure(1);
scatter(x1, y1, 10, 'b', 'filled');
hold on;
scatter(x2, y2, 10, 'r', 'filled');
hold off;
axis equal;
xlabel('X'); ylabel('Y');
title('Two Concentric Circles with Random Noise');
legend('Inner Circle', 'Outer Circle');

% --- Method 1: Multidimensional Scaling (MDS) ---
X = [x1' y1'; x2' y2'];
distances = pdist(X);
k = 2;                          % Target reduced dimension
D = squareform(distances);
[n, ~] = size(D);

% Double centering
I = eye(n);
O = ones(n);
H = I - ((1/n)*O);
B = -0.5 * H * D * H';

[~, L, W] = svd(B);
v = zeros(n, k);
s = zeros(k, k);
for j = 1:k
    v(:, j) = W(:, j);
    for i = 1:k
        s(i, j) = sqrt(L(i, j));
    end
end

z1 = (s * v')';                 % Projected coordinates

% Reconstruction error
errMDS = norm((X'*X) - (z1'*z1));
errrelMDS = errMDS / norm(X'*X);
disp(['MDS relative error: ', num2str(errrelMDS)]);

figure(2);
scatter(z1(M+1:end, 1), 0, 10, 'r', 'filled'); hold on;
scatter(z1(1:M, 1), 0, 10, 'b', 'filled'); hold off;
xlabel('MD1');
title('1D Multidimensional Scaling');
legend('Inner Circle', 'Outer Circle');

figure(3);
scatter(z1(1:M, 1), z1(1:M, 2), 10, 'b', 'filled'); hold on;
scatter(z1(M+1:end, 1), z1(M+1:end, 2), 10, 'r', 'filled'); hold off;
xlabel('MD1'); ylabel('MD2');
title('2D Multidimensional Scaling');
legend('Inner Circle', 'Outer Circle');

% --- Method 2: Kernel PCA (Gaussian / RBF Kernel) ---
sigma = 26.8;
K = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i, j) = exp(-(norm(X(i, :) - X(j, :))^2) / (2 * sigma^2));
    end
end

% Gram matrix centering
In = (1/n) * ones(n, n);
Kt = K - In*K - K*In + In*K*In;

[V, L_eig] = eig(Kt);
v_kpca = V(:, 1:k);

y = v_kpca' * K;
l = diag(L_eig);
l = l(1:k);
y = (y ./ ((n - 1) * l))';

% Error calculation
errkPCA = norm(Kt - v_kpca * v_kpca' * Kt);
errrelkPCA = errkPCA / norm(Kt);
disp(['KPCA relative error: ', num2str(errrelkPCA)]);

figure(4);
scatter(y(1:M, 1), 0, 10, 'b', 'filled'); hold on;
scatter(y(M+1:end, 1), 0, 10, 'r', 'filled'); hold off;
xlabel('PC1');
title('First Principal Component after Kernel PCA');
legend('Inner Circle', 'Outer Circle');

figure(5);
scatter(y(1:M, 1), y(1:M, 2), 10, 'b', 'filled'); hold on;
scatter(y(M+1:end, 1), y(M+1:end, 2), 10, 'r', 'filled'); hold off;
xlabel('PC1'); ylabel('PC2');
title('First Two Principal Components after Kernel PCA');
legend('Inner Circle', 'Outer Circle');

% --- Method 3: Isomap ---
[Y, R, E] = IsomapC(D, 'k', 8);
