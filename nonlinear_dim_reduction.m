% =========================================================================
% Title:       Nonlinear Dimensionality Reduction: MDS, Kernel PCA & Isomap
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Comparison of dimensionality reduction techniques on nonlinear
%              datasets (Concentric Circles and Swiss Roll):
%                1. Classical Multidimensional Scaling (MDS) via double centering & SVD
%                2. Kernel PCA with Gaussian (RBF) Kernel
%                3. Isomap algorithm
% =========================================================================

%% ========================================================================
%  PART 1: TWO CONCENTRIC CIRCLES DATASET
%  ========================================================================

clear; clc; close all;

% --- Data Generation & Parameters ---
M = 600;                        % Number of points
t1 = 2*pi*rand(1,M);            % Generate random angles

% Radii definitions
inR1 = 5;                       % Inner radius of the first concentric circle
outR1 = 9;                      % Outer radius of the first concentric circle
inR2 = 20;                      % Inner radius of the second concentric circle
outR2 = 25;                     % Outer radius of the second concentric circle

noise = 2;                      % Magnitude of random noise (Center in [0,0])

% Generate random radii within the concentric circles
r1 = sqrt((outR1^2 - inR1^2)*rand(1,M) + inR1^2);
r2 = sqrt((outR2^2 - inR2^2)*rand(1,M) + inR2^2);

% Add random noise to the radii
r1 = r1 + noise*randn(1,M);
r2 = r2 + noise*randn(1,M);

% Convert polar coordinates to Cartesian coordinates
x1 = r1.*cos(t1);
y1 = r1.*sin(t1);
x2 = r2.*cos(t1);
y2 = r2.*sin(t1);

% --- Visualization: Concentric Circles ---
figure(1)
scatter(x1, y1, 10, 'b', 'filled');
hold on;
scatter(x2, y2, 10, 'r', 'filled');
hold off;
axis equal;
xlabel('X');
ylabel('Y');
title('Two Concentric Circles with Random Noise');
legend('Inner Circle', 'Outer Circle');

% --- Method 1: Multidimensional Scaling (MDS) ---
X = [x1' y1'; x2' y2'];
distances = pdist(X);
k = 2;                          % Reduced dimension
D = squareform(distances);
[n, ~] = size(D);               % Number of data points

% Double centering operation
I = eye(n);
O = ones(n);
H = I - ((1/n)*O);
B = -0.5 * H * D * H';

[U, L, W] = svd(B);
v = zeros(n, k);
s = zeros(k, k);
for j = 1:k
    v(:,j) = W(:,j);
    for i = 1:k
        s(i,j) = sqrt(L(i,j));
    end
end

z1 = s * v';                    % Projected data in the k dimension
z1 = z1'; 

% Error calculation with MDS
errMDS = norm((X'*X) - (z1'*z1));
errrelMDS = errMDS / norm(X'*X);
disp('MDS error='); disp(errrelMDS);

% MDS Plots
figure(2)
hold on;
scatter(z1(M+1:end, 1), 0, 10, 'r', 'filled');
scatter(z1(1:M, 1), 0, 10, 'b', 'filled');
hold off;
xlabel('MD1');
title('1D Multidimensional Scaling');
legend('Inner Circle', 'Outer Circle');

figure(3)
scatter(z1(1:M, 1), z1(1:M, 2), 10, 'b', 'filled');
hold on;
scatter(z1(M+1:end, 1), z1(M+1:end, 2), 10, 'r', 'filled');
hold off;
xlabel('MD1');
ylabel('MD2');
title('2D Multidimensional Scaling');
legend('Inner Circle', 'Outer Circle');

% --- Method 2: Kernel PCA with Gaussian Kernel ---
sigma = 26.8;
K = zeros(n, n);
K1 = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i,j) = exp(-(norm(X(i,:) - X(j,:))^2) / (2*sigma^2));
    end
end

% Construct centered Gram matrix K tilde
In = (1/n) * ones(n, n);
Kt = K - In*K - K*In + In*K*In; 
k = 2;                          % Reduced dimension

% Eigenvalue decomposition (V: eigenvectors, L: eigenvalues)
[V, L] = eig(Kt); 
v = zeros(n, k);                % Initialization
for j = 1:k
    v(:,j) = V(:,j);
end

y = v' * K;                     % Projected data in the k dimension
l = diag(L); 
l = l(1:k);
y = y ./ ((n-1)*l);
y = y';

% Error calculation with Kernel PCA
errkPCA = norm(Kt - v*v'*Kt); 
errrelkPCA = errkPCA / norm(Kt); 
disp('PCA error='); disp(errrelkPCA);

% Kernel PCA Plots
figure(4)
scatter(y(1:M, 1), 0, 10, 'b', 'filled');
hold on;
scatter(y(M+1:end, 1), 0, 10, 'r', 'filled');
hold off;
xlabel('PC1');
title('First principal component after Kernel PCA');
legend('Inner Circle', 'Outer Circle');

figure(5)
scatter(y(1:M, 1), y(1:M, 2), 10, 'b', 'filled');
hold on;
scatter(y(M+1:end, 1), y(M+1:end, 2), 10, 'r', 'filled');
hold off;
xlabel('PC1');
ylabel('PC2');
title('First two principal components after Kernel PCA');
legend('Inner Circle', 'Outer Circle');

% --- Method 3: Isomap ---
[Y, R, E] = IsomapC(D, 'k', 8);


%% ========================================================================
%  PART 2: SWISS ROLL DATASET
%  ========================================================================

clear; clc; close all;

% --- Data Generation: Swiss Roll ---
N = 2048;                       % Number of points
t = rand(1, N);
t = sort(4*pi*sqrt(t))';
z = 8*pi*rand(N, 1);
x = (t + .1).*cos(t);
y = (t + .1).*sin(t);
data = [x, y, z];

% --- Visualization: Swiss Roll ---
figure(1)
cmap = jet(N);                  % Data visualization 
scatter3(x, y, z, 20, cmap);
xlabel('x');
ylabel('y');
zlabel('z');
title('3D Swiss Roll');

% --- Method 1: Multidimensional Scaling (MDS) ---
distances = pdist(data);
k = 3;                          % Reduced dimension
D = squareform(distances);

% Double centering operation
I = eye(N);
O = ones(N);
H = I - ((1/N)*O);
B = -0.5 * H * D * H';

[U, L, W] = svd(B);
v = zeros(N, k);
s = zeros(k, k);
for j = 1:k
    v(:,j) = W(:,j);
    for i = 1:k
        s(i,j) = sqrt(L(i,j));
    end
end

z1 = s * v';                    % Projected data in the k dimension
z1 = z1';

% Error calculation with MDS
errMDS = norm((data'*data) - (z1'*z1));
errrelMDS = errMDS / norm(data'*data);
disp('MDS error='); disp(errrelMDS);

% MDS Plots
figure(2)
scatter(z1(:, 2), 0, 20, cmap);
xlabel('MD1');
title('1D Multidimensional Scaling');

figure(3)
scatter(z1(:, 2), z1(:, 3), 20, cmap);
xlabel('MD1');
ylabel('MD2');
title('2D Multidimensional Scaling');

% --- Method 2: Kernel PCA with Gaussian Kernel ---
K = zeros(N);                   % Initialization
sigma = 15.8;
for i = 1:N
    for j = 1:N
        K(i,j) = exp(-(norm(data(i,:) - data(j,:))^2)) / (2*(sigma)^2);
    end
end

In = (1/N) * ones(N);
Kt = K - In*K - K*In + In*K*In; % K tilde centered matrix

% Eigenvalue decomposition (V: eigenvectors, L: eigenvalues)
[V, L] = eig(Kt); 

% Normalization of V
norm_eigVector = sqrt(sum(V.^2));
V = V ./ repmat(norm_eigVector, size(V, 1), 1);

% Dimensionality reduction
V = V(:, 1:k);
y = V' * K;                     % Projected data in the k dimension
Diag = diag(L);                 % Eigenvalue extraction
[l, ~] = sort(Diag, 'descend'); % Eigenvalues in descending order
l = l(1:k);
y = y ./ ((N-1).*l);
% y = y';

% Error calculation with Kernel PCA
errkPCA = norm(Kt - V*V'*Kt); 
errrelkPCA = errkPCA / norm(Kt); 
disp('PCA error='); disp(errrelkPCA);

% Kernel PCA Plots
figure(4)
hold on;
scatter(y(2, 1:N), 0, 20, cmap);
hold off;
xlabel('PC1');
title('First principal component after Kernel PCA');

figure(5)
hold on;
scatter(y(2, 1:N), y(3, 1:N), 20, cmap);
hold off;
xlabel('PC1');
ylabel('PC2');
title('First two principal component after Kernel PCA');

% --- Method 3: Isomap ---
[Y, R, E] = IsomapSwissRoll(D, 'k', 8);
