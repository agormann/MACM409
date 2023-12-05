function [U, B, V] = lrf(A, k, p)
% Computes the low rank factorization of the matrix A.
% Input:
%   A = real matrix of dimension m x n.
%   k = desired rank.
%   p = either 0, 1, or k;
%       0 -> economy QR factorization;
%       1 -> full QR factorization;
%       k -> truncated QR factorization.
% Output:
%   U = m x k matrix.
%   B = k x k submatrix of A.
%   V = k x n matrix.

[m, n] = size(A);

% Step 1
[Q, R, P1] = pgs(A, p);

Q11 = Q(1:k, 1:k);
Q21 = Q(k+1:m, 1:k);
R11 = R(1:k, 1:k);
R12 = R(1:k, k+1:n);

% Step 2
T = decomposition(R11) \ R12;
ACS = [(Q11*R11); (Q21*R11)];

% Step 3
[q, r, P2] = pgs(ACS', p);
r11 = r(:, 1:k);
r12 = r(:, k+1:m);
AS = (q*r11)';

% Step 4
S = (r12') / decomposition(r11');

% Output
U = P2*[eye(k); S];
B = AS;
V = P1*[eye(k); T'];

end

