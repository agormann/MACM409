function [Q, R, P] = pgs(A, p)
% Computes the QR-factorization of a matrix A, such that AP = QR.
% Input:
%   A = real matrix of dimension m x n.
%   p = either 0, 1, or k;
%       0 -> economy QR factorization;
%       1 -> full QR factorization;
%       k -> truncated QR factorization.
% Output:
%   Q = matrix with orthonormal columns, whose dimensions are dictated by p.
%   R = upper triangular matrix, whose dimensions are dictated by p.
%   P = unitary pivot matrix.

[m, n] = size(A);
if p == 0
    r = min(m, n);
    Q = zeros(m, r);
    R = zeros(r, n);
    P = eye(n);
elseif p == 1
    r = min(m, n);
    Q = zeros(m, m);
    R = zeros(m, n);
    P = eye(n);
else
    r = p;
    Q = zeros(m, r);
    R = zeros(r, n);
    P = eye(n);
end

for k = 1 : r

% Step 1 (pivoting)
% Compute maximal column norm from columns k+1:n.
M = norm(A(:, k))^2;
jstar = k;
for j = k+1 : n
    testM = norm(A(:, j))^2;
    if testM > M
        M = testM;
        jstar = j;
    end
end
% If column k is not maximal, then interchange column k with column jstar
% of A, and the relevant subcolumns (a portion of the column) of R.
if jstar ~= k
    A(:, [jstar, k]) = A(:, [k, jstar]);
    P(:, [jstar, k]) = P(:, [k, jstar]);
    for i = 1 : k-1
        R(i, [jstar, k]) = R(i, [k, jstar]);
    end
end

% Step 2 (iterative reorthogonalization)
% Iteratively reorthogonalize until a preset tolerance.
if k ~= 1
    U = A(:, k);
    for i = 1 : k - 1
        a = Q(:, i)' * U;
        R(i, k) = R(i, k) + a;
        Unew = U - a*Q(:, i);
        if 2*norm(Unew)^2 >= norm(U)^2
            break;
        else
            U = Unew;
        end
    end
end

% Step 3 (normalization)
R(k, k) = norm(A(:, k));
Q(:, k) = A(:, k) / R(k, k);

% Step 4 (orthogonalization)
if k ~= n
    for j = k+1 : n
        R(k, j) = Q(:, k)' * A(:, j);
        A(:, j) = A(:, j) - R(k, j)*Q(:, k);
    end
end

end

end

