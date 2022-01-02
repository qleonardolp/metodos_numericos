%% FEM matrices symbolic manipulation:

% a = x_{e}
% b = x_{e+1}
% h = b - a
% x = integrando...
clear
clc
syms a b h x
assume([a b h x], 'real')

% h = b - a;
b = h + a;

Ns = [(b - x)/h; 0; 0; (x - a)/h; 0; 0];

Nv = [0; (x - b)^2*(2*x - 3*a + b)/(h^3); (x - a)*(x - b)^2/(h^2); ...
      0; -(x - a)^2*(2*x + a - 3*b)/(h^3); (x - a)^2*(x - b)/(h^2)];
Nw = [(x - b)^2*(2*x - 3*a + b)/(h^3); (x - a)*(x - b)^2/(h^2); ...
     -(x - a)^2*(2*x + a - 3*b)/(h^3); (x - a)^2*(x - b)/(h^2)];

%% Symbolic integration: element matrices/ vectors
% symmetry check using: subs(m_e, [a b], [11.4 12]),
% issymmetric(double(ans)) ...

% --------------------------- %
% Chordwise Motion Equations: %
% Element Mass Mtx:
m_e = int(Ns*Ns' + Nv*Nv', x, a, b); % lembre de *rho*A (is symmetric)
% Element Gyroscopic Mtx:
g_e = int(Nv*Ns' - Ns*Nv', x, a, b); % lembre de *rho*A (is skew-symmetric)

% Element Stiffness Mtx:
dNs = diff(Ns, x)*diff(Ns', x);
d2Nv = diff(Nv,x,x)*diff(Nv',x,x);

k_e_s = int(dNs, x, a, b); % lembre de *E*A (is symmetric)
k_e_v = int(d2Nv, x, a, b); % lembre de *E*Iz (is symmetric)
% k_e = E*A*k_e_s + E*Iz*k_e_v ...              (is symmetric)

% Element "Motion-induced Stiffness" Mtx
s_e_0ord = int(diff(Nv,x)*diff(Nv',x), x, a, b);    % (is symmetric)
s_e_1ord = int(x*diff(Nv,x)*diff(Nv',x), x, a, b);  % (is symmetric)
s_e_2ord = int(x^2*diff(Nv,x)*diff(Nv',x), x, a, b);% (is symmetric)

% Element Load Vector:
f_e_0ord_s = int(Ns, x, a, b);
f_e_1ord_s = int(x*Ns, x, a, b);
f_e_0ord_v = int(Nv, x, a, b);
f_e_1ord_v = int(x*Nv, x, a, b);

% -------------------------- %
% Flapwise Motion Equations: %
% Element Mass Mtx:
mw_e = int(Nw*Nw', x, a, b); % lembre de *rho*A     % (is symmetric)

% Element Stiffness Mtx:
d2Nw = diff(Nw,x,x)*diff(Nw',x,x);
kw_e = int(d2Nw, x, a, b); % lembre de *E*Iy        % (is symmetric)

% Element "Motion-induced Stiffness" Mtx
sw_e_0ord = int(diff(Nw,x)*diff(Nw',x), x, a, b);     % (is symmetric)
sw_e_1ord = int(x*diff(Nw,x)*diff(Nw',x), x, a, b);   % (is symmetric)
sw_e_2ord = int(x^2*diff(Nw,x)*diff(Nw',x), x, a, b); % (is symmetric)

