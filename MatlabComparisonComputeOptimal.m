N = 50;
x=sdpvar(2*ones(1,N+1),ones(1,N+1));

u=sdpvar(ones(1,N),ones(1,N));
time = 10;
Q = eye(2);
R = 1;
A = [1 1;
     0 1];
B = [0;
     1];
x0 = [1;
      1];
    
QF=Q;

tic()
Constraints = [x{1} ==x0(:,1)];
for i = 1:N
    Constraints=[Constraints;
             x{i+1} ==    A*x{i} + B*u{i};
             -4 <= x{i}(1) <= 4
             -4 <= x{i}(1) <= 4
             -1 <= u{i}(1) <= 1];
end    

% State Cost
Cost = 0;

for i=1:N        
    Cost = Cost + x{i}'*Q*x{i} + u{i}'*R*u{i};
end 

% Terminal Cost
Cost = Cost + x{N+1}'*QF*x{N+1};

Problem = solvesdp(Constraints,Cost);

SolverTime = toc()

for i = 1:N
    x_sol(:,i) = double(x{i})
end
%%
figure
plot(x_sol(1,:), x_sol(2,:), 'or')
