N = 5;
x=sdpvar(2*ones(1,N+1),ones(1,N+1));

u=sdpvar(ones(1,N),ones(1,N));
time = 10;
Q = eye(2);
R = 0.5;
A = [1 1;
     0 1];
B = [0;
     1];
x_cl = [-3.95;
        -0.05];
    
[QF, ~, ~] = dare(A,B,Q,R);
for t = 1:time;
    tic()
    Constraints = [x{1} ==x_cl(:,t)];
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

    x_cl(:,t+1) = A*x_cl(:,t) + B*double(u{1});
    u_cl(:,t) = double(u{1});
    SolverTime(t) = toc()

end
%%
figure
plot(x_cl(1,:), x_cl(2,:), 'or')
axis([-4, 4, -4, 4])