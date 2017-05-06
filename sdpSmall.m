function [G, Constraint, G_v, G_0, A, R_0, R_v, R, X_p, X_m] = sdpSmall(epsilon, mu_p, mu_m, p_p, p_m, r_p, r_m, s_p, s_m, opts, M_p, M_m, y_p, y_m, num_x)
    nc_p = size(M_p,1);
    nc_m = size(M_m,1);
    %assert(size(M_m,1) == nc_p);
    sizeG = 4 + nc_p + nc_m;
    Gx = sdpvar(2);
    Gs = sdpvar(2, sizeG-2, 'full');
    Mo = [mu_p'; mu_m'; M_p; M_m];
    Go = Mo * Mo';
    G = [Gx Gs; Gs' Go]; %sdpvar(sizeG);
    Gc = [Gx Gs(1:2,1:2); Gs(1:2,1:2)' Go(1:2,1:2)];
    %Gc = G(1:4,1:4);
    u_p = 3;
    u_m = 4;
    c_p = @(i) i+4;
    c_m = @(i) i+nc_p+4;
    
    % inner product of known terms
    Constraint = [G >= 0];
%     Constraint = [Constraint;
%                   G(u_p,u_p) == mu_p' * mu_p;
%                   G(u_p,u_m) == mu_p' * mu_m;
%                   G(u_m,u_p) == mu_m' * mu_p;
%                   G(u_m,u_m) == mu_m' * mu_m];
%     for i=1:nc_p
%         Constraint = [Constraint;
%                       G(u_p, c_p(i)) == M_p(i,:) * mu_p;
%                       G(u_m, c_p(i)) == M_p(i,:) * mu_m];
%         for j=1:nc_p
%             Constraint = [Constraint;
%                           G(c_p(j), c_p(i)) == M_p(j,:) * M_p(i,:)'];
%         end
%         for j=1:nc_m
%             Constraint = [Constraint;
%                           G(c_m(j), c_p(i)) == M_m(j,:) * M_p(i,:)'];
%         end
%     end
%     for i=1:nc_m
%         Constraint = [Constraint;
%                       G(u_p, c_m(i)) == M_m(i,:) * mu_p;
%                       G(u_m, c_m(i)) == M_m(i,:) * mu_m];
%         for j=1:nc_p
%             Constraint = [Constraint;
%                           G(c_p(j), c_m(i)) == M_p(j,:) * M_m(i,:)'];
%         end
%         for j=1:nc_m
%             Constraint = [Constraint;
%                           G(c_m(j), c_m(i)) == M_m(j,:) * M_m(i,:)'];
%         end
%     end

    
    % M_p * x_p <= y_p
    % M_m * x_m <= y_m
    for i=1:nc_p
        Constraint = [Constraint;
                      G(1, c_p(i)) <= y_p(i)];
        %Constraint = [Constraint;
        %              G(2, c_p(i)) <= y_p(i)];
    end
    for i=1:nc_m
        Constraint = [Constraint;
                      G(2, c_m(i)) <= y_m(i)];
        %Constraint = [Constraint;
        %              G(1, c_m(i)) <= y_m(i)];
    end
    
    % constrain \ell_2 norm to class center
    Constraint = [Constraint;
                  [1 0 -1 0] * Gc * [1; 0; -1; 0] <= r_p^2;
                  [0 1 0 -1] * Gc * [0; 1; 0; -1] <= r_m^2];
              
    % constrain attack to be orthogonal to mean
    mean_vec = [epsilon/(1+epsilon); -epsilon/(1+epsilon); 1/(1+epsilon); -1/(1+epsilon)];
    Constraint = [Constraint;
                  [1 0 -1 0] * Gc * mean_vec <= s_p;
                  [1 0 -1 0] * Gc * mean_vec >= -s_p;
                  [0 1 0 -1] * Gc * mean_vec <= s_m; 
                  [0 1 0 -1] * Gc * mean_vec >= -s_m]; 
              
    Objective = [p_p -p_m 0 0]*Gc*[0; 0; 1; -1] + 0.00 * ([1 0 -1 0] * Gc * [1; 0; -1; 0] + [0 1 0 -1] * Gc * [0; 1; 0; -1]);
    
    optimize(Constraint, [], opts);
    fprintf(1, 'Objective: %.4f\n', double(Objective));
    G_v = double(G);
    
    d = size(mu_p,1);
    
    X_p = zeros(d, num_x);
    X_m = zeros(d, num_x);
    for i=1:num_x
        x0_p = randn(d,1);
        x0_m = randn(d,1);

        A_0 = [x0_p x0_m mu_p mu_m M_p' M_m'];
        G_0 = A_0' * A_0;
        G_0 = G_0(sizeG:-1:1,sizeG:-1:1);
        G_v = G_v(sizeG:-1:1,sizeG:-1:1);
        R_0 = chol(G_0 + 1e-6 * eye(sizeG));
        R_v = chol(G_v + 1e-6 * eye(sizeG));
        R = R_0\R_v;
        R = R(sizeG:-1:1,sizeG:-1:1);
        G_0 = G_0(sizeG:-1:1,sizeG:-1:1);
        G_v = G_v(sizeG:-1:1,sizeG:-1:1);
        A = A_0 * R;
        X_p(:,i) = A(:,1);
        X_m(:,i) = A(:,2);
    end
end