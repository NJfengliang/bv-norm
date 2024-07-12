
clear;
close all

ubc; % Running the ubc.m file to generate the boundary conditions

load bc;
global bc_cond
num = 1;

notch_left = 0.49;
notch_right = 0.51;
for i=1:num
    %--PDE solver----------------------------------------------
    bc_cond = f_bc(i, :);
    model = createpde;
    % 构建两个基本几何图形并做布尔运算
    % 10个元素，前两34代表是长方形，后分分别是4个点的x坐标、y坐标
    R1 = [3, 4, 0, 1,1, 0,0, 0, 1, 1  ]';
    R2 = [3, 4, notch_left, notch_right, notch_right, notch_left, 0, 0, 0.4, 0.4]';

    E1_ = [4;0.5;1;0.4;.2;0.3]; % Outside ellipse
    E1 = [E1_;zeros(length(R1) - length(E1_),1)];

    % E2_ = [4;0.5;-0.3;0.4;.2;1.4]; % Outside ellipse
    % E2 = [E2_;zeros(length(R1) - length(E2_),1)];

    C1_ = [1,0.5+1e-12, 0.5-1e-12,0.04]';
    C1 = [C1_;zeros(length(R1) - length(C1_),1)];
    C2_ = [1,0,0.4+1e-12, 0.1]';
    C2 = [C2_;zeros(length(R1) - length(C2_),1)];
    C3_ = [1,1.04, 0.6-1e-12,0.1]';
    C3 = [C3_;zeros(length(R1) - length(C3_),1)];
    gm = [R1,R2,C1,E1,C2,C3];

    sf = 'R1-E1-C1-R2-C2-C3';

    ns = char('R1','E1','C1','R2','C2','C3');

    ns = ns';
    g = decsg(gm,sf,ns);

    geometryFromEdges(model,g);
%     pdegplot(model,'EdgeLabels','on')

    applyBoundaryCondition(model,'dirichlet','Edge',1, 'u', 0);
    applyBoundaryCondition(model,'dirichlet','Edge',2, 'u', 0);
    applyBoundaryCondition(model,'dirichlet','Edge',3:11, 'u', @bcvalues, 'Vectorized', 'on');
    % % applyBoundaryCondition(model,'dirichlet','Edge',5, 'u', @bcvalues, 'Vectorized', 'on');
    % applyBoundaryCondition(model,'dirichlet','Edge',8, 'u', @bcvalues, 'Vectorized', 'on');
    % applyBoundaryCondition(model,'dirichlet','Edge',9:10, 'u', @bcvalues, 'Vectorized', 'on');
    % % applyBoundaryCondition(model,'dirichlet','Edge',9:10, 'u', 0);
%     applyBoundaryCondition(model,'dirichlet','Edge',11, 'u', 0);
    applyBoundaryCondition(model,'dirichlet','Edge',12, 'u', 1);
    applyBoundaryCondition(model,'dirichlet','Edge',13, 'u', 1);
    applyBoundaryCondition(model,'dirichlet','Edge',14, 'u', 1);
    applyBoundaryCondition(model,'dirichlet','Edge',15, 'u', 1);
    applyBoundaryCondition(model,'dirichlet','Edge',16, 'u', 1);
%     applyBoundaryCondition(model,'dirichlet','Edge',17, 'u', 0);
%     applyBoundaryCondition(model,'dirichlet','Edge',18, 'u', 0);
%     applyBoundaryCondition(model,'dirichlet','Edge',19, 'u', 0);
%     applyBoundaryCondition(model,'dirichlet','Edge',20, 'u', 0);
%     applyBoundaryCondition(model,'dirichlet','Edge',21, 'u', 0);
%     applyBoundaryCondition(model,'dirichlet','Edge',22, 'u', 0);
applyBoundaryCondition(model,'dirichlet','Edge',17:22, 'u', @bcvalues, 'Vectorized', 'on');



    specifyCoefficients(model,'m',0,...
        'd',0,...
        'c',1,...
        'a',0,...
        'f',10);
    % 
    hmax = 0.03;
    generateMesh(model,'Hmax',hmax,'GeometricOrder','linear');
%     figure
%     pdemesh(model);
    results = solvepde(model);
    X = results.Mesh.Nodes;
    X = X';
    xx = X(:, 1);
    yy = X(:, 2);
    u = results.NodalSolution;
    u_field(i, :) = u';
    if mod(i,100)==0
        disp(i);
    end
    
end
num1 = 101;
x_bc = linspace(0, 1, num1);
MeshNodes    = model.Mesh.Nodes;
MeshElements = model.Mesh.Elements;
%writeSurfaceMesh(mesh,"dacy.stl");



save('mult_holes', 'x_bc', 'f_bc', 'xx', 'yy', 'u_field','MeshNodes','MeshElements');
disp('Finished!');



    
    %end