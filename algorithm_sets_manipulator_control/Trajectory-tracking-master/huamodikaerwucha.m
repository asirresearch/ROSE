close all;
clear;

%% 0. 轨迹起点终点
xyz_start=[0,-120.02,533.96];
xyz_end=[-30,-45,385];
T=10; %轨迹规划总时间

%% 1. 七段加加速度轨迹规划
L=sqrt((xyz_end(1)-xyz_start(1))^2+(xyz_end(2)-xyz_start(2))^2+(xyz_end(3)-xyz_start(3))^2);
dt=T/7;

v1=L/(16*dt);
J=2*v1/(dt*dt);
amax=dt*J;
v2=v1+dt*amax;
vmax=v2+v1;

t1 = 1*dt; t2 = 2*dt; t3 = 3*dt;
t4 = 4*dt; t5 = 5*dt; t6 = 6*dt; t7 = 7*dt;

t=0:0.1:T;

vt1=1/2*J*t.^2.*(t>=0 & t<t1);
vt2=(v1+amax*(t-t1)).*(t>=t1 & t<t2);
vt3=(vmax-1/2*J*(t3-t).^2).*(t>=t2 & t<t3);
vt4=vmax.*(t>=t3 & t<t4);
vt5=(vmax-1/2*J*(t-t4).^2).*(t>=t4 & t<t5);
vt6=(v2-amax*(t-t5)).*(t>=t5 & t<t6);
vt7=(1/2*J*(t7-t).^2).*(t>=t6 & t<t7);

vt=vt1+vt2+vt3+vt4+vt5+vt6+vt7; %速度

S=zeros(1,length(t));
for i=2:length(t)
    S(i)=trapz(t(1:i),vt(1:i));
end

% 各时刻三维轨迹
x_s=xyz_start(1)+(xyz_end(1)-xyz_start(1))/L*S;
y_s=xyz_start(2)+(xyz_end(2)-xyz_start(2))/L*S;
z_s=xyz_start(3)+(xyz_end(3)-xyz_start(3))/L*S;

% 各时刻速度
v_x=(xyz_end(1)-xyz_start(1))/L*vt;
v_y=(xyz_end(2)-xyz_start(2))/L*vt;
v_z=(xyz_end(3)-xyz_start(3))/L*vt;

%% 2. 轨迹跟踪（简化控制）
dth = zeros(6,1);
th = zeros(6,1);

x=[xyz_start';0;0;0];

lamda=1;
k = 0.1;
ita = 0.0002;
c = 5;
alpha = 0;

e = zeros(6,length(t));
de = zeros(6,length(t));
e_cart = zeros(1,length(t));   % 新增：末端笛卡尔误差

for i = 1 : length(t)
    xd=[x_s(i);y_s(i);z_s(i);0;0;0]; 
    dxd=[v_x(i);v_y(i);v_z(i);0;0;0];

    q=th(:, i);

    Jac = Jacob_cross_SDH(q');  % 你自己的雅可比

    e(:, i) = xd - x(:,i);
    e_cart(i) = norm(e(1:3,i));    % 新增：欧氏距离误差

    s = c*e(:, i);
    v=dxd + (1/c)*ita*sign(s);

    de(:, i) = dxd - v;

    dth(:, i) = inv(Jac+lamda.*diag(ones(1,6)))*v;
    th(:, i + 1) = th(:, i) + dth(:, i)*0.1;

    x(:, i+1) = x(:, i) + v*0.1;
end

e_cart(end) = e_cart(end-1);   % 修补最后一点

%% 3. 误差图：改为笛卡尔整体误差
figure(2)
plot(t, e_cart, 'k', 'LineWidth', 2);
xlabel('时间/s','FontName','黑体');
ylabel('误差/mm','FontName','黑体');
title('末端笛卡尔误差（欧氏距离）','FontName','黑体');
grid on;

%% 4. 关节角变化图（保持你的格式）
figure(3)
for k=1:6
    subplot(2,3,k);
    plot(t, th(k, 1:length(t)),'k','LineWidth',1.5);
    xlabel('时间'); ylabel('角度/rad');
    title(['关节',num2str(k),'角度变化']);
    grid on;
end

%% 5. 末端轨迹对比图（期望 + 实际）
figure(4); clf; hold on;

plot3(x_s, y_s, z_s, 'r--', 'LineWidth', 2);      % 期望轨迹

idx = 1:5:length(t);
plot3(x(1,idx), x(2,idx), x(3,idx), 'bo', ...
      'MarkerSize', 3, 'MarkerFaceColor','b');   % 实际轨迹点

xlabel('X/mm','FontSize',12);
ylabel('Y/mm','FontSize',12);
zlabel('Z/mm','FontSize',12);

title('末端轨迹对比（期望：红虚线；实际：蓝小点）','FontSize',12);
legend('期望轨迹（红虚线）','实际轨迹点（蓝色小点）');
grid on;
axis equal;
view(45,30);
