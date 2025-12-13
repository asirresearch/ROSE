close all;
clear;

%% ===================  0. 读取 CSV 轨迹  =======================
data = readmatrix('hand_trajectory2.csv');

x_s = data(:,1)';   % x轨迹
y_s = data(:,2)';   % y轨迹
z_s = data(:,3)';   % z轨迹

N = length(x_s);        % 轨迹长度
T = 10;                 % 总时长（保持原值）
t = linspace(0, T, N);  % 构造时间轴

dt = t(2) - t(1);

% 简单差分求速度
v_x = [0 diff(x_s)] / dt;
v_y = [0 diff(y_s)] / dt;
v_z = [0 diff(z_s)] / dt;

%% ===================  1. 初始变量 =======================
dth = zeros(6,1);
th = zeros(6,1);

% 初始末端位置从 CSV 的第一个点来
x = [x_s(1); y_s(1); z_s(1); 0; 0; 0];

lamda = 1;
ita = 0.0002;
c = 5;

e = zeros(6, N);
de = zeros(6, N);
e_cart = zeros(1, N);   % 末端整体笛卡尔误差

%% ===================  2. 控制循环 =======================
for i = 1:N
    % 期望位置 / 速度
    xd = [x_s(i); y_s(i); z_s(i); 0;0;0];
    dxd = [v_x(i); v_y(i); v_z(i); 0;0;0];

    if i == 1
        q = zeros(6,1);   % 初始关节角为零
    else
        q = th(:, i);
    end

    % 雅可比矩阵函数（用户自定义）
    Jac = Jacob_cross_SDH(q');  

    % 误差
    e(:, i) = xd - x(:,i);
    e_cart(i) = norm(e(1:3,i));   % 欧氏距离误差

    % 控制律
    s = c * e(:, i);
    v = dxd + (1/c)*ita*sign(s);

    de(:, i) = dxd - v;

    % 关节角速度
    dth(:, i) = (Jac + lamda * diag(ones(1,6))) \ v;

    % 更新关节角
    th(:, i+1) = th(:, i) + dth(:, i) * dt;

    % 更新末端位置（简化积分）
    x(:, i+1) = x(:, i) + v * dt;
end

e_cart(end) = e_cart(end-1);  % 修补最后一点

%% ===================  3. 末端笛卡尔误差图 =======================
figure(2)
plot(t, e_cart, 'k', 'LineWidth', 2);
xlabel('时间/s','FontName','黑体');
ylabel('误差/mm','FontName','黑体');
title('末端笛卡尔误差（欧氏距离）','FontName','黑体');
grid on;

%% ===================  4. 关节角变化图 =======================
figure(3)
for k = 1:6
    subplot(2,3,k);
    plot(t, th(k, 1:N), 'k', 'LineWidth', 1.5);
    xlabel('时间'); ylabel('角度/rad');
    title(['关节', num2str(k), '角度变化']);
    grid on;
end

%% ===================  5. 末端轨迹对比（期望 vs 实际） =======================
figure(4); clf; hold on;

% 期望轨迹 —— 红色虚线
plot3(x_s, y_s, z_s, 'r--', 'LineWidth', 2);

% 实际轨迹 —— 小蓝点
idx = 1:5:N;
plot3(x(1,idx), x(2,idx), x(3,idx), 'bo', ...
      'MarkerSize', 3, 'MarkerFaceColor','b');

% --- 坐标标签（英文数字 Times New Roman，中文 SimSun） ---
xlabel('\fontname{Times New Roman}X/mm', ...
    'FontName','SimSun','FontSize',7);   % 中文仍能显示宋体

ylabel('\fontname{Times New Roman}Y/mm', ...
    'FontName','SimSun','FontSize',7);

zlabel('\fontname{Times New Roman}Z/mm', ...
    'FontName','SimSun','FontSize',7);

% --- 标题与图例（双字体） ---
title(['\fontname{SimSun}轨迹跟踪对比（', ...
       '\fontname{Times New Roman}Expected: red dashed ; Actual: blue dots', ...
       '\fontname{SimSun}）'], ...
       'FontName','SimSun','FontSize',7);

legend({ ...
    '\fontname{SimSun}期望轨迹（\fontname{Times New Roman}red dashed\fontname{SimSun}）', ...
    '\fontname{SimSun}实际轨迹（\fontname{Times New Roman}blue dots\fontname{SimSun}）' ...
    }, ...
    'FontSize',7, 'Interpreter','tex');

% --- 坐标轴刻度数字（Times New Roman 六号） ---
set(gca,'FontName','Times New Roman','FontSize',7);

grid on;
axis equal;
view(45,30);

%% ===================  6. 量化指标计算 =======================
RMSE = sqrt(mean(e_cart.^2));
MaxError = max(e_cart);
MeanError = mean(e_cart);

fprintf('轨迹跟踪量化指标：\n');
fprintf('均方误差 RMSE = %.3f mm\n', RMSE);
fprintf('最大误差 MaxError = %.3f mm\n', MaxError);
fprintf('平均误差 MeanError = %.3f mm\n', MeanError);

% 可选：计算允许误差阈值下的跟踪成功率
epsilon = 4.5;  % mm
SuccessRate = sum(e_cart < epsilon) / N * 100;
fprintf('允许误差 %.1f mm 的跟踪成功率 = %.2f%%\n', epsilon, SuccessRate);

%% ===================  7. 可视化指标 =======================
e_cart_smooth = movmean(e_cart, 10);  

figure(5); clf;
plot(t, e_cart_smooth, 'b', 'LineWidth',1.5); hold on;

% RMSE 直线（英文数字全部是 Times New Roman）
yline(RMSE, 'r--', '\fontname{Times New Roman}RMSE', 'LineWidth',1.5, ...
    'Interpreter','tex');

% ======== 坐标轴标签（中文宋体 + 英文数字 Times New Roman） ========
xlabel('\fontname{SimSun}时间/\fontname{Times New Roman}s', ...
    'FontSize',7.5, 'Interpreter','tex');

ylabel('\fontname{SimSun}末端误差/\fontname{Times New Roman}mm', ...
    'FontSize',7.5, 'Interpreter','tex');

% ======== 标题（中英混排，同样双字体） ========
title(['\fontname{SimSun}末端误差随时间变化与', ...
       '\fontname{Times New Roman}RMSE'], ...
      'FontSize',7.5, 'Interpreter','tex');

% ======== 图例（中文 SimSun + 英文数字 TimesNewRoman） ========
legend({ ...
    '\fontname{SimSun}瞬时误差', ...
    '\fontname{Times New Roman}RMSE' ...
    }, ...
    'FontSize',7.5, 'Interpreter','tex');

grid on;

% ======== 坐标轴刻度（纯数字 → Times New Roman 六号） ========
set(gca, 'FontName','Times New Roman', 'FontSize',7);

