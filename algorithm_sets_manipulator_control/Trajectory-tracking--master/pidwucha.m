function main
    close all;
    clear;

    %% 0. 初始化参数
    % 轨迹起点和终点
    xyz_start = [0, -120.02, 533.96]; % 起点
    xyz_end = [-30, -45, 385]; % 终点
    T = 10; % 完成轨迹规划的时间

    %% 1. 轨迹规划
    % 计算轨迹长度
    L = sqrt((xyz_end(1) - xyz_start(1))^2 + (xyz_end(2) - xyz_start(2))^2 + (xyz_end(3) - xyz_start(3))^2);

    % 时间参数
    dt = T / 7; % 每段的时间长度
    v1 = L / (16 * dt); % 第一次加速度拐点
    J = 2 * v1 / (dt * dt); % 加加速度
    amax = dt * J; % 最大加速度
    v2 = v1 + dt * amax; % 第二次加速度拐点
    vmax = v2 + v1; % 第三次速度拐点

    % 时间节点
    t1 = 1 * dt;
    t2 = 2 * dt;
    t3 = 3 * dt;
    t4 = 4 * dt;
    t5 = 5 * dt;
    t6 = 6 * dt;
    t7 = 7 * dt;

    % 时间向量
    t = 0:0.1:T;

    % 速度规划
    vt1 = 1/2 * J * t.^2 .* (t >= 0 & t < t1);
    vt2 = (v1 + amax * (t - t1)) .* (t >= t1 & t < t2);
    vt3 = (vmax - 1/2 * J * (t3 - t).^2) .* (t >= t2 & t < t3);
    vt4 = vmax .* (t >= t3 & t < t4);
    vt5 = (vmax - 1/2 * J * (t - t4).^2) .* (t >= t4 & t < t5);
    vt6 = (v2 - amax * (t - t5)) .* (t >= t5 & t < t6);
    vt7 = (1/2 * J * (t7 - t).^2) .* (t >= t6 & t < t7);

    vt = vt1 + vt2 + vt3 + vt4 + vt5 + vt6 + vt7; % 各时刻速度

    % 位移计算
    S = zeros(1, length(t)); % 各时刻位移
    for i = 2:length(t)
        S(i) = trapz(t(1:i), vt(1:i));
    end

    % 各时刻xyz的位移
    x_s = xyz_start(1) + (xyz_end(1) - xyz_start(1)) / L * S;
    y_s = xyz_start(2) + (xyz_end(2) - xyz_start(2)) / L * S;
    z_s = xyz_start(3) + (xyz_end(3) - xyz_start(3)) / L * S;

    % 各时刻xyz轴的速度分量
    v_x = (xyz_end(1) - xyz_start(1)) / L * vt;
    v_y = (xyz_end(2) - xyz_start(2)) / L * vt;
    v_z = (xyz_end(3) - xyz_start(3)) / L * vt;

    % 绘制轨迹
    figure(1)
    subplot(1, 3, 1)
    plot(t, x_s, 'k', 'linewidth', 1);
    xlabel('时间', 'FontName', '黑体', 'FontSize', 12);
    ylabel('x轴位移', 'FontName', '黑体', 'FontSize', 12);
    title('x轴', 'FontName', '黑体', 'FontSize', 12)
    axis square;
    grid on;

    subplot(1, 3, 2)
    plot(t, y_s, 'k', 'linewidth', 1);
    xlabel('时间', 'FontName', '黑体', 'FontSize', 12);
    ylabel('y轴位移', 'FontName', '黑体', 'FontSize', 12);
    title('y轴', 'FontName', '黑体', 'FontSize', 12)
    axis square;
    grid on;

    subplot(1, 3, 3)
    plot(t, z_s, 'k', 'linewidth', 1);
    xlabel('时间', 'FontName', '黑体', 'FontSize', 12);
    ylabel('z轴位移', 'FontName', '黑体', 'FontSize', 12);
    title('z轴', 'FontName', '黑体', 'FontSize', 12)
    axis square;
    grid on;

    %% 2. 轨迹跟踪 - PID控制器
    % PID控制器参数
    Kp = 2.4;  % 比例增益
    Ki = 0.1; % 积分增益
    Kd = 0.15; % 微分增益
    lamda = 0.08; % 正则化项

    % 初始化变量
    th = zeros(6, length(t)); % 关节角度
    x = [xyz_start'; 0; 0; 0]; % 初始位姿
    e_sum = zeros(6, 1); % 误差的积分
    e = zeros(6, length(t)); % 初始化误差矩阵

    % 调整循环范围，避免索引超出范围
    for i = 1:(length(t) - 1)
        % 期望位姿和速度
        xd = [x_s(i); y_s(i); z_s(i); 0; 0; 0];
        dxd = [v_x(i); v_y(i); v_z(i); 0; 0; 0];

        % 当前实际位姿
        q = th(:, i);
        Jac = Jacob_cross_SDH(q'); % 雅可比矩阵
        e(:, i) = xd - x; % 误差
        de = dxd - Jac * (th(:, i + 1) - th(:, i)) / 0.1; % 误差的微分

        % PID控制器计算控制输入
        u = Kp * e(:, i) + Ki * e_sum + Kd * de;

        % 更新误差积分
        e_sum = e_sum + e(:, i) * 0.1;

        % 计算关节角的增量
        dth = inv(Jac + lamda * diag(ones(1, 6))) * u;

        % 更新关节角度和位姿
        th(:, i + 1) = th(:, i) + dth * 0.1;
        x = xd; % 假设系统完美跟踪
    end

    % 处理最后一个时间步
    i = length(t);
    xd = [x_s(i); y_s(i); z_s(i); 0; 0; 0];
    q = th(:, i);
    Jac = Jacob_cross_SDH(q');
    e(:, i) = xd - x; % 误差
    de = dxd - Jac * (th(:, i) - th(:, i - 1)) / 0.1; % 误差的微分

    % PID控制器计算控制输入
    u = Kp * e(:, i) + Ki * e_sum + Kd * de;

    % 更新误差积分
    e_sum = e_sum + e(:, i) * 0.1;

    % 计算关节角的增量
    dth = inv(Jac + lamda * diag(ones(1, 6))) * u;

    % 更新关节角度和位姿
    th(:, i) = th(:, i - 1) + dth * 0.1;
    x = xd; % 假设系统完美跟踪

    %% 3. 绘制结果
    % 位置跟踪误差
figure(2)
plot(t, e(1:3, :), 'LineWidth', 2);

% 坐标轴标签（中文宋体 + 英文 Times New Roman）
xlabel('\fontname{SimSun}时间/\fontname{Times New Roman}s', ...
    'FontSize', 7.5, 'Interpreter','tex');
ylabel('\fontname{SimSun}误差/\fontname{Times New Roman}mm', ...
    'FontSize', 7.5, 'Interpreter','tex');

% 图例（中英混排）
legend({
    '\fontname{Times New Roman}x\fontname{SimSun}轴的位置跟踪误差', ...
    '\fontname{Times New Roman}y\fontname{SimSun}轴的位置跟踪误差', ...
    '\fontname{Times New Roman}z\fontname{SimSun}轴的位置跟踪误差' ...
    }, ...
    'FontSize', 7.5, 'Interpreter','tex');

% 标题
title('\fontname{SimSun}位置跟踪误差曲线', ...
    'FontSize', 7.5, 'Interpreter','tex');

grid on;
axis on;

% 坐标轴刻度数字 → Times New Roman 六号
set(gca, 'FontName','Times New Roman', 'FontSize',7);

    % 关节角度变化曲线
    figure(3)
    for i = 1:6
        subplot(2, 3, i);
        plot(t, th(i, :), 'LineWidth', 1.5, 'color', 'k');
        xlabel('时间', 'FontName', '黑体', 'FontSize', 12);
        ylabel('角度值/rad', 'FontName', '黑体', 'FontSize', 12);
        set(gca, 'XLim', [min(t) max(t)]);
        grid on;
        title(['第', num2str(i), '关节角度变化曲线'], 'FontName', '黑体', 'FontSize', 12)
    end
end

function [J] = Jacob_cross_SDH(q)
    % JACOB_CROSS_SDH 函数摘要
    % 输入q为逼近角，单位为弧度，矩阵大小1*6;
    % 输出J为速度雅各比矩阵，矩阵大小6*6；
    % 说明：利用向量积的方法求解系统的雅各比矩阵，方法1和方法2任选一种
    % 说明：此求解方法基于SDH参数建模，若MDH方法建模，需进行一定的下标改动

    d = [105.03, 0, 0, 75.66, 80.09, 44.36];
    a = [0, -174.42, -174.42, 0, 0, 0];
    alp = [pi/2, 0, 0, pi/2, -pi/2, 0];
    offset = [0, -pi/2, 0, -pi/2, 0, 0];
    thd = q + offset;

    % 求各个关节间的变换矩阵
    T0 = trotz(0) * transl(0, 0, 0) * trotx(0) * transl(0, 0, 0);
    T1 = trotz(thd(1)) * transl(0, 0, d(1)) * trotx(alp(1)) * transl(a(1), 0, 0);
    T2 = trotz(thd(2)) * transl(0, 0, d(2)) * trotx(alp(2)) * transl(a(2), 0, 0);
    T3 = trotz(thd(3)) * transl(0, 0, d(3)) * trotx(alp(3)) * transl(a(3), 0, 0);
    T4 = trotz(thd(4)) * transl(0, 0, d(4)) * trotx(alp(4)) * transl(a(4), 0, 0);
    T5 = trotz(thd(5)) * transl(0, 0, d(5)) * trotx(alp(5)) * transl(a(5), 0, 0);
    T6 = trotz(thd(6)) * transl(0, 0, d(6)) * trotx(alp(6)) * transl(a(6), 0, 0);

    % 求各个关节相对于惯性坐标系的变换矩阵
    T00 = T0;
    T01 = T1;
    T02 = T1 * T2;
    T03 = T1 * T2 * T3;
    T04 = T1 * T2 * T3 * T4;
    T05 = T1 * T2 * T3 * T4 * T5;
    T06 = T1 * T2 * T3 * T4 * T5 * T6;

    % 求各个关节相对于末端坐标系的变换矩阵
    T06 = T1 * T2 * T3 * T4 * T5 * T6;
    T16 = T2 * T3 * T4 * T5 * T6;
    T26 = T3 * T4 * T5 * T6;
    T36 = T4 * T5 * T6;
    T46 = T5 * T6;
    T56 = T6;

    % 提取各变换矩阵的旋转矩阵
    R00 = t2r(T00);
    R01 = t2r(T01);
    R02 = t2r(T02);
    R03 = t2r(T03);
    R04 = t2r(T04);
    R05 = t2r(T05);
    R06 = t2r(T06);

    % 取旋转矩阵第3列，即Z轴方向分量
    Z0 = R00(:, 3);
    Z1 = R01(:, 3);
    Z2 = R02(:, 3);
    Z3 = R03(:, 3);
    Z4 = R04(:, 3);
    Z5 = R05(:, 3);
    Z6 = R06(:, 3);

    %% Method.1
    % 求末端关节坐标系相对于前面各个坐标系的位置，即齐次变换矩阵的第四列
    % pi6为坐标系i和末端坐标系的相对位置在坐标系i下的表示
    P06 = T06(1:3, 4);
    P16 = T16(1:3, 4);
    P26 = T26(1:3, 4);
    P36 = T36(1:3, 4);
    P46 = T46(1:3, 4);
    P56 = T56(1:3, 4);
    P66 = [0; 0; 0];

    % 使用向量积求出雅可比矩阵
    % R0i为坐标系0到坐标系i的旋转矩阵
    % R0i*Pi6指坐标系i和末端坐标系的相对位置在0坐标系下的表示
    J1 = [cross(Z0, R00 * P06); Z0];
    J2 = [cross(Z1, R01 * P16); Z1];
    J3 = [cross(Z2, R02 * P26); Z2];
    J4 = [cross(Z3, R03 * P36); Z3];
    J5 = [cross(Z4, R04 * P46); Z4];
    J6 = [cross(Z5, R05 * P56); Z5];

    J = [J1, J2, J3, J4, J5, J6];
end