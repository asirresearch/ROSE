function main
    close all;
    clear;

    %% 0. 初始化参数
    xyz_start = [0, -120.02, 533.96];
    xyz_end = [-30, -45, 385];
    T = 10;

    %% 1. 七段加加速度轨迹规划
    L = sqrt((xyz_end(1) - xyz_start(1))^2 + (xyz_end(2) - xyz_start(2))^2 + (xyz_end(3) - xyz_start(3))^2);

    dt = T / 7;
    v1 = L / (16 * dt);
    J = 2 * v1 / (dt^2);
    amax = dt * J;
    v2 = v1 + dt * amax;
    vmax = v2 + v1;

    t1 = 1 * dt;  t2 = 2 * dt;  t3 = 3 * dt;
    t4 = 4 * dt;  t5 = 5 * dt;  t6 = 6 * dt;  t7 = 7 * dt;
    t = 0:0.1:T;

    vt1 = 1/2 * J * t.^2 .* (t >= 0 & t < t1);
    vt2 = (v1 + amax * (t - t1)) .* (t >= t1 & t < t2);
    vt3 = (vmax - 1/2 * J * (t3 - t).^2) .* (t >= t2 & t < t3);
    vt4 = vmax .* (t >= t3 & t < t4);
    vt5 = (vmax - 1/2 * J * (t - t4).^2) .* (t >= t4 & t < t5);
    vt6 = (v2 - amax * (t - t5)) .* (t >= t5 & t < t6);
    vt7 = 1/2 * J * (t7 - t).^2 .* (t >= t6 & t < t7);

    vt = vt1 + vt2 + vt3 + vt4 + vt5 + vt6 + vt7;

    S = zeros(1, length(t));
    for i = 2:length(t)
        S(i) = trapz(t(1:i), vt(1:i));
    end

    x_s = xyz_start(1) + (xyz_end(1) - xyz_start(1)) / L * S;
    y_s = xyz_start(2) + (xyz_end(2) - xyz_start(2)) / L * S;
    z_s = xyz_start(3) + (xyz_end(3) - xyz_start(3)) / L * S;

    v_x = (xyz_end(1) - xyz_start(1)) / L * vt;
    v_y = (xyz_end(2) - xyz_start(2)) / L * vt;
    v_z = (xyz_end(3) - xyz_start(3)) / L * vt;

    %% 2. PID控制参数
    Kp = 2.4;
    Ki = 0.1;
    Kd = 0.15;
    lamda = 0.08;

    th = zeros(6, length(t));
    x = [xyz_start'; 0; 0; 0];
    e_sum = zeros(6, 1);
    e = zeros(6, length(t));

    % ⭐ 新增：末端笛卡尔误差
    e_cart = zeros(1, length(t));

    %% 轨迹跟踪循环
    for i = 1:(length(t) - 1)
        xd = [x_s(i); y_s(i); z_s(i); 0; 0; 0];
        dxd = [v_x(i); v_y(i); v_z(i); 0; 0; 0];

        q = th(:, i);
        Jac = Jacob_cross_SDH(q');
        e(:, i) = xd - x;

        % 欧氏距离误差（笛卡尔误差）
        e_cart(i) = norm(e(1:3, i));

        de = dxd - Jac * (th(:, i + 1) - th(:, i)) / 0.1;
        u = Kp * e(:, i) + Ki * e_sum + Kd * de;
        e_sum = e_sum + e(:, i) * 0.1;

        dth = inv(Jac + lamda * diag(ones(1, 6))) * u;
        th(:, i + 1) = th(:, i) + dth * 0.1;
        x = xd;
    end

    %% 最后一个时间步
    i = length(t);
    xd = [x_s(i); y_s(i); z_s(i); 0; 0; 0];
    q = th(:, i);
    Jac = Jacob_cross_SDH(q');
    e(:, i) = xd - x;

    % 末端笛卡尔误差（最后一点）
    e_cart(i) = norm(e(1:3, i));

    de = dxd - Jac * (th(:, i) - th(:, i-1)) / 0.1;
    u = Kp * e(:, i) + Ki * e_sum + Kd * de;
    e_sum = e_sum + e(:, i) * 0.1;
    dth = inv(Jac + lamda * diag(ones(1, 6))) * u;
    th(:, i) = th(:, i-1) + dth * 0.1;
    x = xd;

    %% 3. 绘图
    % ⭐ 末端笛卡尔误差图
    figure(2)
    plot(t, e_cart, 'k', 'linewidth', 2);
    xlabel('时间/s', 'FontName', '黑体', 'FontSize', 12);
    ylabel('误差/mm', 'FontName', '黑体', 'FontSize', 12);
    title('末端笛卡尔误差', 'FontName', '黑体', 'FontSize', 12);
    grid on;

    % 关节角度曲线
    figure(3)
    for i = 1:6
        subplot(2, 3, i);
        plot(t, th(i, :), 'LineWidth', 1.5, 'color', 'k');
        xlabel('时间', 'FontName', '黑体', 'FontSize', 12);
        ylabel('角度值/rad', 'FontName', '黑体', 'FontSize', 12);
        grid on;
        title(['第', num2str(i), '关节角度变化曲线'], 'FontName', '黑体', 'FontSize', 12)
    end
end


%% ============================
%        雅可比函数
% ============================
function [J] = Jacob_cross_SDH(q)

    d = [105.03, 0, 0, 75.66, 80.09, 44.36];
    a = [0, -174.42, -174.42, 0, 0, 0];
    alp = [pi/2, 0, 0, pi/2, -pi/2, 0];
    offset = [0, -pi/2, 0, -pi/2, 0, 0];
    thd = q + offset;

    T1 = trotz(thd(1)) * transl(0, 0, d(1)) * trotx(alp(1)) * transl(a(1), 0, 0);
    T2 = trotz(thd(2)) * transl(0, 0, d(2)) * trotx(alp(2)) * transl(a(2), 0, 0);
    T3 = trotz(thd(3)) * transl(0, 0, d(3)) * trotx(alp(3)) * transl(a(3), 0, 0);
    T4 = trotz(thd(4)) * transl(0, 0, d(4)) * trotx(alp(4)) * transl(a(4), 0, 0);
    T5 = trotz(thd(5)) * transl(0, 0, d(5)) * trotx(alp(5)) * transl(a(5), 0, 0);
    T6 = trotz(thd(6)) * transl(0, 0, d(6)) * trotx(alp(6)) * transl(a(6), 0, 0);

    T01 = T1;
    T02 = T1 * T2;
    T03 = T02 * T3;
    T04 = T03 * T4;
    T05 = T04 * T5;
    T06 = T05 * T6;

    R00 = eye(3);
    R01 = t2r(T01);
    R02 = t2r(T02);
    R03 = t2r(T03);
    R04 = t2r(T04);
    R05 = t2r(T05);
    R06 = t2r(T06);

    Z0 = R00(:,3);
    Z1 = R01(:,3);
    Z2 = R02(:,3);
    Z3 = R03(:,3);
    Z4 = R04(:,3);
    Z5 = R05(:,3);

    P06 = T06(1:3,4);
    P16 = T2*T3*T4*T5*T6;
    P16 = P16(1:3,4);
    P26 = T3*T4*T5*T6;
    P26 = P26(1:3,4);
    P36 = T4*T5*T6;
    P36 = P36(1:3,4);
    P46 = T5*T6;
    P46 = P46(1:3,4);
    P56 = T6(1:3,4);

    J1 = [cross(Z0, P06); Z0];
    J2 = [cross(Z1, P16); Z1];
    J3 = [cross(Z2, P26); Z2];
    J4 = [cross(Z3, P36); Z3];
    J5 = [cross(Z4, P46); Z4];
    J6 = [cross(Z5, P56); Z5];

    J = [J1 J2 J3 J4 J5 J6];
end
