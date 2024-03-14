clc;clear;close all
load ('/Users/g.park/Downloads/OCV_fit5.mat')
% 가중치 적용 최적화
window_size = 95;

x_values = OCV2(:,1);
y_values = OCV2(:,2);
OCV_mov = OCV2(:,1);

dvdq = diff(y_values) ./ diff(x_values);
dvdq = [dvdq; dvdq(end)];
dvdq_mov = movmean(dvdq, window_size);

%  주어진 x와 y 데이터 
y_values = dvdq_mov(:,1);

%  x 범위 설정
x_min = 0.1; 
x_max = 0.2; 
x_min2 = 0.8; 
x_max2 = 0.9; 

% 첫 번째 x 범위 내에서 가장 작은 y 값 찾기
x_in_range = x_values(x_values >= x_min & x_values <= x_max); 
y_in_range = y_values(x_values >= x_min & x_values <= x_max); 
[min_y1, min_index1] = min(y_in_range); 
corresponding_x1 = x_in_range(min_index1); 
min1 = find(OCV2(:,1) == corresponding_x1);

% 두 번째 x 범위 내에서 가장 작은 y 값 찾기
x_in_range2 = x_values(x_values >= x_min2 & x_values <= x_max2); 
y_in_range2 = y_values(x_values >= x_min2 & x_values <= x_max2); 
[min_y2, min_index2] = min(y_in_range2);    
corresponding_x2 = x_in_range2(min_index2); 
min2 = find(OCV2(:,1) == corresponding_x2);

% 결과 출력
fprintf('첫 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min, x_max, min_y1);
fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x1);
fprintf('두 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min2, x_max2, min_y2);
fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x2);


x3 = OCV2(:,1);
w1 = ones(size(dvdq_mov))*0.1;
start_index = min1; 
end_index = min2;
w1(start_index:end_index) = dvdq_mov(start_index:end_index);


%Plot: w1 = dvdq_mov;
figure(1);

plot(x3,w1,'-g');
xlabel('SOC');
ylabel('Weight');
xlim([0 1]);
title('w1(dvdq)');

customsettings;
% Save the file as PNG
print('w1(dvdq)','-dpng','-r300');


Cap = OCV2(:, 1);
Q_cell = abs(Cap(end));
x_guess = [0,Q_cell,1,Q_cell];
x_lb = [0,Q_cell*0.5,0,Q_cell*0.5];
x_ub = [1,Q_cell*2,1,1.6285];


results = struct();

counter = 1;


w1_values = {w1,dvdq_mov};
w1_values = cell2mat(w1_values);


% selected_values를 weights_range에 추가


for k = 1
        w1 = w1_values(:,k) .* ones(size(OCV2(:,1)));


      for j =[5]
         w2 = j .* ones(size(OCV2(:,1)));
        
        

%fmincon을 사용하여 최적화 수행  
options = optimoptions(@fmincon,'MaxIterations',100, 'StepTolerance',1e-10,'ConstraintTolerance', 1e-10, 'OptimalityTolerance', 1e-10);

problem = createOptimProblem('fmincon', 'objective', @(x) OCV_dvdq_model_07(x,OCP_n1,OCP_p1,OCV2,w1,w2), ...
            'x0', x_guess, 'lb', x_lb, 'ub', x_ub, 'options', options);
ms = MultiStart('Display','iter','UseParallel',true,'FunctionTolerance',1e-100,'XTolerance',1e-100);

[x_id,f_val, exitflag, output] = run(ms,problem,200);
[cost_hat,OCV_hat] = OCV_dvdq_model_07 (x_id,OCP_n1,OCP_p1,OCV2,w1,w2);



figure(2)
plot(OCV2(:, 1), OCV2(:, 2), 'g-');hold on;
plot(OCV2(:, 1), OCV_hat, 'm-');
xlabel('SOC');
ylabel('OCV (V)');
title('SOC vs. OCV');


customsettings;
% Save the file as PNG
%saveas(gcf,'combined plot.png')
print('OCV 피팅_c20_aged','-dpng','-r300');
hold off;


%dv/dq 생성
x = OCV2 (:,1);
y = OCV2 (:,2);

x_values = [];
for i = 1:(length(x)-1)
    dvdq77(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
     x_values = [x_values; x(i)];
end


y = OCV_hat (:,1);
for i = 1:(length(x) - 1)
    dvdq88(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i)); 
end


%dvdq에 이동 평균 적용
dvdq77_moving_avg = movmean(dvdq77(1:end), window_size);
dvdq88_moving_avg = movmean(dvdq88(1:end), window_size);


%Plot: dV/dQ 그래프
figure(3)
plot(x_values, dvdq77_moving_avg, 'b-'); hold on;
plot(x_values, dvdq88_moving_avg, 'r-');
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
title('SOC vs. dV/dQ');
ylim([0 3]);
xlim([0 1]);

customsettings;

% Save the file as PNG
print('dVdQ 그래프','-dpng','-r300');
hold off;

% %Plot: w1 = dvdq_mov;
% plot(OCV2(:,1),w1,'-g');
% xlabel('SOC');
% ylabel('dV/dQ /  V (mAh)^-1');
% xlim([0 1]);
% ylim([0 2]);



%peak값 추정
y_values = dvdq77_moving_avg(1,:); % y 값 배열
y_values2 = dvdq88_moving_avg(1,:); % y1 겂 배열 

% 첫 번째 x 범위
x_min = 0.21; % 최소 x 범위
x_max = 0.25; % 최대 x 범위

% 두 번째 x 범위
x_min2 = 0.27; % 다른 최소 x 범위
x_max2 = 0.36; % 다른 최대 x 범위

%세 번째 x 범위
x_min3 = 0.72; % 다른 최소 x 범위
x_max3 = 0.82; % 다른 최대 x 범위



% 첫 번째 x 범위 내에서 가장 큰 y 값 찾기
x_in_range = x_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 y 값 추출
 % 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
[max_y1, max_index1] = max(y_in_range);   
[max2_y1, max2_index1] = max(y_in_range2);

corresponding_x1 = x_in_range(max_index1);   % 해당 y 값에 대응하는 x 값 찾기
corresponding_x11 = x_in_range(max2_index1); % 해당 y2 값에 대응하는 x 값 찾기
sim_max1 = y_values2(max2_index1);

%OCV에서 해당하는 인덱스 찾기
index1 = find(OCV2(:,1) == corresponding_x1);
index11 = find(OCV2(:,1) == corresponding_x11);



% 두 번째 x 범위 내에서 가장 큰 y 값 찾기
x_in_range = x_values(x_values >= x_min2 & x_values <= x_max2); % 다른 x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min2 & x_values <= x_max2); % 다른 x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min2 & x_values <= x_max2);  % x 범위에 해당하는 y 값 추출
% 두 번째 x 범위 내에서 최소 y 값 및 해당 인덱스
[max_y2, max_index2] = max(y_in_range);
[max2_y2, max2_index2] = max(y_in_range2);

corresponding_x2 = x_in_range(max_index2);   % 해당 y 값에 대응하는 x 값 찾기
corresponding_x22 = x_in_range(max2_index2); % 해당 y2 값에 대응하는 x 값 찾기
sim_max2 = y_values2(max2_index2);

% OCV에서 해당하는 인덱스 찾기
index2 = find(OCV2(:,1) == corresponding_x2);
index22 = find(OCV2(:,1) == corresponding_x22);



% 세 번째 x 범위 내에서 가장 큰 y 값 찾기
x_in_range = x_values(x_values >= x_min3 & x_values <= x_max3); % 다른 x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min3 & x_values <= x_max3); % 다른 x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min3 & x_values <= x_max3);
%% 
% 세 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 
[max_y3, max_index3] = max(y_in_range);    
[max2_y3, max2_index3] = max(y_in_range2); 

corresponding_x3 = x_in_range(max_index3); % 해당 y 값에 대응하는 x 값 찾기
corresponding_x33 = x_in_range(max2_index3); % 해당 y2 값에 대응하는 x 값 찾기
sim_max3 = y_values2(max2_index3);

% OCV에서 해당하는 인덱스 찾기
index3 = find(OCV2(:,1) == corresponding_x3);
index33 = find(OCV2(:,1) == corresponding_x33);




e1 =sqrt((max_y1-sim_max1).^2);
e2 =sqrt((max_y2-sim_max2).^2);
e3 =sqrt((max_y3-sim_max3).^2);

e5 =sqrt((corresponding_x1-corresponding_x11).^2);
e6 =sqrt((corresponding_x2-corresponding_x22).^2);
e7 =sqrt((corresponding_x3-corresponding_x33).^2);



%peak값의 average
error_dvdq = sqrt((max_y1-sim_max1).^2+(max_y2-sim_max2).^2+(max_y3-sim_max3).^2);
error_soc = sqrt(((corresponding_x1-corresponding_x11).^2+(corresponding_x2-corresponding_x22).^2+(corresponding_x3-corresponding_x33).^2)/3);
disp(['error dvdq값: ', num2str(error_dvdq)]);
disp(['error soc값: ', num2str(error_soc)])
error_disp = ['e1: ', num2str(e1), ' e2: ', num2str(e2), ' e3: ', ' e5: ', num2str(e5), ' e6: ', num2str(e6), ' e7: ', num2str(e7)];



peak_position = ['Dpeak soc1: ', num2str(corresponding_x1), 'Fpeak soc1: ', num2str(corresponding_x11), 'Dpeak soc2: ' , num2str(corresponding_x2), 'Fpeak soc2: ', num2str(corresponding_x22), 'Dpeak soc3: ', num2str(corresponding_x3), ' Fpeak soc3: ', num2str(corresponding_x33)];


disp(['Dpeak soc1', num2str(corresponding_x1)]);
disp(['Fpeak soc1', num2str(corresponding_x11)]);
disp(['Dpeak soc2', num2str(corresponding_x2)]);
disp(['Fpeak soc2', num2str(corresponding_x22)]);
disp(['Dpeak soc3', num2str(corresponding_x3)]);
disp(['Fpeak soc3', num2str(corresponding_x33)]);



figure(4)
plot(x_values, dvdq77_moving_avg, 'b-','DisplayName', 'data'); hold on;
plot(x_values, dvdq88_moving_avg, 'r-','Color',[1,0,0,0.5],'DisplayName', 'fit');
ylim([0 2]);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
customsettings;
% Save the file as PNG
print('Peak(dVdQ)','-dpng','-r300');


% 별 마커와 텍스트 추가
x_star =[corresponding_x1,corresponding_x2,corresponding_x3]; 
roundedValue1 = round(x_star * 10^3) / 10^3;
y_star = dvdq77_moving_avg([index1,index2,index3]); 
star_handle = plot(x_star, y_star, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 12, 'DisplayName', 'Data Peak');
text(x_star, zeros(size(x_star))+0.2, arrayfun(@num2str, roundedValue1, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'b','FontSize', 8); 


% 네모 마커와 텍스트 추가
x_square = [corresponding_x11,corresponding_x22,corresponding_x33]; 
roundedValue2 = round(x_square * 10^3) / 10^3; 
y_square = dvdq88_moving_avg([index11,index22,index33]); 
square_handle = plot(x_square, y_square, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'Color', [1, 0, 0, 0.5], 'DisplayName', 'Fit Peak');
text(x_square, zeros(size(x_square))+0.1, arrayfun(@num2str, roundedValue2, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center','Color', 'r','FontSize', 8);  


% 네모표시에 대한 선
for i = 1:length(x_star)
    line([x_star(i), x_star(i)], [0, y_star(i)], 'LineStyle', ':', 'Color', 'b');
end
% 네모표시에 대한 선
for i = 1:length(x_square)
    line([x_square(i), x_square(i)], [0, y_square(i)], 'LineStyle', '--', 'Color','r');
end


legend([star_handle, square_handle], {'Data Peak', 'Fit Peak'});


% %OCP_n,OCP_p dvdq
x_1 = x_id(1,1) + (1/x_id(1,2));
y_1 = x_id(1,3) - (1/x_id(1,4));


OCP_n1(:,3) = ((OCP_n1(:,1)-x_id(1,1))/(x_1-x_id(1,1)));
OCP_p1(:,3) = ((OCP_p1(:,1)-x_id(1,3))/(y_1-x_id(1,3))); 


x = OCP_n1 (1:end,3);
y = OCP_n1 (1:end,2);


start_value = 0;
end_value = 1;



x_values2 = [];
for i = 1:(length(x) - 1)
    if x(i) >= start_value && x(i)<=end_value
    dvdq_n(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));   
    x_values2 = [x_values2; x(i)];
    end
end


x = OCP_p1 (1:end,3);
y = OCP_p1 (1:end,2);


x_values3 = [];
for i = 1:(length(x) - 1)
    if x(i) >= start_value && x(i)<=end_value
    dvdq_p(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));   
    x_values3 = [x_values3; x(i)];
    end
end





% dvdq_n에 이동 평균 적용
idx_start = find(OCP_n1 (1:end,3) >= 0);
first_different_idx1= idx_start(1);
dvdq_n = movmean(dvdq_n(first_different_idx1:end),window_size);
%x_values_moving_avg = movmean(x_values, window_size);


% dvdq_p에 이동 평균 적용
idx_start = find(OCP_p1 (1:end,3) >= 0);
first_different_idx2 = idx_start(1);
dvdq_p = movmean(dvdq_p(first_different_idx2:end),window_size);
%x_values2_moving_avg = movmean(x_values2, window_size);




% ocp_n,ocp_p플롯
% ocp_n,ocp_p플롯
figure(5)
min_length = min(length(x_values2), length(abs(dvdq_n)));
plot(x_values2(1:min_length), abs(dvdq_n(1:min_length)), 'b-'); hold on;
min_length = min(length(x_values3), length(abs(dvdq_p)));
plot(x_values3(1:min_length), abs(dvdq_p(1:min_length)), 'r-'); 
ylim([0 2]); hold off;
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');

customsettings;
% Save the file as PNG
print('Peak(dVdQ)','-dpng','-r300');


% 실제 관측값과 모델의 예측 값 사이의 차이

error = OCV_hat(min1:min2)-OCV2(min1:min2,2);
ERROR = OCV_hat-OCV2(:,2);

% 제곱을 계산하여 평균
squared_error = error.^2;
Squared_error = ERROR.^2;

mean_squared_error = mean(squared_error);
Mean_squared_error = mean(Squared_error);
root_squared_error = sqrt(error.^2);
Root_squared_error = sqrt(ERROR.^2);
% 평균값의 제곱근을 취하여 RMSE 값
rmse_value = sqrt(mean_squared_error);
RMSE_value = sqrt(Mean_squared_error);

disp(['MIN-MIN RMSE 값: ', num2str(rmse_value)]);
disp(['RMSE 값: ', num2str(rmse_value)]);




figure(6)
plot(OCV2(min1:min2,1),error,'DisplayName',['No. =' num2str(counter)]);
xlabel('SOC');
ylabel('Error (mV)');
xlim([0 1]);
xticks(0:0.1:1); 
customsettings;
% Save the file as PNG
print('Peak(dVdQ)','-dpng','-r300');
hold on;

d_error = dvdq77_moving_avg(min1:min2) - dvdq88_moving_avg(min1:min2);

figure(7)
plot(x_values(min1:min2),d_error,'DisplayName',['No. =' num2str(counter)]);
xlabel('SOC');
ylabel('Error (mV)');
xlim([0 1]);
xticks(0:0.1:1); 
customsettings;
% Save the file as PNG
print('Peak(dVdQ)','-dpng','-r300');
hold on;


        % 결과 저장
        results(counter).w1 = w1;
        results(counter).w2 = w2;
        results(counter).x_id = x_id;
        results(counter).f_val = f_val;
        results(counter).exitflag = exitflag;
        results(counter).output = output;
        results(counter).MINRMSE = rmse_value; 
        results(counter).RMSE = RMSE_value;
        results(counter).Error_soc = error_soc;
        results(counter).Error_dvdq = error_dvdq;
        results(counter).Error_each = error_disp;
        results(counter).each_peak_position = peak_position; 
        results(counter).OCV_hat = OCV_hat;
        results(counter).dvdq88_moving_avg = dvdq88_moving_avg;
        results(counter).dvdq77_moving_avg = dvdq77_moving_avg;
        results(counter).x_star = x_star;
        results(counter).x_square = x_square;
        results(counter).index1 = index1;
        results(counter).index2 = index2;
        results(counter).index3 = index3;

        results(counter).index11 = index11;
        results(counter).index22 = index22;
        results(counter).index33 = index33;



        results(counter).corresponding_x1 = corresponding_x1;
        results(counter).corresponding_x2 = corresponding_x2;
        results(counter).corresponding_x3 = corresponding_x3;

        results(counter).corresponding_x11 = corresponding_x11;
        results(counter).corresponding_x22 = corresponding_x22;
        results(counter).corresponding_x33 = corresponding_x33;

        
        results(counter).e5 = e5;
        results(counter).e6 = e6;
        results(counter).e7 = e7;


                counter = counter + 1;

        end
end



width = 6;     % Width in inches
height = 6;    % Height in inches
alw = 2;    % AxesLineWidth
fsz = 20;      % Fontsize
lw = 2;      % LineWidth
msz = 16;       % MarkerSize

Error_soc_values = [results.Error_soc]'; 
[soc_val,soc_idx] = min(Error_soc_values);

disp(['soc 값: ', num2str(soc_val)]);
disp(['soc 인덱스: ', num2str(soc_idx)]);


maxError = max([results(soc_idx).e5,results(soc_idx).e6,results(soc_idx).e7]);
disp(['Max_error: ', num2str(maxError)])


% plot


% plot

figure('position', [0 0 500 400] );
% 첫 번째 그래프
subplot(2, 1, 1);

plot(OCV2(:,1),OCV2(:,2),'b-', 'linewidth',2);
hold on;
plot(OCV2(:,1),results(soc_idx).OCV_hat,'r-', 'linewidth',2);
xlabel('SOC');
ylabel('OCV (V)');
title('OCV 피팅 결과');
yyaxis right;
ax = gca;  % 현재 축 객체 가져오기
ax.YColor = 'k';  % 검정색으로 설정
ylabel('Weight');
plot(OCV2(:,1),results(soc_idx).w2,'-g', 'linewidth',2);
legend('FCC data','FCC fit','Weight','Location', 'none', 'Position', [0.73, 0.93, 0.1 0.05],'fontsize',10);
xlim([0 1]);
ylim([0 10]);
customsettings;



%두 번째 그래프
subplot(2, 1, 2);

plot(x_values, [results(soc_idx).dvdq77_moving_avg]', 'b-', 'linewidth',2); hold on;
plot(x_values, [results(soc_idx).dvdq88_moving_avg]', 'r-','Color',[1,0,0,0.6], 'linewidth',2);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
title('dV/dQ 피팅 결과');
ylim([0 2]);
yyaxis right;
ax = gca; 
ax.YColor = 'k';  
ylabel('Weight');
plot(OCV2(:,1),results(soc_idx).w1,'-g','Color',[0,1,0,0.5], 'linewidth',2);

ylim([0 2]);
xlim([0 1]);


% 별 마커와 텍스트 추가
x_star = [results(soc_idx).corresponding_x1,results(soc_idx). corresponding_x2,results(soc_idx).corresponding_x3]; 
roundedValue1 = round(x_star * 10^3) / 10^3;
y_star = results(soc_idx).dvdq77_moving_avg([results(soc_idx). index1,results(soc_idx). index2, results(soc_idx). index3]); 
star_handle = plot(x_star, y_star, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 12, 'DisplayName', 'Data Peak');
text(x_star, zeros(size(x_star))+0.2, arrayfun(@num2str, roundedValue1, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'b','FontSize', 8); 


% 네모 마커와 텍스트 추가
x_square = [results(soc_idx). corresponding_x11,results(soc_idx).corresponding_x22,results(soc_idx). corresponding_x33]; 
roundedValue2 = round(x_square * 10^3) / 10^3; 
y_square = results(soc_idx).dvdq88_moving_avg([results(soc_idx). index11,results(soc_idx). index22,results(soc_idx). index33]); 
square_handle = plot(x_square, y_square, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'Color', [1, 0, 0, 0.5], 'DisplayName', 'Fit Peak');
text(x_square, zeros(size(x_square))+0.1, arrayfun(@num2str, roundedValue2, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center','Color', 'r','FontSize', 8);  



% 네모표시에 대한 선
for i = 1:length(x_star)
    line([x_star(i), x_star(i)], [0, y_star(i)], 'LineStyle', ':', 'Color', 'b');
end
% 네모표시에 대한 선
for i = 1:length(x_square)
    line([x_square(i), x_square(i)], [0, y_square(i)], 'LineStyle', '--', 'Color','r');
end


legend([star_handle, square_handle], {'Data Peak', 'Fit Peak'},'Position', [0.73, 0.45, 0.1 0.05], 'fontsize',10);

customsettings;
% Save the file as PNG
hold off;
print('결과 그래프','-dpng','-r300');


function customsettings
width = 6;     % Width in inches
height = 6;    % Height in inches
alw = 2;    % AxesLineWidth
fsz = 20;      % Fontsize
lw = 6;      % LineWidth
msz = 16;       % MarkerSize
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); 
set(gca, 'FontSize', fsz, 'LineWidth', alw); 

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

if ispc % Use Windows ghostscript call
  system('gswin64c -o -q -sDEVICE=png256 -dEPSCrop -r300 -oimprovedExample_eps.png improvedExample.eps');
else % Use Unix/OSX ghostscript call
  system('gs -o -q -sDEVICE=png256 -dEPSCrop -r300 -oimprovedExample_eps.png improvedExample.eps');
end
end







function [cost,OCV_sim] = OCV_dvdq_model_07(x, OCP_n1, OCP_p1, OCV2, w1, w2)
    x_0 = x(1);
    QN = x(2);
    y_0 = x(3);
    QP = x(4);

    Cap = OCV2(:, 1);
    if (OCV2(end, 2) < OCV2(1, 2)) % Discharge OCV
        x_sto = -(Cap - Cap(1)) / QN + x_0;
        y_sto = (Cap - Cap(1)) / QP + y_0;
    else  % Charge OCV
        x_sto = (Cap - Cap(1)) / QN + x_0;
        y_sto = -(Cap - Cap(1)) / QP + y_0;
    end

    OCP_n_sim = interp1(OCP_n1(:, 1), OCP_n1(:, 2), x_sto, 'linear', 'extrap');
    OCP_p_sim = interp1(OCP_p1(:, 1), OCP_p1(:, 2), y_sto, 'linear', 'extrap');
    OCV_sim = OCP_p_sim - OCP_n_sim;

    
    % dV/dQ 값들 계산
    window_size = 95;

    x_values = OCV2(:, 1);
    y_values = OCV2(:, 2);
    y_sim_values = OCV_sim(:, 1);

    dvdq = diff(y_values) ./ diff(x_values);
    dvdq_sim = diff(y_sim_values) ./ diff(x_values);
    dvdq = [dvdq; dvdq(end)];
    dvdq_mov = movmean(dvdq, window_size);

    dvdq_sim = [dvdq_sim; dvdq_sim(end)];
    dvdq_sim_mov = movmean(dvdq_sim,window_size);

    OCV_sim_mov =  movmean(OCV_sim,window_size);

    OCV_mov = movmean(OCV2(:,2),window_size);

    cost_dvdq = sum(((dvdq_sim_mov - dvdq_mov).^2./mean(dvdq_mov)).*w1);

    % OCV 비용 계산
    cost_OCV = sum(((OCV_sim_mov - OCV_mov).^2./mean(OCV_mov)).*w2);
   
    % 비용 합산 
    cost = cost_dvdq + cost_OCV;
end
    