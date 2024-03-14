



clc;clear;close all


%% Config
window_size = 95;
load ('G:\Shared drives\BSL-Data\Processed_data\Hyundai_dataset\OCV\AHC_(5)_OCV_C20.mat')
ocpn = OCV_golden.OCVdis;
clear OCV_golden OCV_all Q_cell
load ('G:\Shared drives\BSL-Data\Processed_data\Hyundai_dataset\OCV\CHC_(5)_OCV_C20.mat')
ocpp = OCV_golden.OCVchg;
clear OCV_golden OCV_all Q_cell


% file loop / detect
% load Merged data
load ('G:\Shared drives\BSL_Data2\HNE_AgingDOE_Processed\HNE_FCC\4CPD 1C (25-42)\25degC\HNE_FCC_4CPD 1C (25-42)_25degC_s01_3_6_Merged.mat')


% figure(1)
% yyaxis left
% plot(ocpn(:,1),ocpn(:,2)); hold on
% yyaxis right
% plot(ocpp(:,1),ocpp(:,2)); hold on

% loop over steps
% find OCV index


j_count = 0;
tic;
for i = 1:size(data_merged,1)
    
    data_merged(i).step = i;


    % detect OCV
    if data_merged(i).OCVflag == 1
    
    j_count = j_count+1; % number of OCV now

    data_merged(i).sococv = [data_merged(i).soc data_merged(i).V];
    %% 

    soc = data_merged(i).sococv(:,1); % SOC
    ocv = data_merged(i).sococv(:,2); % OCV
    q = data_merged(i).cumQ(:,1); % assuemd charging
 
    
    
    %% Initial guess
     
            if j_count == 1
                Q_cell = abs(data_merged(i).Q); % 1
                x_guess = [0,Q_cell,1,Q_cell]; % x0, Qn, y0, Qp
                x_lb = [0,  Q_cell*0.5, 0.8,  Q_cell*0.5];
                x_ub = [0.2,Q_cell*2,   1,  Q_cell*2];
            else

            % detect previous OCV
                i_prev = find([data_merged(1:i-1).OCVflag] ==1,1,'first');
            
                Qp_prev = data_merged(i_prev).ocv_para_hat(4);
                Qn_prev = data_merged(i_prev).ocv_para_hat(2);

                Q_cell = abs(data_merged(i).Q); % 1
                x_guess =   [0,     min(Q_cell,Qn_prev),     1,      min(Q_cell,Qp_prev)]; % x0, Qn, y0, Qp
                x_lb =      [0,     Q_cell*0.5, 0.8,    Q_cell*0.5];
                x_ub =      [0.2,   Qn_prev,    1,      Qp_prev];
                
            end

    %%
    dvdq = diff(ocv)./diff(q);
    dvdq = [dvdq; dvdq(end)];
    dvdq_mov = movmean(dvdq, window_size);


    %% Weighting

        % OCV weighting
        w_ocv = 0*ones(size(q)); % empirical scale

        % dvdq weighting
        soc_min1 = 0.1;
        soc_max1 = 0.2;
        soc_min2 = 0.8;
        soc_max2 = 0.9;
        
        soc_range1 = soc(soc >= soc_min1 & soc <= soc_max1);
        dvdq_range1 = dvdq_mov(soc >= soc_min1 & soc <= soc_max1);
        [ocv_min1, ind_min1] = min(dvdq_range1);
        soc_min1 = soc_range1(ind_min1);
        ind_min1_tot = find(soc == soc_min1);
    
        soc_range2 = soc(soc >= soc_min2 & soc <= soc_max2);
        dvdq_range2 = dvdq_mov(soc >= soc_min2 & soc <= soc_max2);
        [ocv_min2, ind_min2] = min(dvdq_range2);
        soc_min2 = soc_range2(ind_min2);
        ind_min2_tot = find(soc == soc_min2);
        
        w_dvdq = ones(size(dvdq_mov))*0.1; % empiricial
        w_dvdq(ind_min1_tot:ind_min2_tot) = dvdq_mov(ind_min1_tot:ind_min2_tot);

%         figure()
%         plot(soc,w_dvdq,'-g');
%         xlabel('SOC');
%         ylabel('Weight');
%         xlim([0 1]);
%         title('w1(dvdq)');
        %ylim([0 2])

        %% Fitting 
     
        options = optimoptions(@fmincon,'MaxIterations',100, 'StepTolerance',1e-10,'ConstraintTolerance', 1e-10, 'OptimalityTolerance', 1e-10);

        problem = createOptimProblem('fmincon', 'objective', @(x)func_ocvdvdq_cost(x,ocpn,ocpp,[q ocv],w_dvdq,w_ocv,window_size), ...
            'x0', x_guess, 'lb', x_lb, 'ub', x_ub, 'options', options);
        ms = MultiStart('Display','iter','UseParallel',true,'FunctionTolerance',1e-100,'XTolerance',1e-100);

        [x_hat, f_val, exitflag, output] = run(ms,problem,100);
        [cost_hat, ocv_hat, dvdq_mov, dvdq_sim_mov] = func_ocvdvdq_cost(x_hat,ocpn,ocpp,[q ocv],w_dvdq,w_ocv,window_size);

        
%         figure()
%         plot(soc,ocv); hold on
%         plot(soc,ocv_hat); hold on
% 
%         figure()
%         plot(soc,dvdq_mov); hold on
%         plot(soc,dvdq_sim_mov); hold on
%         ylim([0 2])


        data_merged(i).ocv_para_hat = x_hat;
        data_merged(i).ocv_hat = ocv_hat;
        data_merged(i).dvdq_mov = dvdq_mov;
        data_merged(i).dvdq_sim_mov = dvdq_sim_mov;


    end
end
toc

data_ocv = data_merged([data_merged.OCVflag]' == 1);
J = size(data_ocv,1);
c_mat = lines(J);

for j = 1:size(data_ocv,1)

% (temporal) plot
soc_now =  data_ocv(j).sococv(:,1);
ocv_now = data_ocv(j).sococv(:,2);
ocv_sim_now = data_ocv(j).ocv_hat;
dvdq_now = data_ocv(j).dvdq_mov;
dvdq_sim_now = data_ocv(j).dvdq_sim_mov;

figure()
set(gcf,'position',[100,100,1600,800])

subplot(1,2,1)
plot(soc_now,ocv_now,'-','color',c_mat(j,:)); hold on
plot(soc_now,ocv_sim_now,'--','color',c_mat(j,:));
subplot(1,2,2)
plot(soc_now,dvdq_now,'-','color',c_mat(j,:)); hold on
plot(soc_now,dvdq_sim_now,'--','color',c_mat(j,:));
ylim([0 400])


%% (LAMP, LLI)
data_ocv(j).LAMp = data_ocv(1).ocv_para_hat(4)...
                    -data_ocv(j).ocv_para_hat(4);

data_ocv(j).LAMn = data_ocv(1).ocv_para_hat(2)...
                    -data_ocv(j).ocv_para_hat(2); 

data_ocv(j).LLI = (data_ocv(1).ocv_para_hat(4)*data_ocv(1).ocv_para_hat(3) + data_ocv(1).ocv_para_hat(2)*data_ocv(1).ocv_para_hat(1))...
                    -(data_ocv(j).ocv_para_hat(4)*data_ocv(j).ocv_para_hat(3) + data_ocv(j).ocv_para_hat(2)*data_ocv(j).ocv_para_hat(1)); 

data_ocv(j).dQ_LLI = (data_ocv(1).ocv_para_hat(4)*(data_ocv(1).ocv_para_hat(3)-1))...
                    -(data_ocv(j).ocv_para_hat(4)*(data_ocv(j).ocv_para_hat(3)-1));


data_ocv(j).dQ_LAMp = (data_ocv(1).ocv_para_hat(4) - data_ocv(j).ocv_para_hat(4))...
                        *(1-data_ocv(1).ocv_para_hat(3)+data_ocv(1).Q/data_ocv(1).ocv_para_hat(4));


data_ocv(j).dQ_data = data_ocv(1).Q - data_ocv(j).Q;

dQ_data_now = data_ocv(j).dQ_data;
dQ_total_now = data_ocv(j).dQ_LLI + data_ocv(j).dQ_LAMp;

scale_now = dQ_data_now/dQ_total_now;

% manipulate loss scale to be consistent with the data Q
data_ocv(j).dQ_LLI = data_ocv(j).dQ_LLI*scale_now;
data_ocv(j).dQ_LAMp = data_ocv(j).dQ_LAMp*scale_now;



end

figure()
bar([data_ocv.cycle],[[data_ocv.dQ_LLI];[data_ocv.dQ_LAMp]],'stacked')
legend({'Loss by LLI','Loss by LAMp'}); hold on
plot([data_ocv.cycle],[data_ocv.dQ_data])


function [cost, ocv_sim, dvdq_mov, dvdq_sim_mov] = func_ocvdvdq_cost(x,ocpn,ocpp,ocv,w_dvdq,w_ocv,window_size)

    % assign parameters
    x_0 = x(1);
    QN = x(2);
    y_0 = x(3);
    QP = x(4);

    Cap = ocv(:, 1);
    if (ocv(end, 2) < ocv(1, 2)) % Discharge OCV
        x_sto = -(Cap - Cap(1)) / QN + x_0;
        y_sto = (Cap - Cap(1)) / QP + y_0;
    else  % Charge OCV
        x_sto = (Cap - Cap(1)) / QN + x_0;
        y_sto = -(Cap - Cap(1)) / QP + y_0;
    end

    ocpn_sim = interp1(ocpn(:, 1), ocpn(:, 2), x_sto, 'linear', 'extrap');
    ocpp_sim = interp1(ocpp(:, 1), ocpp(:, 2), y_sto, 'linear', 'extrap');
    ocv_sim = ocpp_sim - ocpn_sim;
    
    dvdq = diff(ocv(:,2))./diff(ocv(:,1));
    dvdq_sim = diff(ocv_sim) ./diff(ocv(:,1));
    
    dvdq = [dvdq; dvdq(end)];
    dvdq_sim = [dvdq_sim; dvdq_sim(end)];
    
    dvdq_mov = movmean(dvdq, window_size);
    dvdq_sim_mov = movmean(dvdq_sim,window_size);

    % cost 
    cost_ocv = sum(w_ocv.*(ocv_sim - ocv(:,2)).^2./mean(ocv(:,2)));

    cost_dvdq = sum(w_dvdq.*(dvdq_sim_mov - dvdq_mov).^2./mean(dvdq_mov));

    % total cost
    cost = cost_ocv + cost_dvdq;



end