clear all;
close all;
clc;

global ALPHA; ALPHA = 0.01;
global VERBOSE; VERBOSE = 1;



% %{   NETWORK NEUROPHYSIOLOGY STATISTICS   }%
% FN_DATA = 'pitchnet_paper_stats_data_neurophysiology_2021JUL12.json';
% DATA = jsondecode(fileread(FN_DATA));
% DATA_NAMES = fieldnames(DATA);
% 
% mmANOVA(DATA,...
%     {'natural_speech_music',},...
%     'bernox2005', 'fc_top_low_harm', 'low_harm', [1,30]);




%{   NETWORK PSYCHOPHYSICS STATISTICS   }%
FN_DATA = 'pitchnet_paper_stats_data_psychophysics_2021AUG05.json';
DATA = jsondecode(fileread(FN_DATA));
DATA_NAMES = fieldnames(DATA);


% twosample_f0dl_bernox(DATA, 'IHC9000Hz', 'IHC3000Hz', 0, 1);
% twosample_f0dl_bernox(DATA, 'IHC6000Hz', 'IHC3000Hz', 0, 1);
% twosample_f0dl_bernox(DATA, 'IHC1000Hz', 'IHC3000Hz', 0, 1);
% twosample_f0dl_bernox(DATA, 'IHC0320Hz', 'IHC3000Hz', 0, 1);
% twosample_f0dl_bernox(DATA, 'IHC0050Hz', 'IHC3000Hz', 0, 1);

% twosample_transition_point_bernox(DATA, 'IHC9000Hz', 'IHC3000Hz');
% twosample_transition_point_bernox(DATA, 'IHC6000Hz', 'IHC3000Hz');
% twosample_transition_point_bernox(DATA, 'IHC1000Hz', 'IHC3000Hz');
% twosample_transition_point_bernox(DATA, 'IHC0320Hz', 'IHC3000Hz');
% twosample_transition_point_bernox(DATA, 'IHC0050Hz', 'IHC3000Hz');

% mmANOVA(DATA,...
%     {'IHC0050Hz', 'IHC0320Hz', 'IHC1000Hz', 'IHC3000Hz', 'IHC6000Hz', 'IHC9000Hz'},...
%     'f0dlspl', 'f0dl', 'dbspl', [10.0, 100.0]);


% twosample_f0dl_bernox(DATA, 'BW20eN1', 'BW10eN1', 0, 1);
% twosample_f0dl_bernox(DATA, 'BW05eN1', 'BW10eN1', 0, 1);
% twosample_f0dl_bernox(DATA, 'BWlinear', 'BW10eN1', 0, 1);
% twosample_transition_point_bernox(DATA, 'BW20eN1', 'BW10eN1');
% twosample_transition_point_bernox(DATA, 'BW05eN1', 'BW10eN1');
% twosample_transition_point_bernox(DATA, 'BWlinear', 'BW10eN1');
% twosample_validation_accuracy(DATA, 'BW20eN1', 'BW10eN1')



% mmANOVA(DATA,...
%     {'natural', 'natural_hp'},...
%     'bernox2005', 'f0dl', 'low_harm', [1,30]);
% mmANOVA(DATA,...
%     {'natural', 'natural_lp'},...
%     'bernox2005', 'f0dl', 'low_harm', [1,30]);

% transposedtones_ttest(DATA, {'natural', 'natural_hp'})

% twosample_f0dl_bernox(DATA, 'spch_only', 'inst_only', 0, 1);

% mmANOVA(DATA,...
%     {'natural', 'noise_low', 'noise_none'},...
%     'bernox2005', 'f0dl', 'low_harm', [2,5]);


% twosample_human_model_similarity(DATA, 'noise_none', 'natural', 'bernox2005');
% twosample_human_model_similarity(DATA, 'noise_none', 'natural', 'altphasecomplexes');
% twosample_human_model_similarity(DATA, 'noise_none', 'natural', 'freqshiftedcomplexes');
% twosample_human_model_similarity(DATA, 'noise_none', 'natural', 'mistunedharmonics');
% twosample_human_model_similarity(DATA, 'noise_none', 'natural', 'transposedtones');

% twosample_human_model_similarity(DATA, 'BWlinear', 'BW10eN1', 'bernox2005');
% twosample_human_model_similarity(DATA, 'BWlinear', 'BW10eN1', 'altphasecomplexes');


function d = cohend(x1, x2)
% Compute Cohen's d to quantify effect size between two groups
% ( https://en.wikipedia.org/wiki/Effect_size#Cohen's_d )
n1 = length(x1);
n2 = length(x2);
v1 = var(x1);
v2 = var(x2);
d = (mean(x2) - mean(x1)) / sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1 + n2 - 2));
end


function print_ttest2(x1, x2)
% Display results of two-sample t-test
global ALPHA;
global VERBOSE;
[h,p,ci,stats] = ttest2(x1, x2, 'Alpha', ALPHA);
if VERBOSE
    fprintf('h=%d, p=%0.4e, t(%d)=%0.2f, d=%.2f, sd=%.2f, ci=[%.2f,%.2f]\n',...
        h, p, stats.df, stats.tstat, cohend(x1, x2), stats.sd, ci(1), ci(2));
else
    fprintf('h=%d, p=%0.4e, t(%d)=%0.2f, d=%.2f\n', h, p, stats.df, stats.tstat, cohend(x1, x2));
end
end



function print_signtest(x1, x2)
% Display results of two-sample sign-test
global ALPHA;
[p,h,stats] = signtest(x1, x2, 'Alpha', ALPHA);
fprintf('h=%d, p=%0.4e, stats.zval=%d, stats.sign=%d\n',...
    h, p, stats.zval, stats.sign);
end


function twosample_f0dl_bernox(...
    DATA, model_tag1, model_tag2, phase_mode, low_harm)
% Used to compare bernox2005 thresholds between two model conditions
data_tag1 = [model_tag1, '_bernox2005'];
data_tag2 = [model_tag2, '_bernox2005'];

low_harm_list = DATA.(data_tag1).low_harm;
phase_mode_list = DATA.(data_tag1).phase_mode;
if length(low_harm) == 2
    cond_idx = (low_harm_list >= low_harm(1)) &...
        (low_harm_list <= low_harm(2)) &...
        (phase_mode_list == phase_mode);
else
    cond_idx = (low_harm_list==low_harm) & (phase_mode_list==phase_mode);
end

f0dl_model_tag1 = DATA.(data_tag1).f0dl(:, cond_idx);
f0dl_model_tag1(f0dl_model_tag1 > 100.0) = 100.0; % F0DLs capped at 100%F0
f0dl_model_tag2 = DATA.(data_tag2).f0dl(:, cond_idx);
f0dl_model_tag1(f0dl_model_tag1 > 100.0) = 100.0; % F0DLs capped at 100%F0
x1 = log10(f0dl_model_tag1(:)); % Stats performed on log-transformed F0DLs
x2 = log10(f0dl_model_tag2(:)); % Stats performed on log-transformed F0DLs

fprintf('\nComparing f0dl_bernox of `%s` and `%s`:\n',...
    data_tag1, data_tag2)
disp('t-test');
print_ttest2(x1, x2)
end


function effect_size_f0dl_bernox(DATA, list_model_tag)
% Used to quantify effect size of lowest harmonic number on F0DL
for itr1 = 1:length(list_model_tag)
    data_tag = [list_model_tag{itr1}, '_bernox2005'];
    cond_idx = DATA.(data_tag).phase_mode == 0;
    f0dl = DATA.(data_tag).f0dl(:, cond_idx);
    f0dl(f0dl > 100.0) = 100.0;
%     f0dl_min_values = min(f0dl, [], 2);
%     f0dl_max_values = max(f0dl, [], 2);
    f0dl_lharm01 = f0dl(:, 1);
    f0dl_lharm30 = f0dl(:, size(f0dl, 2));
    d = cohend(log10(f0dl_lharm01), log10(f0dl_lharm30));
    mean_f0dl = 10.^mean(log10(f0dl), 1);
    factor_increase = max(mean_f0dl) ./ min(mean_f0dl);
    disp([data_tag,...
        sprintf(' | Cohen d (lharm=01 vs. lharm=30) = %.2f ', d),...
        sprintf(' | mean_F0DL range = [%.2f, %.2f]', min(mean_f0dl), max(mean_f0dl)),...
        sprintf(' | max_mean_F0DL/min_mean_F0DL = %.2f ', factor_increase)]);
end
end


function twosample_transition_point_bernox(...
    DATA, model_tag1, model_tag2)
% Used to compare bernox2005 thresholds between two model conditions
data_tag1 = [model_tag1, '_bernox2005'];
data_tag2 = [model_tag2, '_bernox2005'];

phase_mode_list = DATA.(data_tag1).phase_mode;
low_harm_list = DATA.(data_tag1).low_harm;
phase_mode_idx = phase_mode_list == 0;
f0dl_model_tag1 = DATA.(data_tag1).f0dl(:, phase_mode_idx);
f0dl_model_tag2 = DATA.(data_tag2).f0dl(:, phase_mode_idx);
phase_mode_list = phase_mode_list(phase_mode_idx);
low_harm_list = low_harm_list(phase_mode_idx);

% Transition point is defined as the first lowest harmonic number with
% F0 discrimination thresholds > 1.0%
low_harm_idx_model_tag1 = f0dl_model_tag1 > 1.0;
low_harm_idx_model_tag2 = f0dl_model_tag2 > 1.0;

N_subjects = size(f0dl_model_tag1, 1);
transition_point_model_tag1 = zeros(N_subjects, 1);
transition_point_model_tag2 = zeros(N_subjects, 1);
for itr0 = 1:N_subjects
    tmp_low_harm_list1 = low_harm_list(low_harm_idx_model_tag1(itr0, :));
    tmp_low_harm_list2 = low_harm_list(low_harm_idx_model_tag2(itr0, :));
    if length(tmp_low_harm_list1) > 1
        transition_point_model_tag1(itr0) = min(tmp_low_harm_list1);
    else
        transition_point_model_tag1(itr0) = max(low_harm_list);
    end
    if length(tmp_low_harm_list2) > 2
        transition_point_model_tag2(itr0) = min(tmp_low_harm_list2);
    else
        transition_point_model_tag2(itr0) = max(low_harm_list);
    end
end
x1 = transition_point_model_tag1;
x2 = transition_point_model_tag2;
fprintf('\nComparing transition_point_bernox of `%s` and `%s`:\n',...
    data_tag1, data_tag2)
disp('t-test');
print_ttest2(x1, x2)
end



function twosample_human_model_similarity(...
    DATA, model_tag1, model_tag2, expt_tag)
% Used to compare human_model_similarity_coef between two model conditions
global VERBOSE;
data_tag1 = [model_tag1, '_', expt_tag];
data_tag2 = [model_tag2, '_', expt_tag];
x1 = DATA.(data_tag1).human_model_similarity_coef(:);
p1 = DATA.(data_tag1).human_model_similarity_pval(:);
x2 = DATA.(data_tag2).human_model_similarity_coef(:);
p2 = DATA.(data_tag1).human_model_similarity_pval(:);
% Summarize individiual human vs. model similarity metrics
fprintf('\nComparing human_model_similarity_coef of `%s` and `%s`:\n',...
    data_tag1, data_tag2)
% if VERBOSE
%     disp([model_tag1,...
%         sprintf(' | mean coef +/- s.d. = %.2f +/- %.2f', mean(x1), std(x1)),...
%         sprintf(' | coef range = [%.2f, %.2f]', min(x1), max(x1)),...
%         sprintf(' | pval range = [%.3e, %.3e]', min(p1), max(p1))])
%     disp([model_tag2,...
%         sprintf(' | mean coef +/- s.d. = %.2f +/- %.2f', mean(x2), std(x2)),...
%         sprintf(' | coef range = [%.2f, %.2f]', min(x2), max(x2)),...
%         sprintf(' | pval range = [%.3e, %.3e]', min(p2), max(p2))])
% end
% Stats tests
disp('::: t-test with norminv(coef/2 + 1/2)');
print_ttest2(norminv(x1/2 + 1/2), norminv(x2/2 + 1/2));
disp('::: t-test');
print_ttest2(x1, x2);
if VERBOSE
    disp('::: sign-test');
    print_signtest(x1, x2);
end
end



function twosample_human_model_similarity_combined(...
    DATA, model_tag1, model_tag2)
% Used to compare human_model_similarity_coef between two model conditions
global VERBOSE;
expt_tag_list = { 'bernox2005', 'altphasecomplexes', 'freqshiftedcomplexes', 'mistunedharmonics', 'transposedtones'};
for eki = 1:length(expt_tag_list)
    data_tag1 = [model_tag1, '_', expt_tag_list{eki}];
    data_tag2 = [model_tag2, '_', expt_tag_list{eki}];

    x1 = DATA.(data_tag1).human_model_similarity_coef(:);
    p1 = DATA.(data_tag1).human_model_similarity_pval(:);
    x2 = DATA.(data_tag2).human_model_similarity_coef(:);
    p2 = DATA.(data_tag1).human_model_similarity_pval(:);

    if eki == 1
        combined_similarity_ranks = zeros(length(x1)+length(x2), 1);
    end

    [~, sort_idx] = sort([x1;x2], 'descend');
    ranks = transpose(1:length(combined_similarity_ranks));
    ranks = ranks(sort_idx);
    combined_similarity_ranks = combined_similarity_ranks + ranks;
end

x1 = combined_similarity_ranks(1:length(x1));
x2 = combined_similarity_ranks(length(x1)+1:end);
fprintf('\nComparing EXPERIMENT-COMBINED human_model_similarity_coef rankings of `%s` and `%s`:\n',...
    model_tag1, model_tag2)
fprintf('%s : mean rank +/- s.d. = %.2f +/- %.2f\n', model_tag1, mean(x1), std(x1))
fprintf('%s : mean rank +/- s.d. = %.2f +/- %.2f\n', model_tag2, mean(x2), std(x2))
% Stats tests
disp('::: sign-test');
print_signtest(x1, x2);
disp('::: t-test');
print_ttest2(x1, x2);
end



function twosample_validation_accuracy(...
    DATA, model_tag1, model_tag2)
% Used to compare validation accuracy between two model conditions
global VERBOSE;
data_tag1 = [model_tag1, '_bernox2005'];
data_tag2 = [model_tag2, '_bernox2005'];
x1 = DATA.(data_tag1).validation_accuracy(:);
x2 = DATA.(data_tag2).validation_accuracy(:);
% Summarize individiual human vs. model similarity metrics
fprintf('\nComparing validation accuracy of `%s` and `%s`:\n',...
    model_tag1, model_tag2)
% Stats tests
disp('::: t-test');
print_ttest2(x1, x2);
if VERBOSE
    disp('::: sign-test');
    print_signtest(x1, x2);
end
end



function transposedtones_ttest(DATA, list_model_tag)
%
global ALPHA
global VERBOSE
for itr1 = 1:length(list_model_tag)
    data_tag = [list_model_tag{itr1}, '_transposedtones'];
    f0dl = DATA.(data_tag).f0dl;
    f0dl(f0dl > 100.0) = 100.0;
    log10_f0dl = log10(f0dl);
    f_carrier = DATA.(data_tag).f_carrier;
    PT_log10_f0dl = log10_f0dl(:, f_carrier == 0);
    TT_log10_f0dl = log10_f0dl(:, f_carrier > 1);

    disp([data_tag, '| two-sample t-test between mean pure tone and transpsoed tone thresholds'])
    disp('::: t-test');
    x1 = mean(PT_log10_f0dl, 2);
    x2 = mean(TT_log10_f0dl, 2);
    print_ttest2(x1, x2);
    if VERBOSE
        disp('::: sign-test');
        print_signtest(x1, x2);
    end
    fprintf('\n')
end
end



function [tbl, rm] = mmANOVA(DATA, list_model_tag, varargin)
%{
Wrapper function around `simple_mixed_anova_partialeta.m` from Malinda.
Prints out mmANOVA results with Greenhouse-Geisser correction.
%}
expt_tag = 'bernox2005';
resp_tag = 'f0dl';
cond_tag = 'low_harm';
cond_range = [];
if length(varargin) >= 1
    expt_tag = varargin{1};
    if length(varargin) >= 2
        resp_tag = varargin{2};
        if length(varargin) >= 3
            cond_tag = varargin{3};
            if length(varargin) >= 4
                cond_range = varargin{4};
            end
        end
    end
end

offset = 0;
for itr1 = 1:length(list_model_tag)

    if isfield(DATA, list_model_tag{itr1})
        SUBDATA = DATA.(list_model_tag{itr1});
        data_tag = expt_tag;
    else
        SUBDATA = DATA;
        data_tag = [list_model_tag{itr1}, '_', expt_tag];
    end

    resp = SUBDATA.(data_tag).(resp_tag);
    cond = SUBDATA.(data_tag).(cond_tag);

    if isempty(cond_range)
        filt_idx = cond == cond;
    else
        filt_idx = (cond >= cond_range(1)) & (cond <= cond_range(2));
    end

    % Special adjustments for F0 discrimination experiments
    if contains(resp_tag, 'f0dl')
        % F0 discrimination thresholds are capped at 100% and
        % log-transformed before being used for statistics.
        resp(resp > 100.0) = 100.0;
        resp = log10(resp);
        if contains(expt_tag, 'bernox2005')
            % Only use sine-phase F0 discrimination thresholds
            filt_idx = filt_idx & (SUBDATA.(data_tag).phase_mode == 0);
        end
    end

    resp = resp(:, filt_idx);
    cond = cond(filt_idx, :);

    if itr1 == 1
        between_factors = zeros(length(list_model_tag) * size(resp, 1), 1);
        within_factor_names = {cond_tag};
        between_name = list_model_tag{itr1};
        datamat = zeros(length(between_factors), length(cond));
    end
    between_name = intersect(between_name, list_model_tag{itr1}, 'stable');

    IDX = (1:size(resp, 1)) + offset;
    between_factors(IDX) = itr1;
    datamat(IDX, :) = resp;
    offset = offset + size(resp, 1);
end
between_factor_names = {between_name};
if length(list_model_tag) == 1
    % Adjustment for one factor RMANOVA (just within subject effects)
    % (Based on demo code from M.McPherson, May, 2020)
    between_factors = [];
end
[tbl,rm] = simple_mixed_anova_partialeta(...
    datamat, between_factors);%, within_factor_names, between_factor_names);

% csvwrite('/Users/mark/Desktop/datamat.csv', datamat)
% csvwrite('/Users/mark/Desktop/factors_between_subjects.csv', between_factors)
% csvwrite('/Users/mark/Desktop/factors_within_subjects.csv', cond)

fprintf('\n____________________________________________________________\n')
fprintf('BETWEEN FACTOR (%s): %s\n', 'BS01', strjoin(list_model_tag, '   |   '))
fprintf('WITHIN FACTOR (%s): %d levels of %s\n', 'WS01', length(cond), strjoin(within_factor_names))
disp(tbl);

mauchly_table = mauchly(rm);
epsilon_table = epsilon(rm);
epsilon_gg = epsilon_table.GreenhouseGeisser(1);
fprintf('Greenhouse-Geisser epsilon estimate = %.4f (Mauchly test pValue=%.4f)\n', epsilon_gg, mauchly_table.pValue(1));

if length(list_model_tag) > 1
    % Between-subjects main effect (Greenhouse-Geissler correction does not affect non-repeated-measures)
    BS01_row = tbl('BS01', :);
    fprintf('MAIN EFFECT(BS01): F(%.0f, %.0f)=%.2f, pValueGG=%d, partEtaSq=%.2f\n',...
        BS01_row.DF(1),...
        tbl('Error', :).DF(1),...
        BS01_row.F(1),...
        BS01_row.pValueGG(1),...
        BS01_row.partEtaSq(1))
end

if length(cond) > 1
    % Within-subjects main effect (Greenhouse-Geissler correction applies)
    WS01_row = tbl('(Intercept):WS01', :);
    fprintf('MAIN EFFECT(WS01): F(%.2f, %.2f)=%.2f, pValueGG=%d, partEtaSq=%.2f\n',...
        epsilon_gg * WS01_row.DF(1),...
        epsilon_gg * tbl('Error(WS01)', :).DF(1),...
        WS01_row.F(1),...
        WS01_row.pValueGG(1),...
        WS01_row.partEtaSq(1))
end

if (length(list_model_tag) > 1) && (length(cond) > 1)
    % Interaction (Greenhouse-Geissler correction applies)
    INT_row = tbl('BS01:WS01', :);
    fprintf('INTERACTION(BS01:BS01): F(%.2f, %.2f)=%.2f, pValueGG=%d, partEtaSq=%.2f\n',...
        epsilon_gg * INT_row.DF(1),...
        epsilon_gg * tbl('Error(WS01)', :).DF(1),...
        INT_row.F(1),...
        INT_row.pValueGG(1),...
        INT_row.partEtaSq(1))
end

end
