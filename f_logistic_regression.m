% Plugin richiesti: Statistics and Machine Learning Toolbox
% Nota: tenere i flag relativi al parallelismo DISATTIVATI se si riscontrano limiti di RAM.

clear; clc; close all;
rng(42, 'twister');
tAll = tic;

%% ====================== PROFILO ======================
% Scelta del profilo:
%   'max-acc'  = massimizzazione accuracy (prior empirico, nessun costo)
%   'balanced' = mild-cost su x41_50 e x51_60 (migliora recall/balanced)
profile = 'max-acc';   % <-- flag di impostazione del profilo

%% ====================== OPZIONI GENERALI ======================
opts.outputDir      = 'results_models';
opts.showFigures    = true;
opts.saveFigures    = true;
opts.saveModel      = true;
opts.targetMetric   = 'accuracy';   % 'accuracy' | 'balanced'

% --- CV / Tuning ---
opts.CV.scheme        = 'holdout';
opts.CV.holdout       = 0.2;        % 20% validazione
opts.CV.useParallel   = true;      % esecuzione parallela su più core
opts.CV.tuneSubsetMax = 15000;      % subset per tuning (riduce tempi/RAM)

% --- Griglie di tuning (micro-tuning attorno ai tuoi best attuali) ---
opts.Tune.pcaGrid     = [384 448];
opts.Tune.lambdaGrid  = [1e-3 3e-3 7e-3];
opts.Tune.codingGrid  = {'onevsone','onevsall'};
opts.Tune.regGrid     = {'ridge','lasso'};     

% --- Logistic Learner ---
opts.Logistic.solver         = 'lbfgs';  % robusto; per lasso proviamo 'sparsa' e fallback a 'lbfgs'
opts.Logistic.iterLimit      = 300;
opts.Logistic.betaTol        = 1e-6;
opts.Logistic.gradTol        = 1e-6;

% --- PCA ---
opts.PCA.method    = 'incremental';  % 'incremental' (consigliato) | 'svd'
opts.PCA.blockSize = 3000;           % aumentare per velocizzare il processo, aumenta il consumo di RAM
opts.PCA.useSingle = false;          % esecuzione dei calcoli della PCA su single core

%% ====================== CARICAMENTO DATI ======================
assert(isfile('features_train_bal.mat') && isfile('features_test_bal.mat'), ...
    'Mancano features_train.mat o features_test.mat');

S_tr = load('features_train_bal.mat');
S_te = load('features_test_bal.mat');

Xtr = S_tr.featuresTrainNorm;  ytr = S_tr.labelsTrain;
Xte = S_te.featuresTestNorm;   yte = S_te.labelsTest;

if ~iscategorical(ytr); ytr = categorical(ytr); end
if ~iscategorical(yte); yte = categorical(yte); end
classNames = categories(ytr);
ytr = categorical(ytr, classNames);
yte = categorical(yte, classNames);

fprintf('Train: %d campioni × %d feature, %d classi.\n', size(Xtr,1), size(Xtr,2), numel(classNames));
fprintf('Test : %d campioni × %d feature.\n', size(Xte,1), size(Xte,2));

%% ====================== PROFILO SBILANCIAMENTO ======================
opts = configureProfile(profile, opts, classNames);
[priorSpec, Cost] = buildImbalancePolicy(ytr, classNames, opts.Imbalance);

%% ====================== TUNING: PCA × λ × reg × CODING ======================
tTune = tic;
best = tuneLogisticPCA(Xtr, ytr, classNames, priorSpec, Cost, opts);
tTuneSec = toc(tTune);

fprintf('\n=== MIGLIOR CONFIGURAZIONE (hold-out) ===\n');
fprintf('nComp=%d | reg=%s | coding=%s | lambda=%.2e | metric(%s)=%.4f | cumExpl~%.1f%%\n', ...
    best.nComp, best.reg, best.coding, best.lambda, opts.targetMetric, best.metric, best.cumExplainedPct);

%% ====================== TRAINING FINALE CON BEST PCA ======================
tTrain = tic;
[pcaParam, Xtr_pca, cumExplFinal] = fitPCA(Xtr, best.nComp, opts);
model = trainLogisticFinalPCA(Xtr_pca, ytr, classNames, best, priorSpec, Cost, opts);
trainTime = toc(tTrain);
fprintf('Training completato in %.3f s\n', trainTime);

%% ====================== PREDIZIONE SU TEST ======================
tPred = tic;
Xte_pca = applyPCA(Xte, pcaParam, opts); % trasformazione a blocchi (no OOM)
[ypred, scores] = predict(model.clf, Xte_pca);
ypred = categorical(ypred, classNames);
predTime = toc(tPred);
fprintf('Predizione su test in %.3f s\n', predTime);

%% ====================== PARAMETRI DI VALUTAZIONE ======================
[metrics, C] = computeMetrics(yte, ypred, classNames);
fprintf('\n=== RISULTATI ===\n');
fprintf('Accuracy            : %.4f\n', metrics.accuracy);
fprintf('Balanced Accuracy   : %.4f\n', metrics.balancedAccuracy);
fprintf('Macro Precision     : %.4f\n', metrics.macroPrecision);
fprintf('Macro Recall        : %.4f\n', metrics.macroRecall);
fprintf('Macro F1            : %.4f\n', metrics.macroF1);
fprintf('Weighted F1         : %.4f\n', metrics.weightedF1);
disp('F1 per classe:');
disp(array2table(metrics.F1_per_class(:)', 'VariableNames', matlab.lang.makeValidName(classNames)));

plotAndMaybeSaveConfusions(yte, ypred, classNames, opts);

%% ====================== SALVATAGGIO ======================
if ~exist(opts.outputDir, 'dir'); mkdir(opts.outputDir); end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
modelFile  = fullfile(opts.outputDir, sprintf('age_model_logistic_PCA_%s.mat', timestamp));
reportFile = fullfile(opts.outputDir, sprintf('report_logistic_PCA_%s.txt', timestamp));

if opts.saveModel
    save(modelFile, 'model', 'opts', 'classNames', 'best', 'pcaParam', '-v7.3');
    fprintf('Modello salvato in: %s\n', modelFile);
end

fid = fopen(reportFile, 'w');
fprintf(fid, 'LOGISTIC (ECOC) + PCA (incremental) | profilo: %s\n', profile);
fprintf(fid, 'Train: %d x %d | Test: %d x %d | Classi: %d\n', size(Xtr,1), size(Xtr,2), size(Xte,1), size(Xte,2), numel(classNames));
fprintf(fid, 'Best: nComp=%d | reg=%s | coding=%s | lambda=%.2e | metric(%s)=%.4f | cumExpl~%.1f%%\n', ...
    best.nComp, best.reg, best.coding, best.lambda, opts.targetMetric, best.metric, best.cumExplainedPct);
fprintf(fid, 'Accuracy=%.4f | Balanced=%.4f | MacroF1=%.4f | WeightedF1=%.4f\n', ...
    metrics.accuracy, metrics.balancedAccuracy, metrics.macroF1, metrics.weightedF1);
fprintf(fid, 'Tempo tuning=%.3f s | Training=%.3f s | Pred=%.3f s | Totale=%.3f s\n', ...
    tTuneSec, trainTime, predTime, toc(tAll));
fclose(fid);
fprintf('Report salvato in: %s\n', reportFile);

fprintf('\nTempo totale script: %.3f s\n', toc(tAll));

%% ====================== FUNZIONI AUSILIARIE ======================

function opts = configureProfile(profile, opts, classNames)
% Configurazione della politica di sbilanciamento, in base al profilo scelto.
    switch lower(profile)
        case 'max-acc'
            opts.Imbalance.mode  = 'none';        % prior empirico, nessun costo
            opts.Imbalance.alpha = 0.5;
            % nessun boost

        case 'balanced'
            opts.Imbalance.mode  = 'mild-cost';   % costo morbido su classi deboli
            opts.Imbalance.alpha = 0.5;
            opts.Imbalance.boostClasses = ["x41_50","x51_60"];
            opts.Imbalance.boostFactor  = 1.3;    % 1.2–1.5

        otherwise
            error('Profilo non riconosciuto: %s', profile);
    end
end

function [priorSpec, Cost] = buildImbalancePolicy(ytr, classNames, imb)
% Creazione della prior/cost matrix in base alla politica scelta.
    switch lower(imb.mode)
        case 'none'
            priorSpec = 'empirical'; Cost = [];

        case 'prior-uniform'
            priorSpec = 'uniform';   Cost = [];

        case 'cost-sensitive'
            priorSpec = 'empirical';
            counts = countcats(ytr); maxc = max(counts);
            w = (maxc ./ max(counts,1)).^imb.alpha;
            K = numel(classNames); Cost = ones(K) - eye(K);
            for c=1:K, Cost(c,:) = Cost(c,:)*w(c); end

        case 'mild-cost'
            priorSpec = 'empirical';
            counts = countcats(ytr); maxc = max(counts);
            baseW = (maxc ./ max(counts,1)).^imb.alpha;
            K = numel(classNames); Cost = ones(K) - eye(K);
            boost = ones(K,1);
            if isfield(imb,'boostClasses') && ~isempty(imb.boostClasses)
                for j = 1:numel(imb.boostClasses)
                    idx = find(classNames == imb.boostClasses(j));
                    if ~isempty(idx), boost(idx) = imb.boostFactor; end
                end
            end
            w = baseW .* boost;
            for c=1:K, Cost(c,:) = Cost(c,:)*w(c); end

        otherwise
            error('Imbalance mode non riconosciuto: %s', imb.mode);
    end
end

function best = tuneLogisticPCA(X, y, classNames, priorSpec, Cost, opts)
% Grid search con hold-out e subset: PCA (subset) → Logistic → valutazione

    % Hold-out stratificato
    cvp = cvpartition(y, 'HoldOut', opts.CV.holdout);
    Xtr = X(training(cvp), :);  ytr = y(training(cvp));
    Xva = X(test(cvp), :);      yva = y(test(cvp));

    % Subset di tuning (stratificato)
    if size(Xtr,1) > opts.CV.tuneSubsetMax
        cvsub = cvpartition(ytr, 'HoldOut', 1 - opts.CV.tuneSubsetMax/numel(ytr));
        Xtr = Xtr(test(cvsub), :);
        ytr = ytr(test(cvsub));
    end

    [priorSub, CostSub] = adaptCostToSubset(ytr, classNames, priorSpec, Cost);

    % Costruzione combinazioni
    combos = {};
    for nComp = opts.Tune.pcaGrid
        for reg = opts.Tune.regGrid
            for lam = opts.Tune.lambdaGrid
                for c = 1:numel(opts.Tune.codingGrid)
                    combos{end+1,1} = struct( ...
                        'nComp', nComp, ...
                        'reg',   reg{1}, ...
                        'lambda',lam, ...
                        'coding',opts.Tune.codingGrid{c});
                end
            end
        end
    end

    results = struct('nComp',{},'reg',{},'lambda',{},'coding',{},'metric',{},'cumExplainedPct',{});
    % Parallel OFF per evitare OOM
    for i = 1:numel(combos)
        cfg = combos{i};
        [mval, cumExplPct] = evalOneConfigPCA(Xtr, ytr, Xva, yva, classNames, priorSub, CostSub, opts, cfg);
        results(i).nComp = cfg.nComp;
        results(i).reg   = cfg.reg;
        results(i).lambda= cfg.lambda;
        results(i).coding= cfg.coding;
        results(i).metric= mval;
        results(i).cumExplainedPct = cumExplPct;
    end

    [~, idx] = max([results.metric]);
    best = results(idx);
end

function [metricVal, cumExplPct] = evalOneConfigPCA(Xtr, ytr, Xva, yva, classNames, priorSpec, Cost, opts, cfg)
% Fit PCA (subset), proietta train/val, addestra logistic e valuta.

    [pcaParam, Xtr_pca, cumExplPct] = fitPCA(Xtr, cfg.nComp, opts);
    Xva_pca = applyPCA(Xva, pcaParam, opts);

    % Template logistic con ridge/lasso (solver adattivo)
    solverToUse = opts.Logistic.solver;
    if strcmpi(cfg.reg,'lasso')
        % 'sparsa' spesso è migliore per L1; se non disponibile, fallback a 'lbfgs'
        try
            solverToUse = 'sparsa';
            tLinear = templateLinear('Learner','logistic','Regularization','lasso','Lambda',cfg.lambda, ...
                'Solver',solverToUse, 'BetaTolerance',opts.Logistic.betaTol, ...
                'GradientTolerance',opts.Logistic.gradTol, 'IterationLimit',opts.Logistic.iterLimit,'Verbose',0);
        catch
            solverToUse = 'lbfgs';
            tLinear = templateLinear('Learner','logistic','Regularization','lasso','Lambda',cfg.lambda, ...
                'Solver',solverToUse, 'BetaTolerance',opts.Logistic.betaTol, ...
                'GradientTolerance',opts.Logistic.gradTol, 'IterationLimit',opts.Logistic.iterLimit,'Verbose',0);
        end
    else
        tLinear = templateLinear('Learner','logistic','Regularization','ridge','Lambda',cfg.lambda, ...
            'Solver',solverToUse, 'BetaTolerance',opts.Logistic.betaTol, ...
            'GradientTolerance',opts.Logistic.gradTol, 'IterationLimit',opts.Logistic.iterLimit,'Verbose',0);
    end

    args = {'Learners', tLinear, 'Coding', cfg.coding, 'ClassNames', classNames, ...
            'Prior', priorSpec, 'ObservationsIn','rows', 'Verbose', 0};
    if ~isempty(Cost), args = [args, {'Cost', Cost}]; end

    M = fitcecoc(Xtr_pca, ytr, args{:});
    yhat = predict(M, Xva_pca);
    yhat = categorical(yhat, classNames);

    [metrics, ~] = computeMetrics(yva, yhat, classNames);
    switch lower(opts.targetMetric)
        case 'accuracy',  metricVal = metrics.accuracy;
        case 'balanced',  metricVal = metrics.balancedAccuracy;
        otherwise,        metricVal = metrics.accuracy;
    end
end

%% ===== PCA memory-safe (incremental compatibile) =====
function [pcaParam, X_pca, cumExplPct] = fitPCA(X, nComp, opts)
% PCA memory-safe:
% - 'incremental': incrementalPCA a blocchi (no OOM, no accesso a proprietà interne)
% - 'svd': SVD classica con centering manuale e senza TSQUARED (no warning)

    method = 'incremental';
    if isfield(opts,'PCA') && isfield(opts.PCA,'method')
        method = lower(opts.PCA.method);
    end

    switch method
        case 'incremental'
            [pcaParam, X_pca, cumExplPct] = fitPCA_incremental(X, nComp, opts);
        case 'svd'
            [pcaParam, X_pca, cumExplPct] = fitPCA_svd_notsq(X, nComp, opts);
        otherwise
            error('opts.PCA.method non riconosciuto: %s', method);
    end
end

function [p, X_pca, cumExplPct] = fitPCA_incremental(X, nComp, opts)
% PCA incrementale a blocchi usando incrementalPCA, senza leggere proprietà interne.

    if exist('incrementalPCA','class') ~= 8
        error('incrementalPCA non disponibile. Imposta opts.PCA.method = ''svd''.');
    end

    bs = 2000;
    if isfield(opts,'PCA') && isfield(opts.PCA,'blockSize'), bs = opts.PCA.blockSize; end
    useSingle = isfield(opts,'PCA') && isfield(opts.PCA,'useSingle') && opts.PCA.useSingle;

    N = size(X,1);
    ip = incrementalPCA('NumComponents', nComp);

    % 1) Fit incrementale
    for i = 1:bs:N
        idx = i:min(i+bs-1, N);
        Xb = X(idx,:);
        if useSingle, Xb = single(Xb); end
        didFit = false;
        try, ip = partialFit(ip, Xb); didFit = true; catch, end
        if ~didFit, try, ip = update(ip, Xb); didFit = true; catch, end, end
        if ~didFit, ip = fit(ip, Xb); end
    end

    % 2) Transformazione a blocchi (TRAIN)
    X_pca = zeros(N, nComp, 'double');
    for i = 1:bs:N
        idx = i:min(i+bs-1, N);
        Xb = X(idx,:);
        if useSingle, Xb = single(Xb); end
        Tb = transform(ip, Xb);
        X_pca(idx,:) = double(Tb);
    end

    % 3) Parametri
    p.type  = 'incremental';
    p.ip    = ip;
    p.nComp = nComp;

    % 4) Varianza spiegata cumulata (se disponibile)
    cumExplPct = NaN;
    if isprop(ip, 'ExplainedVariance') && isprop(ip, 'TotalVariance')
        ev = double(ip.ExplainedVariance); tv = double(ip.TotalVariance);
        if ~isempty(ev) && tv>0, cumExplPct = 100*sum(ev)/tv; end
    elseif isprop(ip, 'ExplainedVarianceRatio')
        evr = double(ip.ExplainedVarianceRatio); if ~isempty(evr), cumExplPct = 100*sum(evr); end
    else
        compVar  = sum(var(X_pca, 0, 1));
        totalVar = sum(var(X,      0, 1));
        if totalVar > 0, cumExplPct = 100 * compVar / totalVar; end
    end
end

function [p, X_pca, cumExplPct] = fitPCA_svd_notsq(X, nComp, opts)
% PCA SVD classica (memoria più pesante): centering manuale e niente TSQUARED.

    useSingle = isfield(opts,'PCA') && isfield(opts.PCA,'useSingle') && opts.PCA.useSingle;
    if useSingle, Xs = single(X); else, Xs = X; end

    mu = mean(Xs, 1);
    Xc = bsxfun(@minus, Xs, mu);

    [coeff, score, latent] = pca(Xc, 'Centered', false, 'NumComponents', nComp, 'Algorithm','svd');

    X_pca    = double(score);
    p.type   = 'svd';
    p.mu     = double(mu);
    p.coeff  = double(coeff);
    p.nComp  = nComp;

    totalVar    = sum(var(double(Xc), 0, 1));
    cumExplPct  = 100 * sum(double(latent)) / max(totalVar, eps);

    clear Xs Xc;
end

function X_out = applyPCA(X, p, opts)
% Applicazione PCA:
% - se p.ip esiste (incrementalPCA), usa transform a blocchi;
% - altrimenti, fallback (SVD classica) con mu/coeff.

    if isfield(p, 'ip') && ~isempty(p.ip)
        bs = size(X,1);
        if nargin >= 3 && isfield(opts,'PCA') && isfield(opts.PCA,'blockSize')
            bs = opts.PCA.blockSize;
        end
        useSingle = nargin >= 3 && isfield(opts,'PCA') && isfield(opts.PCA,'useSingle') && opts.PCA.useSingle;

        N = size(X,1); K = p.nComp;
        X_out = zeros(N, K, 'double');
        for i = 1:bs:N
            idx = i:min(i+bs-1, N);
            Xb = X(idx,:);
            if useSingle, Xb = single(Xb); end
            Tb = transform(p.ip, Xb);
            X_out(idx,:) = double(Tb);
        end
        return;
    end

    X_out = (X - p.mu) * p.coeff; % fallback SVD
end

function [prior2, Cost2] = adaptCostToSubset(ytr, classNames, priorSpec, Cost)
% Allineamento dell'eventuale cost matrix alla distribuzione del subset
    prior2 = priorSpec;
    if isempty(Cost)
        Cost2 = [];
    else
        counts = countcats(ytr);
        maxc = max(counts);
        w = (maxc ./ max(counts,1)).^0.5;
        K = numel(classNames);
        Cost2 = ones(K) - eye(K);
        for c=1:K, Cost2(c,:) = Cost2(c,:)*w(c); end
    end
end

function model = trainLogisticFinalPCA(X_pca, y, classNames, best, priorSpec, Cost, opts)
% Refit finale con i migliori iperparametri (ridge/lasso + coding)

    solverToUse = opts.Logistic.solver;
    if strcmpi(best.reg,'lasso')
        try
            solverToUse = 'sparsa';
            tLinear = templateLinear('Learner','logistic','Regularization','lasso','Lambda',best.lambda, ...
                'Solver',solverToUse, 'BetaTolerance',opts.Logistic.betaTol, ...
                'GradientTolerance',opts.Logistic.gradTol, 'IterationLimit',opts.Logistic.iterLimit,'Verbose',0);
        catch
            solverToUse = 'lbfgs';
            tLinear = templateLinear('Learner','logistic','Regularization','lasso','Lambda',best.lambda, ...
                'Solver',solverToUse, 'BetaTolerance',opts.Logistic.betaTol, ...
                'GradientTolerance',opts.Logistic.gradTol, 'IterationLimit',opts.Logistic.iterLimit,'Verbose',0);
        end
    else
        tLinear = templateLinear('Learner','logistic','Regularization','ridge','Lambda',best.lambda, ...
            'Solver',solverToUse, 'BetaTolerance',opts.Logistic.betaTol, ...
            'GradientTolerance',opts.Logistic.gradTol, 'IterationLimit',opts.Logistic.iterLimit,'Verbose',0);
    end

    args = {'Learners', tLinear, 'Coding', best.coding, 'ClassNames', classNames, ...
            'Prior', priorSpec, 'ObservationsIn','rows', 'Verbose', 0};
    if ~isempty(Cost), args = [args, {'Cost', Cost}]; end

    model.clf = fitcecoc(X_pca, y, args{:});
    model.type = 'logistic_ecoc_pca';
end

function [metrics, C] = computeMetrics(ytrue, ypred, classNames)
    [C, ~] = confusionmat(ytrue, ypred, 'Order', classNames);
    total = sum(C(:));
    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;

    precision = TP ./ max(TP + FP, eps);
    recall    = TP ./ max(TP + FN, eps);
    F1        = 2 * (precision .* recall) ./ max(precision + recall, eps);

    accuracy          = sum(TP) / total;
    balancedAccuracy  = mean(recall);
    macroPrecision    = mean(precision);
    macroRecall       = mean(recall);
    macroF1           = mean(F1);

    support = sum(C,2);
    w = support / sum(support);
    weightedF1 = sum(w .* F1);

    metrics = struct();
    metrics.accuracy         = accuracy;
    metrics.balancedAccuracy = balancedAccuracy;
    metrics.macroPrecision   = macroPrecision;
    metrics.macroRecall      = macroRecall;
    metrics.macroF1          = macroF1;
    metrics.weightedF1       = weightedF1;
    metrics.F1_per_class     = F1(:);
    metrics.confusionMatrix  = C;
end

function plotAndMaybeSaveConfusions(ytrue, ypred, classNames, opts)
    if ~exist(opts.outputDir, 'dir'); mkdir(opts.outputDir); end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    [C, ~] = confusionmat(ytrue, ypred, 'Order', classNames);

    usedChart = false;
    if exist('confusionchart','file') == 2
        try
            fig1 = figure('Name','Confusion Matrix - Counts','Color','w');
            ch = confusionchart(C, classNames);
            title('Confusion Matrix (Counts)');
            try
                ch.RowSummary    = 'row-normalized';
                ch.ColumnSummary = 'column-normalized';
            catch, end
            saveFigure(fig1, fullfile(opts.outputDir, ...
                sprintf('confusion_counts_%s.png', timestamp)), opts);
            if ~opts.showFigures, close(fig1); end
            usedChart = true;
        catch, usedChart = false;
        end
    end

    if ~usedChart
        fig1 = figure('Name','Confusion Matrix - Counts (fallback)','Color','w');
        imagesc(C); colormap(parula); colorbar; axis image tight;
        xticks(1:numel(classNames)); xticklabels(classNames); xtickangle(45);
        yticks(1:numel(classNames)); yticklabels(classNames);
        title('Confusion Matrix (Counts)');
        for i=1:size(C,1)
            for j=1:size(C,2)
                text(j, i, sprintf('%d', C(i,j)), 'HorizontalAlignment','center', 'Color', 'k', 'FontSize', 9);
            end
        end
        saveFigure(fig1, fullfile(opts.outputDir, ...
            sprintf('confusion_counts_fallback_%s.png', timestamp)), opts);
        if ~opts.showFigures, close(fig1); end
    end

    rowSums = sum(C, 2); rowSums(rowSums == 0) = 1;
    Cnorm = bsxfun(@rdivide, C, rowSums);
    fig2 = figure('Name','Confusion Matrix - Row Normalized','Color','w');
    imagesc(Cnorm, [0 1]); colormap(parula); colorbar; axis image tight;
    xticks(1:numel(classNames)); xticklabels(classNames); xtickangle(45);
    yticks(1:numel(classNames)); yticklabels(classNames);
    title('Confusion Matrix (Row-normalized)');
    for i=1:size(Cnorm,1)
        for j=1:size(Cnorm,2)
            text(j, i, sprintf('%.2f', Cnorm(i,j)), 'HorizontalAlignment','center', 'Color', 'k', 'FontSize', 9);
        end
    end
    saveFigure(fig2, fullfile(opts.outputDir, ...
        sprintf('confusion_rowNorm_%s.png', timestamp)), opts);
    if ~opts.showFigures, close(fig2); end
end

function saveFigure(fig, filename, opts)
    if ~opts.saveFigures, return; end
    [outDir,~,~] = fileparts(filename);
    if ~exist(outDir,'dir'); mkdir(outDir); end
    if exist('exportgraphics','file') == 2
        exportgraphics(fig, filename, 'Resolution', 300);
    else
        print(fig, filename, '-dpng', '-r300');
    end
end
