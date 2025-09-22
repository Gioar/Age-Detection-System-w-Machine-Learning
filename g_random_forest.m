% Requisiti: Statistics and Machine Learning Toolbox

clear; clc; close all;
rng(42, 'twister');
tAll = tic;

%% ====================== OPZIONI ======================
opts.outputDir      = 'results_models_rf';
opts.showFigures    = true;
opts.saveFigures    = true;
opts.saveModel      = true;
opts.targetMetric   = 'accuracy';   % 'accuracy' | 'balanced'

% --- Gestione sbilanciamento ---
% 'none'           = massimizzazione accuracy (Prior empirico, niente Cost)
% 'prior-uniform'  = pesa tutte le classi uguali (bilancia metriche)
% 'mild-cost'      = costo morbido per classi deboli (consigliato se vuoi +recall su 41-50/51-60)
opts.Imbalance.mode         = 'mild-cost';          % <-- flag da modificare per impostare la modalità
opts.Imbalance.alpha        = 0.5;                  % intensità pesi (solo per cost/prior)
opts.Imbalance.boostClasses = ["x41_50","x51_60"];  % usato in mild-cost
opts.Imbalance.boostFactor  = 1.2;                  % 1.2–1.3 morbido

% --- Preprocessing feature ---
opts.LowVar.enabled = true;
opts.LowVar.tol     = 1e-3;          % filtro low variance

% --- Feature Selection (NO PCA, sconsigliata con Random Forest) ---
opts.FeatureSel.enabled  = true;
opts.FeatureSel.method   = 'chi2';    % 'mrmr' | 'chi2'
opts.FeatureSel.topN     = 8000;      % 6000 consigliato (8000 può dare OOM)
opts.FeatureSel.reportTop = 0;

% --- CV / Tuning ---
opts.CV.scheme        = 'holdout';
opts.CV.holdout       = 0.2;
opts.CV.useParallel   = true;        % da tenere OFF con RAM limitata
opts.CV.tuneSubsetMax = 20000;       % subset per tuning (stratificato)

% --- Griglie RF (tuning leggero) ---
opts.Tune.numTreesGrid        = [300 500 800];
opts.Tune.minLeafSizeGrid     = [5 10 20];
opts.Tune.numVarsSpecGrid     = {'sqrt','cuberoot'}; % numVars = sqrt(P) o P^(1/3)

%% ====================== CARICAMENTO DATI ======================
assert(isfile('features_train.mat') && isfile('features_test.mat'), ...
    'Mancano features_train.mat o features_test.mat');

S_tr = load('features_train.mat');
S_te = load('features_test.mat');

Xtr = S_tr.featuresTrainNorm;  ytr = S_tr.labelsTrain;
Xte = S_te.featuresTestNorm;   yte = S_te.labelsTest;

if ~iscategorical(ytr); ytr = categorical(ytr); end
if ~iscategorical(yte); yte = categorical(yte); end
classNames = categories(ytr);
ytr = categorical(ytr, classNames);
yte = categorical(yte, classNames);

fprintf('Train: %d campioni × %d feature, %d classi.\n', size(Xtr,1), size(Xtr,2), numel(classNames));
fprintf('Test : %d campioni × %d feature.\n', size(Xte,1), size(Xte,2));

%% ====================== LOW-VAR FILTER ======================
if opts.LowVar.enabled
    [Xtr, Xte, keptLowVar] = removeLowVarianceFeatures(Xtr, Xte, opts.LowVar.tol);
    fprintf('Low-variance filter tol=%.1e -> %d feature rimanenti.\n', opts.LowVar.tol, size(Xtr,2));
else
    keptLowVar = true(1,size(Xtr,2));
end

%% ====================== FEATURE SELECTION (NO PCA) ======================
fsInfo = struct('method', opts.FeatureSel.method, 'topN', 0, 'idx', [], 'scores', []);
if opts.FeatureSel.enabled
    tFS = tic;
    [idxFS, scoresFS] = selectFeatures(Xtr, ytr, opts.FeatureSel.method);
    topN = min(opts.FeatureSel.topN, numel(idxFS));
    idxTop = idxFS(1:topN);
    Xtr = Xtr(:, idxTop);
    Xte = Xte(:, idxTop);
    fsInfo.method = opts.FeatureSel.method;
    fsInfo.topN   = topN;
    fsInfo.idx    = idxTop;
    fsInfo.scores = scoresFS(1:topN);
    fprintf('Feature selection [%s]: topN=%d (%.2f s)\n', opts.FeatureSel.method, topN, toc(tFS));
else
    idxTop = find(keptLowVar);
end

%% ====================== COST / PRIOR PER SBILANCIAMENTO ======================
[priorSpec, Cost] = buildImbalancePolicy(ytr, classNames, opts.Imbalance);

%% ====================== TUNING RF (hold-out + subset) ======================
tTune = tic;
best = tuneRF_Holdout(Xtr, ytr, classNames, priorSpec, Cost, opts);
tTuneSec = toc(tTune);

fprintf('\n=== MIGLIOR CONFIGURAZIONE (hold-out) ===\n');
disp(best);

%% ====================== TRAINING FINALE RF ======================
tTrain = tic;
model = trainRF_Final(Xtr, ytr, classNames, best, priorSpec, Cost);
trainTime = toc(tTrain);
fprintf('Training RF completato in %.3f s\n', trainTime);

%% ====================== PREDIZIONE SU TEST ======================
tPred = tic;
[ypred, scores] = predict(model.clf, Xte);
ypred = categorical(ypred, classNames);
predTime = toc(tPred);
fprintf('Predizione su test in %.3f s\n', predTime);

%% ====================== VALUTAZIONE ======================
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
modelFile  = fullfile(opts.outputDir, sprintf('age_model_rf_%s.mat', timestamp));
reportFile = fullfile(opts.outputDir, sprintf('report_rf_%s.txt', timestamp));
idxFile    = fullfile(opts.outputDir, sprintf('selected_features_rf_%s.mat', timestamp));

if opts.saveModel
    save(modelFile, 'model', 'opts', 'classNames', 'best', 'fsInfo', 'keptLowVar', '-v7.3');
    save(idxFile, 'fsInfo', 'keptLowVar', '-v7.3');
    fprintf('Modello RF salvato in: %s\n', modelFile);
end

fid = fopen(reportFile, 'w');
fprintf(fid, 'Random Forest (Bagging)\n');
fprintf(fid, 'Train: %d x %d | Test: %d x %d | Classi: %d\n', size(Xtr,1), size(Xtr,2), size(Xte,1), size(Xte,2), numel(classNames));
fprintf(fid, 'FS: method=%s | topN=%d\n', fsInfo.method, fsInfo.topN);
fprintf(fid, 'Best RF: NumTrees=%d | MinLeaf=%d | NumVarsToSample=%d | target=%s\n', ...
    best.numTrees, best.minLeaf, best.numVars, opts.targetMetric);
fprintf(fid, 'Accuracy=%.4f | Balanced=%.4f | MacroF1=%.4f | WeightedF1=%.4f\n', ...
    metrics.accuracy, metrics.balancedAccuracy, metrics.macroF1, metrics.weightedF1);
fprintf(fid, 'Tempo tuning=%.3f s | Training=%.3f s | Pred=%.3f s | Totale=%.3f s\n', ...
    tTuneSec, trainTime, predTime, toc(tAll));
fclose(fid);
fprintf('Report RF salvato in: %s\n', reportFile);

fprintf('\nTempo totale script: %.3f s\n', toc(tAll));

%% ====================== FUNZIONI AUSILIARIE ======================

function [Xtr2, Xte2, keptIdx] = removeLowVarianceFeatures(Xtr, Xte, tol)
    v = var(Xtr, 0, 1);
    keptIdx = v > tol;
    Xtr2 = Xtr(:, keptIdx);
    Xte2 = Xte(:, keptIdx);
end

function [idx, scores] = selectFeatures(X, y, method)
    switch lower(method)
        case 'chi2'
            % Chi^2 richiede feature non-negative: shift per lo scoring
            if min(X(:)) < 0
                mn = min(X, [], 1);
                Xs = bsxfun(@minus, X, mn);
            else
                Xs = X;
            end
            [idxRaw, scoresRaw] = fscchi2(Xs, y);
            [scores, ord] = sort(scoresRaw, 'descend');
            idx = idxRaw(ord);

        case 'mrmr'
            idx = fscmrmr(X, y);
            % MRMR non restituisce punteggi; è necessario construire una scala decrescente fittizia
            scores = linspace(1, 0, numel(idx))';

        otherwise
            error('Metodo di feature selection non supportato: %s', method);
    end
end

function [priorSpec, Cost] = buildImbalancePolicy(ytr, classNames, imb)
    switch lower(imb.mode)
        case 'none'
            priorSpec = 'empirical'; Cost = [];

        case 'prior-uniform'
            priorSpec = 'uniform';   Cost = [];

        case 'mild-cost'
            priorSpec = 'empirical';
            K = numel(classNames); Cost = ones(K) - eye(K);
            counts = countcats(ytr); maxc = max(counts);
            baseW = (maxc ./ max(counts,1)).^imb.alpha; % pesi inversi alla frequenza
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

function best = tuneRF_Holdout(X, y, classNames, priorSpec, Cost, opts)
    % Hold-out stratificato
    cvp = cvpartition(y, 'HoldOut', opts.CV.holdout);
    Xtr = X(training(cvp), :);  ytr = y(training(cvp));
    Xva = X(test(cvp), :);      yva = y(test(cvp));

    % Subset per tuning
    if size(Xtr,1) > opts.CV.tuneSubsetMax
        cvsub = cvpartition(ytr, 'HoldOut', 1 - opts.CV.tuneSubsetMax/numel(ytr));
        Xtr = Xtr(test(cvsub), :);
        ytr = ytr(test(cvsub));
    end

    combos = {};
    P = size(X,2);
    for nt = opts.Tune.numTreesGrid
        for mleaf = opts.Tune.minLeafSizeGrid
            for spec = opts.Tune.numVarsSpecGrid
                combos{end+1,1} = struct( ...
                    'numTrees', nt, ...
                    'minLeaf',  mleaf, ...
                    'numVars',  numVarsFromSpec(P, spec{1}) );
            end
        end
    end
    results = struct('numTrees',{},'minLeaf',{},'numVars',{},'metric',{});
    % Parallel OFF per evitare OOM
    for i=1:numel(combos)
        cfg = combos{i};
        results(i).numTrees = cfg.numTrees;
        results(i).minLeaf  = cfg.minLeaf;
        results(i).numVars  = cfg.numVars;
        results(i).metric   = evalRFConfig(Xtr, ytr, Xva, yva, classNames, priorSpec, Cost, opts, cfg);
    end

    [~, idx] = max([results.metric]);
    best = results(idx);
end

function k = numVarsFromSpec(P, spec)
    switch lower(spec)
        case 'sqrt'
            k = max(1, floor(sqrt(P)));
        case 'cuberoot'
            k = max(1, floor(P^(1/3)));
        otherwise
            k = max(1, floor(sqrt(P)));
    end
end

function metricVal = evalRFConfig(Xtr, ytr, Xva, yva, classNames, priorSpec, Cost, opts, cfg)
    tTree = templateTree( ...
        'MinLeafSize',          cfg.minLeaf, ...
        'NumVariablesToSample', cfg.numVars, ...
        'Surrogate',            'off');

    args = {'Method','Bag', 'NumLearningCycles', cfg.numTrees, ...
            'Learners', tTree, 'ClassNames', classNames, ...
            'Prior', priorSpec, 'ObservationsIn','rows'};
    if ~isempty(Cost), args = [args, {'Cost', Cost}]; end

    M = fitcensemble(Xtr, ytr, args{:});

    yhat = predict(M, Xva);
    yhat = categorical(yhat, classNames);

    [metrics, ~] = computeMetrics(yva, yhat, classNames);
    switch lower(opts.targetMetric)
        case 'accuracy',  metricVal = metrics.accuracy;
        case 'balanced',  metricVal = metrics.balancedAccuracy;
        otherwise,        metricVal = metrics.accuracy;
    end
end

function model = trainRF_Final(X, y, classNames, best, priorSpec, Cost)
    tTree = templateTree( ...
        'MinLeafSize',          best.minLeaf, ...
        'NumVariablesToSample', best.numVars, ...
        'Surrogate',            'off');

    args = {'Method','Bag', 'NumLearningCycles', best.numTrees, ...
            'Learners', tTree, 'ClassNames', classNames, ...
            'Prior', priorSpec, 'ObservationsIn','rows'};
    if ~isempty(Cost), args = [args, {'Cost', Cost}]; end

    model.clf = fitcensemble(X, y, args{:});
    model.type = 'rf_bag';
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
