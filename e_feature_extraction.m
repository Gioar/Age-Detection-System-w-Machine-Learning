%% Feature Extraction con HOG + LBP + Normalizzazione
clc; clear; tic;

%% === PARAMETRI ===
cellSizeHOG = [8 8];       % HOG cell size (più è grande la griglia, minore saranno le features estratte)
lbpRadius = 1;             % LBP radius
lbpNeighbors = 8;          % LBP neighbors

% Avvio del pool parallelo, se disponibile
if isempty(gcp('nocreate'))
    parpool('Processes', 2);  % Numero workers
end

%% === FUNZIONE DI ESTRAZIONE GENERICA ===
function [features, labels, minFeatHOG, maxFeatHOG, minFeatLBP, maxFeatLBP] = extractFeatures(imds, cellSizeHOG, lbpRadius, lbpNeighbors, minFeatHOG, maxFeatHOG, minFeatLBP, maxFeatLBP, isTrain)
    numImages = numel(imds.Files);
    labels = imds.Labels;
    fprintf('%s: %d immagini in %d classi.\n', ternary(isTrain, 'Train', 'Test'), numImages, numel(unique(labels)));

    % Preallocazione
    sampleImg = imread(imds.Files{1});
    if size(sampleImg,3) == 3; sampleImg = rgb2gray(sampleImg); end
    hogLength = length(extractHOGFeatures(sampleImg, 'CellSize', cellSizeHOG));
    lbpLength = length(extractLBPFeatures(sampleImg, 'Radius', lbpRadius, 'NumNeighbors', lbpNeighbors));
    features = zeros(numImages, hogLength + lbpLength);

    % Contatore per monitoraggio del progresso di estrazione su terminale
    progress = parallel.pool.DataQueue;
    afterEach(progress, @(x) fprintf('%s HOG+LBP: %d/%d\n', ternary(isTrain, 'Train', 'Test'), x, numImages));

    % Estrazione HOG + LBP
    parfor i = 1:numImages
        try
            img = imread(imds.Files{i});
            if size(img,3) == 3; img = rgb2gray(img); end
            hogFeat = extractHOGFeatures(img, 'CellSize', cellSizeHOG);
            lbpFeat = extractLBPFeatures(img, 'Radius', lbpRadius, 'NumNeighbors', lbpNeighbors);
            features(i,:) = [hogFeat, lbpFeat];
        catch err
            warning('Errore su immagine %d: %s. Saltata.', i, err.message);
            features(i,:) = zeros(1, hogLength + lbpLength);
        end
        if mod(i,500)==0; send(progress, i); end  % Invia progresso
    end

    % Normalizzazione Min-Max
    if isTrain
        minFeatHOG = min(features(:,1:hogLength), [], 1);
        maxFeatHOG = max(features(:,1:hogLength), [], 1);
        minFeatLBP = min(features(:,hogLength+1:end), [], 1);
        maxFeatLBP = max(features(:,hogLength+1:end), [], 1);
    end
    featuresNorm = features;
    featuresNorm(:,1:hogLength) = (features(:,1:hogLength) - minFeatHOG) ./ (maxFeatHOG - minFeatHOG + eps);
    featuresNorm(:,hogLength+1:end) = (features(:,hogLength+1:end) - minFeatLBP) ./ (maxFeatLBP - minFeatLBP + eps);
    features = featuresNorm;
end

%% === 1. FEATURE EXTRACTION TRAIN ===
trainFolder = 'final_dataset_face\train_aug';
if ~exist(trainFolder, 'dir')
    error('Cartella %s non trovata. Esegui il preprocessing per il train.', trainFolder);
end
imdsTrain = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[featuresTrainNorm, labelsTrain, minFeatHOG, maxFeatHOG, minFeatLBP, maxFeatLBP] = extractFeatures(imdsTrain, cellSizeHOG, lbpRadius, lbpNeighbors, [], [], [], [], true);

% Salvataggio train
save('features_train.mat', 'featuresTrainNorm', 'labelsTrain', 'minFeatHOG', 'maxFeatHOG', 'minFeatLBP', 'maxFeatLBP', '-v7.3');

%% === 2. FEATURE EXTRACTION TEST ===
testFolder = 'final_dataset_face\test_aug';
if ~exist(testFolder, 'dir')
    error('Cartella %s non trovata. Esegui il preprocessing per il test.', testFolder);
end
imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[featuresTestNorm, labelsTest, ~, ~, ~, ~] = extractFeatures(imdsTest, cellSizeHOG, lbpRadius, lbpNeighbors, minFeatHOG, maxFeatHOG, minFeatLBP, maxFeatLBP, false);

% Salvataggio test
save('features_test.mat', 'featuresTestNorm', 'labelsTest', '-v7.3');

fprintf('Feature extraction completata!\n');
fprintf('Dimensione train: %d × %d\n', size(featuresTrainNorm,1), size(featuresTrainNorm,2));
fprintf('Dimensione test: %d × %d\n', size(featuresTestNorm,1), size(featuresTestNorm,2));
fprintf('Tempo totale: %.2f secondi.\n', toc);

% Chiusura pool
delete(gcp('nocreate'));

%% === HELPER FUNCTION ===
function y = ternary(cond, a, b)
    if cond
        y = a;
    else
        y = b;
    end
end