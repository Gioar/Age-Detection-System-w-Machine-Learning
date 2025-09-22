clc; clear;
tic;

% === PARAMETRI ===
mode = 'test'; % 'train' o 'test' È NECESSARIO ESEGUIRE ENTRAMBE LE MODALITÀ PRIMA DI PROCEDERE CON I CODICI SUCCESSIVI
baseInputPath = 'final_dataset_face'; % Cartella di input
inputFolder = fullfile(baseInputPath, mode); % Cartella di input: final_dataset_face/train o final_dataset_face/test
outputFolder = fullfile(baseInputPath, [mode '_aug']); % Cartella di output: final_dataset_face/train_aug o final_dataset_face/test_aug
resizeDim = [224 224]; % Dimensione finale delle immagini

% Parametri per il training
augmentationTargetRatio = 0.7; 
maxAugPerImage = 3;

% === CARICAMENTO DATASET ===
imds = imageDatastore(inputFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

counts = countEachLabel(imds);
fprintf('%s originale: %d immagini trovate in %d classi.\n', ...
    mode, sum(counts.Count), numel(counts.Label));

% === CREAZIONE CARTELLA DI OUTPUT ===
if exist(outputFolder,'dir')
    warning('⚠️  La cartella di output esiste già. Verrà sovrascritta.');
    rmdir(outputFolder,'s');
end
mkdir(outputFolder);

%% === PARAMETRI AUGMENTER ===
if strcmp(mode,'train')
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation', [-10 10], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5], ...
        'RandXReflection', true);

    maxClass = max(counts.Count);
    targetPerClass = round(maxClass * augmentationTargetRatio);
    fprintf('Target per classe (train): ~%d immagini (%.0f%% della più numerosa)\n', ...
        targetPerClass, augmentationTargetRatio*100);
else
    fprintf('Modalità test: solo resize, nessuna augmentation.\n');
end

%% === LOOP SULLE VARIE CLASSI ===
totalGenerated = 0;
for i = 1:numel(counts.Label)
    label = counts.Label(i);
    nImages = counts.Count(i);

    % creazione sottocartella di output
    outputSubfolder = fullfile(outputFolder, char(label));
    mkdir(outputSubfolder);

    % filtraggio immagini, per classe
    classIdx = imds.Labels == label;
    classFiles = imds.Files(classIdx);

    % copia / salvataggio immagini originali con resize
    for j = 1:numel(classFiles)
        [~,name,ext] = fileparts(classFiles{j});
        imgOriginal = imread(classFiles{j});
        imgOriginal = imresize(imgOriginal, resizeDim);
        imwrite(imgOriginal, fullfile(outputSubfolder, [name ext]));
    end

    % === AUGMENTATION ===
    if strcmp(mode,'train')
        nToGenerate = max(targetPerClass - nImages, 0);
        if nToGenerate == 0
            fprintf('Classe %s già sopra il target, nessuna augmentation.\n', label);
            continue;
        end

        fprintf('Classe %s: %d immagini → generazione di %d immagini...\n', ...
            label, nImages, nToGenerate);

        augCount = 0;
        augPerImage = zeros(numel(classFiles), 1);  % Contatore di augmentation sulla singola immagine

        while augCount < nToGenerate
            % Ricerca di nuove immagini su cui applicare augmentation
            availableIdx = find(augPerImage < maxAugPerImage);
            if isempty(availableIdx)
                warning('Classe %s: Raggiunto limite maxAugPerImage per tutte le immagini. Generati solo %d su %d.', ...
                    label, augCount, nToGenerate);
                break;
            end

            % Scelta random tra quelle disponibili
            idx = availableIdx(randi(numel(availableIdx)));
            img = imread(classFiles{idx});

            % Augmentation geometrica
            augImg = augment(imageAugmenter, img);

            % Variazione luminosità
            augImg = im2double(augImg);
            brightnessFactor = 0.9 + 0.2*rand();
            augImg = augImg * brightnessFactor;

            % Variazione contrasto (gamma random tra 0.8 e 1.2)
            gammaFactor = 0.8 + 0.4*rand();
            augImg = imadjust(augImg, [], [], gammaFactor);

            % Variazione sharpness (amount random tra 0.5 e 1.5)
            sharpnessAmount = 0.5 + rand();
            augImg = imsharpen(augImg, 'Amount', sharpnessAmount);

            % Noise gaussiano (variance random tra 0.001 e 0.005)
            noiseVariance = 0.001 + 0.004*rand();
            augImg = imnoise(augImg, 'gaussian', 0, noiseVariance);

            % Clip e conversione a uint8
            augImg = min(max(augImg,0),1);
            augImg = im2uint8(augImg);

            % Resize
            augImg = imresize(augImg, resizeDim);

            % Salvataggio
            outName = sprintf('aug_%05d.png', augCount+1);
            imwrite(augImg, fullfile(outputSubfolder, outName));

            augCount = augCount + 1;
            augPerImage(idx) = augPerImage(idx) + 1;
            totalGenerated = totalGenerated + 1;
        end
    end
end

%% === REPORT FINALE ===
newImds = imageDatastore(outputFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
newCounts = countEachLabel(newImds);

fprintf('\n=== REPORT FINALE ===\n');
disp(newCounts);
fprintf('Totale immagini dopo preprocessing: %d\n', sum(newCounts.Count));
fprintf('Tempo di esecuzione: %.2f secondi.\n', toc);