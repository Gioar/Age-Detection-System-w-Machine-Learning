% --- CONFIGURAZIONE DEI PERCORSI ---
% Path di base del dataset di input
inputBasePath = 'dataset_1'; 
% Path di base per il dataset di output
outputBasePath = 'dataset_1_compressed';

% Sottocartelle da elaborare all'interno del path di base
subfoldersToProcess = {'train', 'test'};

% Controlla se la cartella di output principale esiste, altrimenti creala
if ~exist(outputBasePath, 'dir')
    mkdir(outputBasePath);
end

% --- PARAMETRI DI COMPRESSIONE ---
% Parametro: soglia di energia (informazione mantenuta)
threshold = 0.75;

% Avvio cronometro per l'intero processo
totalTic = tic;

% --- CICLO PRINCIPALE SULLE SOTTOCARTELLE (train, test) ---
for k = 1:length(subfoldersToProcess)
    
    currentSubfolder = subfoldersToProcess{k};
    fprintf('\n--- Inizio elaborazione della cartella: %s ---\n', upper(currentSubfolder));
    
    % Costruisci il percorso completo per la cartella corrente (es. 'archive/train')
    currentInputPath = fullfile(inputBasePath, currentSubfolder);
    
    % Controlla se la cartella di input esiste prima di procedere
    if ~exist(currentInputPath, 'dir')
        fprintf('ATTENZIONE: La cartella %s non esiste. Salto al prossimo.\n', currentInputPath);
        continue; % Salta al prossimo ciclo (es. a 'test')
    end
    
    % Creazione imageDatastore per la cartella corrente
    imds = imageDatastore(currentInputPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    % Ciclo sulle immagini della cartella corrente
    numImages = numel(imds.Files);
    for i = 1:numImages
        % Prendi il filename
        filename = imds.Files{i};
        
        % Leggi i metadati per ottenere l'orientamento
        info = imfinfo(filename);
        if isfield(info, 'Orientation')
            orient = info.Orientation;
        else
            orient = 1; % Default: normale
        end
        
        % Leggi l'immagine (come uint8)
        img_uint8 = imread(filename);
        
        % Applica la correzione di orientamento
        img_uint8 = applyOrientation(img_uint8, orient);
        
        % Converti in double per la compressione
        img = im2double(img_uint8);

        % Separa i canali
        R = img(:,:,1);
        G = img(:,:,2);
        B = img(:,:,3);

        % Comprimi con SVD
        [R_compressed, kR] = compressSVD(R, threshold);
        [G_compressed, kG] = compressSVD(G, threshold);
        [B_compressed, kB] = compressSVD(B, threshold);

        % Ricostruzione immagine compressa
        compressedImg = cat(3, R_compressed, G_compressed, B_compressed);

        % --- NUOVA LOGICA DI SALVATAGGIO PER MANTENERE LA STRUTTURA ---
        
        % Prendi il percorso completo del file originale
        originalFilePath = filename;
        
        % Sostituisci il path di base di input con quello di output.
        % Questo è il modo più robusto per replicare la struttura delle cartelle.
        % Esempio: 'archive/train/18-20/img.jpg' diventa 'dataset_face_compressed/train/18-20/img.jpg'
        outputFilePath = strrep(originalFilePath, inputBasePath, outputBasePath);
        
        % Estrai la cartella di destinazione dal percorso completo del file
        [outputFolder, ~, ~] = fileparts(outputFilePath);
        
        % Se la cartella di destinazione finale non esiste, creala
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end
        
        % Salvo l'immagine compressa nel percorso corretto
        imwrite(compressedImg, outputFilePath);

        % Calcolo compressione e stampa avanzamento
        [m,n,~] = size(img);
        originalSize = 3 * m * n;
        compressedSize = (kR * (m+n+1)) + (kG * (m+n+1)) + (kB * (m+n+1));
        compressionRatio = originalSize / compressedSize;
        spaceSaving = 1 - (compressedSize / originalSize);

        fprintf('Cartella %s - Immagine %d/%d -> Ratio: %.2f, Saving: %.2f%%\n', ...
            currentSubfolder, i, numImages, compressionRatio, spaceSaving*100);

        % Mostra la prima per debug
        if i == 1
            figure;
            subplot(1,2,1); imshow(img); title('Original Image');
            subplot(1,2,2); imshow(compressedImg); 
            title(['Compressed (', num2str(threshold*100), '%, k_R=',num2str(kR), ...
                ', k_G=',num2str(kG), ', k_B=',num2str(kB), ')']);
        end
    end
    
    fprintf('--- Elaborazione della cartella %s completata ---\n', upper(currentSubfolder));
end

totalElapsedTime = toc(totalTic);
disp('--- DONE: Tutte le immagini di tutte le cartelle sono state compresse e salvate ---');
fprintf('Tempo totale di esecuzione: %.2f secondi\n', totalElapsedTime);


% --- FUNZIONE DI COMPRESSIONE ---
function [compressedChannel, k] = compressSVD(channel, infoThreshold)
    [U, S, V] = svd(channel,'econ'); 
    singularValues = diag(S);
    totalEnergy = sum(singularValues);
    cumulativeEnergy = cumsum(singularValues);
    k = find(cumulativeEnergy >= infoThreshold * totalEnergy, 1);
    
    % Gestisci il caso in cui k sia vuoto (soglia troppo alta)
    if isempty(k)
        k = length(singularValues);
    end
    
    compressedChannel = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
end

% --- FUNZIONE PER CORREGGERE L'ORIENTAMENTO EXIF ---
function correctedImg = applyOrientation(img, orient)
    switch orient
        case 1
            correctedImg = img;  % Normale, nessuna modifica
        case 2
            correctedImg = fliplr(img);  % Flip orizzontale
        case 3
            correctedImg = imrotate(img, 180);  % Rotazione 180°
        case 4
            correctedImg = flipud(img);  % Flip verticale
        case 5
            correctedImg = fliplr(imrotate(img, 90));  % Trasposta + flip verticale (rotazione 90° CW + flip orizzontale)
        case 6
            correctedImg = imrotate(img, -90);  % Rotazione 90° CW
        case 7
            correctedImg = fliplr(imrotate(img, -90));  % Trasposta + flip orizzontale (rotazione 90° CCW + flip orizzontale)
        case 8
            correctedImg = imrotate(img, 90);  % Rotazione 90° CCW
        otherwise
            correctedImg = img;  % Default: nessuna modifica
    end
end