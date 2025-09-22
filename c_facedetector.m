close all; 
clear; 
clc;

% --- CONFIGURAZIONE DEI PERCORSI ---
% Path di base del dataset di input
inputBasePath = 'final_dataset';
% Path di base per il dataset di output
outputBasePath = 'final_dataset_face';

% Sottocartelle da elaborare all'interno del path di base
subfoldersToProcess = {'train', 'test'};

% Controllo esistenza cartella di output
if ~exist(outputBasePath, 'dir')
    mkdir(outputBasePath);
    disp(['Cartella di output principale creata in: ', outputBasePath]);
end

% Creazione oggetto globale per riconoscimento volti
faceDetector = vision.CascadeObjectDetector();

% Avvio cronometro per l'intero processo
totalTic = tic;

% --- CICLO PRINCIPALE SULLE SOTTOCARTELLE (train, test) ---
for k = 1:length(subfoldersToProcess)
    
    currentSubfolder = subfoldersToProcess{k};
    fprintf('\n--- Inizio rilevamento volti nella cartella: %s ---\n', upper(currentSubfolder));
    
    % Costruzione del percorso completo
    currentInputPath = fullfile(inputBasePath, currentSubfolder);
    
    % Controllo esistenza percorso di input
    if ~exist(currentInputPath, 'dir')
        fprintf('ATTENZIONE: La cartella %s non esiste. Salto al prossimo.\n', currentInputPath);
        continue;
    end
    
    % Creazione imageDatastore relativo alla cartella corrente
    imds = imageDatastore(currentInputPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
        
    numImagesToProcess = numel(imds.Files); % Rappresenta il numero di immagini da analizzare nella cartella corrente

    % --- CICLO SULLE IMMAGINI DELLA CARTELLA CORRENTE ---
    for i = 1:numImagesToProcess
        
        % Lettura immagine originale
        imgOriginal = imread(imds.Files{i});
        
        % Rilevazione volto
        bboxes = faceDetector(imgOriginal);
        
        % Controllo di rilevamento
        if ~isempty(bboxes)
            % Ricerca del volto più grande, eventuali più piccoli vengono
            % ignorati
            [~, maxIdx] = max(bboxes(:,3) .* bboxes(:,4));
            mainBbox = bboxes(maxIdx,:);
            
            % Ritaglio dell'immagine usando il rettangolo trovato
            imgCropped = imcrop(imgOriginal, mainBbox);
            
            % --- SALVATAGGIO ---
            
            % Salvataggio percorso originale
            originalFilePath = imds.Files{i};
            
            % Sostituzione del percorso originale con il nuovo di output
            outputFilePath = strrep(originalFilePath, inputBasePath, outputBasePath);
            
            % Estrazione della cartella in cui salvare i volti
            [outputFolder, ~, ~] = fileparts(outputFilePath);
            
            % Creazione della cartella di output, se necessario
            if ~exist(outputFolder, 'dir')
                mkdir(outputFolder);
            end
            
            % Salvataggio dell'immagine con il volto ritagliato
            imwrite(imgCropped, outputFilePath);
            
            % Messaggio di controllo, con volto rilevato
            fprintf('Cartella %s - Immagine %d/%d: Volto ritagliato salvato.\n', currentSubfolder, i, numImagesToProcess);
            
        else
            % Messaggio di controllo, senza volto rilevato
            fprintf('ATTENZIONE: Nessun volto trovato in %s - Immagine %d/%d: %s\n', ...
                currentSubfolder, i, numImagesToProcess, imds.Files{i});
        end
    end
    
    fprintf('--- Rilevamento volti nella cartella %s completato ---\n', upper(currentSubfolder));
end

totalElapsedTime = toc(totalTic);
disp('--- DONE: Tutti i volti di tutte le cartelle sono stati ritagliati e salvati ---');
fprintf('Tempo totale di esecuzione: %.2f secondi\n', totalElapsedTime);

% --- DEBUG: Visualizzazione di alcuni esempi ---
disp('Visualizzazione di un esempio dalla cartella TEST per verifica...');
try
    testPath = fullfile(outputBasePath, 'test');
    imds_test_faces = imageDatastore(testPath, 'IncludeSubfolders', true);
    figure;
    imshow(readimage(imds_test_faces, 1));
    title('Esempio di volto salvato dalla cartella TEST');
catch ME
    disp('Impossibile visualizzare un esempio dalla cartella TEST. Forse non ci sono immagini o la cartella non esiste.');
end