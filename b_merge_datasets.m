clear;
clc;
close all;

% Avvio del cronometro per misurare il tempo di esecuzione
tic;

fprintf('Inizio del processo di unione dei dataset...\n\n');

% --- FASE 0: CONFIGURAZIONE ---
% Definizione dei percorsi di origine e di destinazione dei dataset 
dataset1_dir = 'dataset_1_compressed';
dataset2_dir = 'dataset_2';
dataset3_dir = 'dataset_3';
final_dataset_dir = 'final_dataset';

% Definizione delle classi per la suddivisione (train/test)
splits = {'train', 'test'};
classes = {'18-20', '21-30', '31-40', '41-50', '51-60'};

% Contatore globale di immagini (per rinominarle)
image_counter = 1;


% --- FASE 1: CREAZIONE DELLA STRUTTURA DEL DATASET FINALE ---
fprintf('FASE 1: Creazione della struttura del dataset finale in ''%s''...\n', final_dataset_dir);

if isfolder(final_dataset_dir)
    fprintf('La cartella ''%s'' esiste già. Verrà rimossa e ricreata.\n', final_dataset_dir);
    rmdir(final_dataset_dir, 's'); % Rimuove la cartella e il suo contenuto
end

for i = 1:length(splits)
    split = splits{i};
    for j = 1:length(classes)
        cls = classes{j};
        mkdir(fullfile(final_dataset_dir, split, cls));
    end
end
fprintf('Struttura creata con successo.\n\n');


% --- FASE 2: ELABORAZIONE DATASET 1 ---
fprintf('FASE 2: Inizio elaborazione dataset_1...\n');
if isfolder(dataset1_dir)
    for i = 1:length(splits)
        split = splits{i};
        for j = 1:length(classes)
            cls = classes{j};
            
            source_path = fullfile(dataset1_dir, split, cls);
            dest_path = fullfile(final_dataset_dir, split, cls);
            
            if ~isfolder(source_path)
                fprintf('Attenzione: La cartella %s non esiste. Sarà saltata.\n', source_path);
                continue;
            end
            
            files = dir(fullfile(source_path, '*.*'));
            for k = 1:length(files)
                file = files(k);
                if ~file.isdir % Ignorare le sottodirectory '.' e '..'
                    [~, ~, ext] = fileparts(file.name);
                    source_file = fullfile(source_path, file.name);
                    new_filename = sprintf('%06d%s', image_counter, ext);
                    dest_file = fullfile(dest_path, new_filename);
                    
                    copyfile(source_file, dest_file);
                    image_counter = image_counter + 1;
                end
            end
        end
    end
    fprintf('Elaborazione dataset_1 completata.\n\n');
else
    fprintf('Attenzione: La cartella %s non esiste. Sarà saltata.\n\n', dataset1_dir);
end


% --- FASE 3: ELABORAZIONE DATASET 2 ---
fprintf('FASE 3: Inizio elaborazione dataset_2...\n');
if isfolder(dataset2_dir)
    source_splits_d2 = {'train', 'test', 'valid'};
    dest_splits_map_d2 = containers.Map(source_splits_d2, {'train', 'test', 'train'});
    
    class_map_d2 = containers.Map(...
        {'x16_20', 'x21_30', 'x31_40', 'x41_50', 'x51_60'}, ...
        {'18-20', '21-30', '31-40', '41-50', '51-60'} ...
    );
    class_keys_d2 = keys(class_map_d2);
    
    for i = 1:length(source_splits_d2)
        source_split = source_splits_d2{i};
        dest_split = dest_splits_map_d2(source_split);
        
        csv_path = fullfile(dataset2_dir, source_split, '_classes.csv');
        images_path = fullfile(dataset2_dir, source_split);
        
        if ~isfile(csv_path)
            fprintf('Attenzione: Il file %s non esiste. Sarà saltato.\n', csv_path);
            continue;
        end
        
        opts = detectImportOptions(csv_path);
        opts.VariableNamingRule = 'modify'; 
        dataTable = readtable(csv_path, opts);
        
        for row = 1:height(dataTable)
            for k = 1:length(class_keys_d2)
                csv_class_name = class_keys_d2{k};
                if dataTable.(csv_class_name)(row) == 1
                    filename = dataTable.filename{row};
                    source_file = fullfile(images_path, filename);
                    
                    if isfile(source_file)
                        final_class = class_map_d2(csv_class_name);
                        dest_path = fullfile(final_dataset_dir, dest_split, final_class);
                        
                        [~, ~, ext] = fileparts(filename);
                        new_filename = sprintf('%06d%s', image_counter, ext);
                        dest_file = fullfile(dest_path, new_filename);
                        
                        copyfile(source_file, dest_file);
                        image_counter = image_counter + 1;
                        break; % Passaggio alla riga successiva, una volta trovata la classe di appartenenza
                    end
                end
            end
        end
    end
    fprintf('Elaborazione dataset_2 completata.\n\n');
else
    fprintf('Attenzione: La cartella %s non esiste. Sarà saltata.\n\n', dataset2_dir);
end


% --- FASE 4: ELABORAZIONE DATASET 3 ---
fprintf('FASE 4: Inizio elaborazione dataset_3...\n');
if isfolder(dataset3_dir)
    source_splits_d3 = {'Train', 'Test', 'Validation'};
    dest_splits_map_d3 = containers.Map(source_splits_d3, {'train', 'test', 'train'});
    
    for i = 1:length(source_splits_d3)
        source_split = source_splits_d3{i};
        dest_split = dest_splits_map_d3(source_split);
        
        csv_path = fullfile(dataset3_dir, 'Index', [source_split, '.csv']);
        images_path = fullfile(dataset3_dir, 'Images', source_split);
        
        if ~isfile(csv_path)
            fprintf('Attenzione: Il file %s non esiste. Sarà saltato.\n', csv_path);
            continue;
        end
        
        dataTable = readtable(csv_path);
        
        for row = 1:height(dataTable)
            age = dataTable.age(row);
            
            % Determinazione della classe di età
            age_class = '';
            if age >= 18 && age <= 20
                age_class = '18-20';
            elseif age >= 21 && age <= 30
                age_class = '21-30';
            elseif age >= 31 && age <= 40
                age_class = '31-40';
            elseif age >= 41 && age <= 50
                age_class = '41-50';
            elseif age >= 51 && age <= 60
                age_class = '51-60';
            end
            
            if ~isempty(age_class)
                filename = dataTable.filename{row};
                source_file = fullfile(images_path, filename);
                
                if isfile(source_file)
                    dest_path = fullfile(final_dataset_dir, dest_split, age_class);
                    [~, ~, ext] = fileparts(filename);
                    new_filename = sprintf('%06d%s', image_counter, ext);
                    dest_file = fullfile(dest_path, new_filename);
                    
                    copyfile(source_file, dest_file);
                    image_counter = image_counter + 1;
                end
            end
        end
    end
    fprintf('Elaborazione dataset_3 completata.\n\n');
else
    fprintf('Attenzione: La cartella %s non esiste. Sarà saltata.\n\n', dataset3_dir);
end


% --- FASE 5: COMPLETAMENTO E OUTPUT FINALE ---
% Stop del cronometro
elapsedTime = toc;

fprintf('=======================================================\n');
fprintf('PROCESSO COMPLETATO CON SUCCESSO!\n');
fprintf('Il dataset finale si trova in: ''%s''\n', final_dataset_dir);
fprintf('Numero totale di immagini nel dataset finale: %d\n', image_counter - 1);
fprintf('Tempo di esecuzione totale: %.2f secondi.\n', elapsedTime);
fprintf('=======================================================\n');