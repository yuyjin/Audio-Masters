clear; 
clc; 
close all;
%  list audio train files
audios = {'Zero_train1.wav','Zero_train2.wav','Zero_train3.wav','Zero_train4.wav','Zero_train6.wav','Zero_train8.wav','Zero_train9.wav','Zero_train10.wav','Zero_train11.wav','Zero_train12.wav','Zero_train13.wav','Zero_train15.wav','Zero_train16.wav','Zero_train17.wav','Zero_train18.wav','Zero_train19.wav'};

% parameters for stft
N = 512; %frame and fft length
M = 200; %overlap
dimensions = 2;
D = cell(length(audios),1);

% load and apply stft on audios
% s: 3-D array
% f: vector samples
% t: vector Lu
% y: audio data matrix

    cencell = cell(1,length(audios));
for A = 1:length(audios)
    [y, fs] = audioread(audios{A});
    [s, f, t] = stft(y, fs, 'Window', hamming(N), 'OverlapLength', M, 'FFTLength', N);

    % apply melfb
    K = 20; 
    m = melfb(K, N, fs);
        
    % calculate the spectrum(before)
    Spectrum = abs(s);

    % apply melfb to the spectrum(after)
    melSpectrum = m * Spectrum(1:size(m, 2), :);

    % apply MFCC on the Spectrum
    mfccs{A} = mfcc_test(melSpectrum, K);


    % delete the first K of mfccs
    mfccs{A}(1, :) = []; 

    % apply k-means clustering on the MFCCs
    numCluster = 30;
    [idx, centroids] = kmeans(mfccs{A}.', numCluster);

    cencell{A} = centroids;
end
%% Testing
% We are done with training, we will now do testing
% list audio test files
audio_test = {'Zero_train1.wav','Zero_train2.wav','Zero_train3.wav','Zero_train4.wav','Zero_train6.wav','Zero_train8.wav','Zero_train9.wav','Zero_train10.wav','Zero_train11.wav','Zero_train12.wav','Zero_train13.wav','Zero_train15.wav','Zero_train16.wav','Zero_train17.wav','Zero_train18.wav','Zero_train19.wav'};

% parameters for testing stft
N_t = 512; %frame and fft length
M_t = 200; %overlap
D = cell(length(audio_test),1);

cencell_t = cell(1,length(audio_test));
for B = 1:length(audio_test)
    [y_t, fs_t] = audioread(audio_test{B});
    [s_t, f_t, t_t] = stft(y_t, fs_t, 'Window', hamming(N_t), 'OverlapLength', M_t, 'FFTLength', N_t);

    % apply melfb
    K_t = 20;
    m_t = melfb(K_t, N_t, fs_t);
        
    % calculate the spectrum(before)
    Spectrum_t = abs(s_t);

    % apply melfb to the spectrum(after)
    melSpectrum_t = m_t * Spectrum_t(1:size(m_t, 2), :);

    % apply MFCC on the Spectrum
    mfccs_t{B} = mfcc_test(melSpectrum_t, K_t);

    % calculate the spectrum (before)
    spectrum_t = abs(s_t);

    % apply melfb to the spectrum (after)
    melSpectrum_t = m_t * spectrum_t(1:size(m_t,2),:);

    % apply MFCC on the Spectrum
    mfccs_t{B} = mfcc_test(melSpectrum_t, K_t);

    
    % delete the first K of mfccs
    mfccs_t{B}(1,:) = [];
    
    % k-means
    [idx, centroids] = kmeans(mfccs_t{B}.', numCluster);
    cencell_t{B} = centroids;

end
    
total_min_distances = zeros(length(audio_test), length(audios));

for B = 1:length(audio_test)
    for A = 1:length(audios)
        % Compute distance between each centroid from test and each centroid from train
        distances_each = disteu(transpose(cencell_t{B}), transpose(cencell{A}));

        % Instead of summing all distances, focus on the minimum or average distances
        total_min_distances(B, A) = mean(min(distances_each, [], 2)); 
    end
end

for B = 1:length(audio_test)
    % Identify the speaker (training set) with the lowest total distance for each test set
    [~, identified_speaker] = min(total_min_distances(B, :));
    fprintf('The identified speaker for test %d is: %d\n', B, identified_speaker);
end


