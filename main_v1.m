clear; clc; close all;
%  list audios
audios = {'Zero_train2.wav'};

% parameters for stft
N = 516; %frame and fft length
M = 200; %overlap

% load and apply stft on audios
% s: 3-D array
% f: vector samples
% t: vector Lu
% y: audio data matrix
for i = 1:length(audios)
    [y, fs] = audioread(audios{i});
    [s, f, t] = stft(y, fs, 'Window', hamming(N), 'OverlapLength', M, 'FFTLength', N);

    % apply melfb
    K = 20; 
    m = melfb(K, N, fs);
    
    % Plot melfb responses
    figure;
    plot(linspace(0, fs/2, size(m,2)), m');
    title(['melfb responses' audios{i}]);
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
        
    % calculate the spectrum(before)
    Spectrum = abs(s);

    % apply melfb to the spectrum(after)
    melSpectrum = m * Spectrum(1:size(m, 2), :);

    % Plot the original spectrum
    figure;
    surf(t, f, 20*log10(Spectrum), 'EdgeColor', 'none');
    axis tight;
    view(0, 90);
    title(['Original Spectrum' audios{i}]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;

    % Plot the spectrum after melfb
    figure;
    surf(t, linspace(0, fs/2, K), 20*log10(melSpectrum), 'EdgeColor', 'none');
    axis tight;
    view(0, 90);
    title(['Mel Spectrum' audios{i}]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;
    
    % apply MFCC on the Spectrum
    mfccs = mfcc_test(melSpectrum, K);

    figure;
    for k = 1:K
        subplot(K, 1, k);
        plot(t, mfccs(k, :));
        if k == 1
        title('MFCCs');
        end
        if k < K
            set(gca, 'xtick', []);
        else
            xlabel('Time (s)');
        end
        ylabel(sprintf('C%d', k));
    end

end

% delete the first K of mfccs
mfccs(1, :) = []; 

% apply k-means clustering on the MFCCs
% cluster number equals to Lu/6
numCluster = floor(size(mfccs, 2) / 6);
[idx, codewords] = kmeans(mfccs.', numCluster);

% Plot the clustered MFCC vectors
% Plot the 2nd and 3rd MFCC coefficients
figure;
scatter(mfccs(2, :), mfccs(3, :), 'filled'); 
xlabel('Second MFCC Coefficient');
ylabel('Third MFCC Coefficient');
title('2D Plot of MFCCs');

% next we use LGB to find clusters
% (may also cluster numbers---not just Lu/6)

