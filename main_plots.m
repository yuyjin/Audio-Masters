clear; 
clc; 
close all;
%  list audios
audios = {'Zero_train3.wav'};
audio_test = {'Zero_test3.wav'};

% find the max length of all audios for normalization
max_length = 0;
for A = 1:length(audios)
    info = audioinfo(audios{A});
    max_length = max(max_length, info.TotalSamples);
end
for B = 1:length(audio_test)
    info = audioinfo(audio_test{B});
    max_length = max(max_length, info.TotalSamples);
end

% parameters for stft
N = 516; %frame and fft length
M = 200; %overlap

% load and apply stft on audios
% s: 3-D array
% f: vector samples
% t: vector Lu
% y: audio data matrix
for A = 1:length(audios)
    [y, fs] = audioread(audios{A});
    
    % Plot the signal in the time domain
    t = (0:length(y) - 1) / fs; 
    figure(1); 
    plot(t, y);
    title(['Time-Domain Signal of ' audios{A}]);
    xlabel('Time (seconds)');
    ylabel('Amplitude');

    % normalize the amplitude of each signal to 1
    y = y / max(abs(y));

    % Zero Pad the signal if it is shorter than the longest signal
    if length(y) < max_length
        y = [y; zeros(max_length - length(y), 1)];
    end
    
    % Plot the normalized signal in the time domain
    t = (0:length(y) - 1) / fs; 
    figure(2);
    plot(t, y);
    title(['Normalized Time-Domain Signal of ' audios{A}]);
    xlabel('Time (seconds)');
    ylabel('Normalized Amplitude');

    % apply stft
    [s, f, t] = stft(y, fs, 'Window', hamming(N), 'OverlapLength', M, 'FFTLength', N);

    % apply melfb
    K = 20; 
    m = melfb(K, N, fs);
    
    % Plot melfb responses
    figure(3);
    plot(linspace(0, fs/2, size(m,2)), m');
    title(['mel-spaced filter bank responses' audios{A}]);
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
        
    % calculate the spectrum(before)
    Spectrum = abs(s);

    % apply melfb to the spectrum(after)
    melSpectrum = m * Spectrum(1:size(m, 2), :);

    % Plot the original spectrum
    figure(4);
    surf(t, f, 20*log10(Spectrum), 'EdgeColor', 'none');
    axis tight;
    view(0, 90);
    title(['Original Spectrum' audios{A}]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;

    % Plot the spectrum after melfb
    figure(5);
    surf(t, linspace(0, fs/2, K), 20*log10(melSpectrum), 'EdgeColor', 'none');
    axis tight;
    view(0, 90);
    title(['Mel Spectrum' audios{A}]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;
    
    % apply MFCC on the Spectrum
    mfccs{A} = mfcc_test(melSpectrum, K);

    figure(6);
    for k = 1:K
        subplot(K, 1, k);
        plot(t, mfccs{A}(k, :));
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

    % delete the first K of mfccs
    mfccs{A}(1, :) = []; 
    
    % Plot the clustered MFCC vectors
    % Plot the 2nd and 3rd MFCC coefficients
    figure(7);
    scatter(mfccs{A}(1, :), mfccs{A}(2, :), 'filled'); 
    xlabel('Second MFCC Coefficient');
    ylabel('Third MFCC Coefficient');
    title('2D Plot of MFCCs');

    first_centroid_x{A} = mean(mfccs{A}(1,:));
    first_centroid_y{A} = mean(mfccs{A}(2,:));
    first_centroid_z1{A} = mean(mfccs{A}(3,:)); % 3rd dimension
    first_centroid_z2{A} = mean(mfccs{A}(4,:)); % 4th dimension
    first_centroid_z3{A} = mean(mfccs{A}(5,:)); % 5th dimension
    first_centroid_z4{A} = mean(mfccs{A}(6,:)); % 6th dimension
    first_centroid_z5{A} = mean(mfccs{A}(7,:)); % 7th dimension
    first_centroid_z6{A} = mean(mfccs{A}(8,:)); % 8th dimension
    first_centroid_z7{A} = mean(mfccs{A}(9,:)); % 9th dimension
    first_centroid_z8{A} = mean(mfccs{A}(10,:)); % 10th dimension
    first_centroid_z9{A} = mean(mfccs{A}(11,:)); % 11th dimension
    first_centroid_z10{A} = mean(mfccs{A}(12,:)); % 12th dimension
    first_centroid_z11{A} = mean(mfccs{A}(13,:)); % 13th dimension
    first_centroid_z12{A} = mean(mfccs{A}(14,:)); % 14th dimension
    first_centroid_z13{A} = mean(mfccs{A}(15,:)); % 15th dimension
    first_centroid_z14{A} = mean(mfccs{A}(16,:)); % 16th dimension
    first_centroid_z15{A} = mean(mfccs{A}(17,:)); % 17th dimension
    first_centroid_z16{A} = mean(mfccs{A}(18,:)); % 18th dimension
    first_centroid_z17{A} = mean(mfccs{A}(19,:)); % 19th dimension
    first_centroid_new{A} = cat(1,first_centroid_x{A}, first_centroid_y{A}, first_centroid_z1{A}, first_centroid_z2{A}, first_centroid_z3{A}, first_centroid_z4{A}, first_centroid_z5{A}, first_centroid_z6{A}, first_centroid_z7{A}, first_centroid_z8{A}, first_centroid_z9{A}, first_centroid_z10{A}, first_centroid_z11{A}, first_centroid_z12{A}, first_centroid_z13{A}, first_centroid_z14{A}, first_centroid_z15{A}, first_centroid_z16{A}, first_centroid_z17{A});
    epsilon_split = 0.001;
    
    % select the number of centroids
    n_centroids = 4;
    matching_array_ari{A} = mfccs{A};
    epsilon1 = 0.001;
    ari = 1;
    current_n_centroids{A} = 2;
    
    D2b{A}=[9999];
    D3b{A}=[9999];
    while current_n_centroids{A} <= n_centroids
        % fprintf("number of current centroids is %d\n", current_n_centroids{A});
        l = 1;
        distance_key2{A} = 9999; % choose a number bigger than 0.001
        while abs(distance_key2{A}) > epsilon1
            
            if l == 1
                if current_n_centroids{A} == 2
                    centroids2{A} = [first_centroid_new{A}*(1-epsilon_split) first_centroid_new{A}*(1+epsilon_split)];
                elseif current_n_centroids{A} == 4
                    centroids2{A} = [grouping_sums_ari{A}{1}*(1-epsilon_split) grouping_sums_ari{A}{1}*(1+epsilon_split) grouping_sums_ari{A}{2}*(1-epsilon_split) grouping_sums_ari{A}{2}*(1+epsilon_split)];
                elseif current_n_centroids{A} == 8
                    centroids2{A} = [grouping_sums_ari{A}{1}*(1-epsilon_split) grouping_sums_ari{A}{1}*(1+epsilon_split) grouping_sums_ari{A}{2}*(1-epsilon_split) grouping_sums_ari{A}{2}*(1+epsilon_split) grouping_sums_ari{A}{3}*(1-epsilon_split) grouping_sums_ari{A}{3}*(1+epsilon_split) grouping_sums_ari{A}{4}*(1-epsilon_split) grouping_sums_ari{A}{4}*(1+epsilon_split)];
                elseif current_n_centroids{A} == 16
                    centroids2{A} = [grouping_sums_ari{A}{1}*(1-epsilon_split) grouping_sums_ari{A}{1}*(1+epsilon_split) grouping_sums_ari{A}{2}*(1-epsilon_split) grouping_sums_ari{A}{2}*(1+epsilon_split) grouping_sums_ari{A}{3}*(1-epsilon_split) grouping_sums_ari{A}{3}*(1+epsilon_split) grouping_sums_ari{A}{4}*(1-epsilon_split) grouping_sums_ari{A}{4}*(1+epsilon_split) grouping_sums_ari{A}{5}*(1-epsilon_split) grouping_sums_ari{A}{5}*(1+epsilon_split) grouping_sums_ari{A}{6}*(1-epsilon_split) grouping_sums_ari{A}{6}*(1+epsilon_split) grouping_sums_ari{A}{7}*(1-epsilon_split) grouping_sums_ari{A}{7}*(1+epsilon_split) grouping_sums_ari{A}{8}*(1-epsilon_split) grouping_sums_ari{A}{8}*(1+epsilon_split)];
                elseif current_n_centroids{A} == 32
                    centroids2{A} = [grouping_sums_ari{A}{1}*(1-epsilon_split) grouping_sums_ari{A}{1}*(1+epsilon_split) grouping_sums_ari{A}{2}*(1-epsilon_split) grouping_sums_ari{A}{2}*(1+epsilon_split) grouping_sums_ari{A}{3}*(1-epsilon_split) grouping_sums_ari{A}{3}*(1+epsilon_split) grouping_sums_ari{A}{4}*(1-epsilon_split) grouping_sums_ari{A}{4}*(1+epsilon_split) grouping_sums_ari{A}{5}*(1-epsilon_split) grouping_sums_ari{A}{5}*(1+epsilon_split) grouping_sums_ari{A}{6}*(1-epsilon_split) grouping_sums_ari{A}{6}*(1+epsilon_split) grouping_sums_ari{A}{7}*(1-epsilon_split) grouping_sums_ari{A}{7}*(1+epsilon_split) grouping_sums_ari{A}{8}*(1-epsilon_split) grouping_sums_ari{A}{8}*(1+epsilon_split) grouping_sums_ari{A}{9}*(1-epsilon_split) grouping_sums_ari{A}{9}*(1+epsilon_split) grouping_sums_ari{A}{10}*(1-epsilon_split) grouping_sums_ari{A}{10}*(1+epsilon_split) grouping_sums_ari{A}{11}*(1-epsilon_split) grouping_sums_ari{A}{11}*(1+epsilon_split) grouping_sums_ari{A}{12}*(1-epsilon_split) grouping_sums_ari{A}{12}*(1+epsilon_split) grouping_sums_ari{A}{13}*(1-epsilon_split) grouping_sums_ari{A}{13}*(1+epsilon_split) grouping_sums_ari{A}{14}*(1-epsilon_split) grouping_sums_ari{A}{14}*(1+epsilon_split) grouping_sums_ari{A}{15}*(1-epsilon_split) grouping_sums_ari{A}{15}*(1+epsilon_split) grouping_sums_ari{A}{16}*(1-epsilon_split) grouping_sums_ari{A}{16}*(1+epsilon_split)];
                elseif current_n_centroids{A} == 64
                    centroids2{A} = [grouping_sums_ari{A}{1}*(1-epsilon_split) grouping_sums_ari{A}{1}*(1+epsilon_split) grouping_sums_ari{A}{2}*(1-epsilon_split) grouping_sums_ari{A}{2}*(1+epsilon_split) grouping_sums_ari{A}{3}*(1-epsilon_split) grouping_sums_ari{A}{3}*(1+epsilon_split) grouping_sums_ari{A}{4}*(1-epsilon_split) grouping_sums_ari{A}{4}*(1+epsilon_split) grouping_sums_ari{A}{5}*(1-epsilon_split) grouping_sums_ari{A}{5}*(1+epsilon_split) grouping_sums_ari{A}{6}*(1-epsilon_split) grouping_sums_ari{A}{6}*(1+epsilon_split) grouping_sums_ari{A}{7}*(1-epsilon_split) grouping_sums_ari{A}{7}*(1+epsilon_split) grouping_sums_ari{A}{8}*(1-epsilon_split) grouping_sums_ari{A}{8}*(1+epsilon_split) grouping_sums_ari{A}{9}*(1-epsilon_split) grouping_sums_ari{A}{9}*(1+epsilon_split) grouping_sums_ari{A}{10}*(1-epsilon_split) grouping_sums_ari{A}{10}*(1+epsilon_split) grouping_sums_ari{A}{11}*(1-epsilon_split) grouping_sums_ari{A}{11}*(1+epsilon_split) grouping_sums_ari{A}{12}*(1-epsilon_split) grouping_sums_ari{A}{12}*(1+epsilon_split) grouping_sums_ari{A}{13}*(1-epsilon_split) grouping_sums_ari{A}{13}*(1+epsilon_split) grouping_sums_ari{A}{14}*(1-epsilon_split) grouping_sums_ari{A}{14}*(1+epsilon_split) grouping_sums_ari{A}{15}*(1-epsilon_split) grouping_sums_ari{A}{15}*(1+epsilon_split) grouping_sums_ari{A}{16}*(1-epsilon_split) grouping_sums_ari{A}{16}*(1+epsilon_split) grouping_sums_ari{A}{17}*(1-epsilon_split) grouping_sums_ari{A}{17}*(1+epsilon_split) grouping_sums_ari{A}{18}*(1-epsilon_split) grouping_sums_ari{A}{18}*(1+epsilon_split) grouping_sums_ari{A}{19}*(1-epsilon_split) grouping_sums_ari{A}{19}*(1+epsilon_split) grouping_sums_ari{A}{20}*(1-epsilon_split) grouping_sums_ari{A}{20}*(1+epsilon_split) grouping_sums_ari{A}{21}*(1-epsilon_split) grouping_sums_ari{A}{21}*(1+epsilon_split) grouping_sums_ari{A}{22}*(1-epsilon_split) grouping_sums_ari{A}{22}*(1+epsilon_split) grouping_sums_ari{A}{23}*(1-epsilon_split) grouping_sums_ari{A}{23}*(1+epsilon_split) grouping_sums_ari{A}{24}*(1-epsilon_split) grouping_sums_ari{A}{24}*(1+epsilon_split) grouping_sums_ari{A}{25}*(1-epsilon_split) grouping_sums_ari{A}{25}*(1+epsilon_split) grouping_sums_ari{A}{26}*(1-epsilon_split) grouping_sums_ari{A}{26}*(1+epsilon_split) grouping_sums_ari{A}{27}*(1-epsilon_split) grouping_sums_ari{A}{27}*(1+epsilon_split) grouping_sums_ari{A}{28}*(1-epsilon_split) grouping_sums_ari{A}{28}*(1+epsilon_split) grouping_sums_ari{A}{29}*(1-epsilon_split) grouping_sums_ari{A}{29}*(1+epsilon_split) grouping_sums_ari{A}{30}*(1-epsilon_split) grouping_sums_ari{A}{30}*(1+epsilon_split) grouping_sums_ari{A}{31}*(1-epsilon_split) grouping_sums_ari{A}{31}*(1+epsilon_split) grouping_sums_ari{A}{32}*(1-epsilon_split) grouping_sums_ari{A}{32}*(1+epsilon_split)];
                else
                    disp("centroid number error");
                end
            else
                centroids2{A} = [grouping_sums2_ari{A}];
            end
            
            D2{A} = disteu(matching_array_ari{A},centroids2{A});
            D2b{A} = (1/length(D2{A}))*sum(min(D2{A}));
            [~, index{A}] = min(transpose(D2{A}));
            for check_ari = 1:current_n_centroids{A}
                grouping_data_ari{A}{check_ari} = {};
                for check_ari2 = 1:length(index{A})
                    if index{A}(:,check_ari2) == check_ari
                        grouping_data_ari{A}{check_ari} = [grouping_data_ari{A}{check_ari}, matching_array_ari{A}(:,check_ari2)];
                        grouping_data2_ari{A}{check_ari} = cell2mat(grouping_data_ari{A}{check_ari});
                    else
                        grouping_data_ari{A}{check_ari} = [grouping_data_ari{A}{check_ari}];
                        grouping_data2_ari{A}{check_ari} = cell2mat(grouping_data_ari{A}{check_ari});
                    end
                end
            end
            grouping_sums2_ari{A} = cell(1, current_n_centroids{A});
            grouping_sums_ari{A} = cell(1, current_n_centroids{A});
            for check_ari3 = 1:current_n_centroids{A}
                if isempty(grouping_data2_ari{A}{check_ari3})
                    grouping_sums_ari{A}{check_ari3} = [];
                    grouping_sums2_ari{A} = cell2mat(grouping_sums_ari{A});
                else
                    grouping_sums_ari{A}{check_ari3} = mean(grouping_data2_ari{A}{check_ari3},2);
                    grouping_sums2_ari{A} = cell2mat(grouping_sums_ari{A});
                end
            end

            D3{A} = disteu(matching_array_ari{A},grouping_sums2_ari{A});
            D3b{A} = (1/length(mfccs{A}(2,:)))*sum(min(D3{A}));
            distance_key2{A} = (D2b{A}-D3b{A})/D3b{A};
            l=l+1;
        end
        % fprintf("Number of iterations: %d\n", l-1);
        ari = ari+1;
        current_n_centroids{A} = 2^ari;
    end
% this is to view the centroids in 2D
    test_plot{A} = grouping_sums2_ari{A}(1:2,:);
    test_plot2{A} = mfccs{A}(1:2,:);
    figure(A+7);
    scatter(test_plot2{A}(1,:),test_plot2{A}(2,:),'filled','red');
    hold on
    scatter(test_plot{A}(1,:),test_plot{A}(2,:),'filled','green');
    hold off
end
