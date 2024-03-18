clear; 
clc; 
close all;
%  list audio train files
audios = {'Zero_train1.wav','Zero_train2.wav','Zero_train3.wav','Zero_train4.wav','Zero_train6.wav','Zero_train7.wav','Zero_train8.wav','Zero_train9.wav','Zero_train10.wav','Zero_train11.wav','Zero_train12.wav','Zero_train13.wav','Zero_train14.wav','Zero_train15.wav','Zero_train16.wav','Zero_train17.wav','Zero_train18.wav','Zero_train19.wav'};
% list audio test files
audio_test = {'Zero_test1.wav','Zero_test2.wav','Zero_test3.wav','Zero_test4.wav','Zero_test6.wav','Zero_test7.wav','Zero_test8.wav','Zero_test9.wav','Zero_test10.wav','Zero_test11.wav','Zero_test12.wav','Zero_test13.wav','Zero_test14.wav','Zero_test15.wav','Zero_test16.wav','Zero_test17.wav','Zero_test18.wav','Zero_test19.wav'};

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
N = 256; %frame and fft length
M = 100; %overlap
D = cell(length(audios),1);

% load and apply stft on audios
% s: 3-D array
% f: vector samples
% t: vector Lu
% y: audio data matrix
for A = 1:length(audios)
    fprintf("A is: %d\n", A)
    [y, fs] = audioread(audios{A});
    
    % Plot the signal in the time domain
    %t = (0:length(y) - 1) / fs; 
    %figure; 
    %plot(t, y);
    %title(['Time-Domain Signal of ' audios{A}]);
    %xlabel('Time (seconds)');
    %ylabel('Amplitude');

    % normalize the amplitude of each signal to 1
    y = y / max(abs(y));

    % Zero Pad the signal if it is shorter than the longest signal
    if length(y) < max_length
        y = [y; zeros(max_length - length(y), 1)];
    end
    
    % Plot the normalized signal in the time domain
    %t = (0:length(y) - 1) / fs; 
    %figure;
    %plot(t, y);
    %title(['Normalized Time-Domain Signal of ' audios{A}]);
    %xlabel('Time (seconds)');
    %ylabel('Normalized Amplitude');

    % apply stft
    [s, f, t] = stft(y, fs, 'Window', hamming(N), 'OverlapLength', M, 'FFTLength', N);

    % apply melfb
    K = 20; 
    m = melfb(K, N, fs);
    
    % Plot melfb responses
    %figure;
    %plot(linspace(0, fs/2, size(m,2)), m');
    %title(['melfb responses' audios{A}]);
    %xlabel('Frequency (Hz)');
    %ylabel('Amplitude');
        
    % calculate the spectrum(before)
    Spectrum = abs(s);

    % apply melfb to the spectrum(after)
    melSpectrum = m * Spectrum(1:size(m, 2), :);

    % Plot the original spectrum
    %figure;
    %surf(t, f, 20*log10(Spectrum), 'EdgeColor', 'none');
    %axis tight;
    %view(0, 90);
    %title(['Original Spectrum' audios{A}]);
    %xlabel('Time (s)');
    %ylabel('Frequency (Hz)');
    %colorbar;

    % Plot the spectrum after melfb
    %figure;
    %surf(t, linspace(0, fs/2, K), 20*log10(melSpectrum), 'EdgeColor', 'none');
    %axis tight;
    %view(0, 90);
    %title(['Mel Spectrum' audios{A}]);
    %xlabel('Time (s)');
    %ylabel('Frequency (Hz)');
    %colorbar;

    % apply MFCC on the Spectrum
    mfccs{A} = mfcc_test(melSpectrum, K);

    %figure;
    %for k = 1:K
    %    subplot(K, 1, k);
    %    plot(t, mfccs{A}(k, :));
    %    if k == 1
    %    title('MFCCs');
    %    end
    %end

    % delete the first K of mfccs
    mfccs{A}(1, :) = []; 

    % Plot the 2nd and 3rd MFCC coefficients
    figure;
    scatter(mfccs{A}(1, :), mfccs{A}(2, :), 'filled','red'); 
    xlabel('Second MFCC Coefficient');
    ylabel('Third MFCC Coefficient');
    title('2D Plot of MFCCs');

    % next we use LGB to find clusters
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
    n_centroids = 80;
    matching_array_ari{A} = mfccs{A};
    epsilon1 = 0.01;
    ari = 1;
    current_n_centroids{A} = 2;
    % while centroids2 < size(1,centroids3)
    
    D2b{A}=[9999];
    D3b{A}=[9999];
    while current_n_centroids{A} <= n_centroids
        fprintf("number of current centroids is %d\n", current_n_centroids{A});
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
                    % cell2mat(grouping_sums_ari{A})
                end
            end

            D3{A} = disteu(matching_array_ari{A},grouping_sums2_ari{A});
            D3b{A} = (1/length(mfccs{A}(2,:)))*sum(min(D3{A}));
            distance_key2{A} = (D2b{A}-D3b{A})/D3b{A}

            l=l+1;
            % delay(1)
        end
        fprintf("Number of iterations: %d\n", l-1);

%         if ari == 1
%             d_minus = 99999*ones(length(D2{A}),1);
%             d_zero = sum(D2{A})/length(D2{A});
%         else
%             
%         end
% D{A} = [D{A}; (1/length(mfccs{A}(2,:)))*sum(min(disteuclid{A}))];
% epsilon{A} = (D{A}(l-1)-D{A}(l))/D{A}(l);
%         distance_key = (d_minus - d_zero)/d_zero
%         while distance_key > epsilon1
        ari = ari+1;
        current_n_centroids{A} = 2^ari;

    end
    % this is to view the centroids in 2D
    %figure;
    %test_plot{A} = grouping_sums2_ari{A}(1:2,:);
    %test_plot2{A} = mfccs{A}(1:2,:);
    %scatter(test_plot2{A}(1,:),test_plot2{A}(2,:),'filled','red');
    %xlabel('Second MFCC Coefficient');
    %ylabel('Third MFCC Coefficient');
    %title('2D Plot of MFCCs and centroids');
    %hold on
    %scatter(test_plot{A}(1,:),test_plot{A}(2,:),'filled','green');
    %hold off
end
%% Old Section
%     m = 1;
%     e = [];
%     s1 = cell(n_centroids,1);
%     if n_centroids == 2
%         centroids{A} = [first_centroid_new{A}*(1-epsilon_split) first_centroid_new{A}*(1+epsilon_split)];
%     elseif n_centroids == 4
%         centroids{A} = [first_centroid_new{A}*(1-epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split) first_centroid_new{A}*(1+epsilon_split)^2];
%     elseif n_centroids == 8
%         centroids{A} = [first_centroid_new{A}*(1-epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^2 first_centroid_new{A}*(1+epsilon_split)^3];
%     elseif n_centroids == 16
%         centroids{A} = [first_centroid_new{A}*(1-epsilon_split)^4 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^3  first_centroid_new{A}*(1+epsilon_split)^4];
%     elseif n_centroids == 32
%         centroids{A} = [first_centroid_new{A}*(1-epsilon_split)^5 first_centroid_new{A}*(1-epsilon_split)^4*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^4*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^4*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^4*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^4*(1+epsilon_split) first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^3*(1+epsilon_split)^2 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)^2*(1+epsilon_split)^3 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^4 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^4 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^4 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^4 first_centroid_new{A}*(1-epsilon_split)*(1+epsilon_split)^4 first_centroid_new{A}*(1+epsilon_split)^5];
%     % elseif n_centroids == 64
%     %     centroids{A} = [first_centroid_new*(1-epsilon_split)^6 first_centroid_new*(1-epsilon_split)^5*(1+epsilon_split) first_centroid_new*(1-epsilon_split)^5*(1+epsilon_split) first_centroid_new*(1-epsilon_split)^5*(1+epsilon_split) first_centroid_new*(1-epsilon_split)^5*(1+epsilon_split) first_centroid_new*(1-epsilon_split)^5*(1+epsilon_split) first_centroid_new*(1-epsilon_split)^5*(1+epsilon_split) first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^4*(1+epsilon_split)^2 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^3*(1+epsilon_split)^3 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)^2*(1+epsilon_split)^4 first_centroid_new*(1-epsilon_split)*(1+epsilon_split)^5 first_centroid_new*(1-epsilon_split)*(1+epsilon_split)^5 first_centroid_new*(1-epsilon_split)*(1+epsilon_split)^5 first_centroid_new*(1-epsilon_split)*(1+epsilon_split)^5 first_centroid_new*(1-epsilon_split)*(1+epsilon_split)^5 first_centroid_new*(1-epsilon_split)*(1+epsilon_split)^5 first_centroid_new*(1+epsilon)^6];
%     else
%         disp('reduce number of centroids.  make sure number of centroids is power of 2.  computer cannot handle this many')
%     end
% 
% 
% %     for m = 1:n_centroids
% %         centroids{A} = [first_centroid_new{A}+epsilon first_centroid_new{A} - epsilon]
% %     end
%     while m <= n_centroids
%         % centroid{A}
%          % e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids)]; % for 2D space
%          e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids)];
%          m = m+1;
%     end
%     e = real(e) + imag(e);
%     e = e*((max(mfccs{A}(2,:)) - min(mfccs{A}(2,:)))/4); %scaling e. The way we scale e can be adjusted
%    
%     %% updated find new centroids
% 
%     centroid{A} = e + [first_centroid_x{A} first_centroid_y{A} first_centroid_z1{A} first_centroid_z2{A} first_centroid_z3{A} first_centroid_z4{A} first_centroid_z5{A} first_centroid_z6{A} first_centroid_z7{A} first_centroid_z8{A} first_centroid_z9{A} first_centroid_z10{A} first_centroid_z11{A} first_centroid_z12{A} first_centroid_z13{A} first_centroid_z14{A} first_centroid_z15{A} first_centroid_z16{A} first_centroid_z17{A}];
%     D{A} = [];
%     D{A} = [999999]; % 9 is chosen as a random number that is greater than 0.01, but not too big
%     l = 2;
%     epsilon{A} = 9; % 9 is chosen as a random number that is greater than 0.01
%     while abs(epsilon{A}) > 0.01
%         % for j = 1:length(mfccs{A}(1,:))
%             matching_array{A} = mfccs{A};
%             % matching_array{A}(3:end,:) = []; % Delete this when we want to leave 2D space
%             grouping_data{A} = cell(n_centroids,1);
%             grouping_data2{A} = cell(n_centroids,1);
%             grouping_sums{A} = cell(n_centroids,19);
%             
%             if l == 2
%                 % disteuclid{A} = disteu(matching_array{A},transpose(centroid{A}));
%                 disteuclid{A} = disteu(matching_array{A},centroids{A});
%             else
%                 disteuclid{A} = disteu(matching_array{A},transpose(s_joey{A}));
%             end
%             see = transpose(disteuclid{A});
%             D{A} = [D{A}; (1/length(mfccs{A}(2,:)))*sum(min(disteuclid{A}))];
%             epsilon{A} = (D{A}(l-1)-D{A}(l))/D{A}(l);
%             if epsilon{A} > 0.01
%                 [~, index{A}] = min(transpose(disteuclid{A}));
%                 
%                 for check = 1:n_centroids
%                     grouping_data{A}{check} = {};
%                     
%                     for check2 = 1:length(index{1})
%                         
%                         if index{A}(:,check2) == check
%                             
%                             grouping_data{A}{check} = [grouping_data{A}{check}, matching_array{A}(:,check2)];
%                             grouping_data2{A}{check} = cell2mat(grouping_data{A}{check});
%                         else
%                             grouping_data{A}{check} = [grouping_data{A}{check}];
%                             grouping_data2{A}{check} = cell2mat(grouping_data{A}{check});
%                         end
%                         
%                     end
%                 end
%                 
%                 for check3 = 1:n_centroids
%                     if length(grouping_data2{A}{check3}) == 0
%                     % if isempty(grouping_data{A}{check3})
%                         grouping_sums{A}{check3,1} = [];
%                         grouping_sums{A}{check3,2} = [];
%                         grouping_sums{A}{check3,3} = [];
%                         grouping_sums{A}{check3,4} = [];
%                         grouping_sums{A}{check3,5} = [];
%                         grouping_sums{A}{check3,6} = [];
%                         grouping_sums{A}{check3,7} = [];
%                         grouping_sums{A}{check3,8} = [];
%                         grouping_sums{A}{check3,9} = [];
%                         grouping_sums{A}{check3,10} = [];
%                         grouping_sums{A}{check3,11} = [];
%                         grouping_sums{A}{check3,12} = [];
%                         grouping_sums{A}{check3,13} = [];
%                         grouping_sums{A}{check3,14} = [];
%                         grouping_sums{A}{check3,15} = [];
%                         grouping_sums{A}{check3,16} = [];
%                         grouping_sums{A}{check3,17} = [];
%                         grouping_sums{A}{check3,18} = [];
%                         grouping_sums{A}{check3,19} = [];
% 
%                     else
%                         grouping_sums{A}{check3,1} = sum(grouping_data2{A}{check3}(1,:));
%                         grouping_sums{A}{check3,2} = sum(grouping_data2{A}{check3}(2,:));
%                         grouping_sums{A}{check3,3} = sum(grouping_data2{A}{check3}(3,:));
%                         grouping_sums{A}{check3,4} = sum(grouping_data2{A}{check3}(4,:));
%                         grouping_sums{A}{check3,5} = sum(grouping_data2{A}{check3}(5,:));
%                         grouping_sums{A}{check3,6} = sum(grouping_data2{A}{check3}(6,:));
%                         grouping_sums{A}{check3,7} = sum(grouping_data2{A}{check3}(7,:));
%                         grouping_sums{A}{check3,8} = sum(grouping_data2{A}{check3}(8,:));
%                         grouping_sums{A}{check3,9} = sum(grouping_data2{A}{check3}(9,:));
%                         grouping_sums{A}{check3,10} = sum(grouping_data2{A}{check3}(10,:));
%                         grouping_sums{A}{check3,11} = sum(grouping_data2{A}{check3}(11,:));
%                         grouping_sums{A}{check3,12} = sum(grouping_data2{A}{check3}(12,:));
%                         grouping_sums{A}{check3,13} = sum(grouping_data2{A}{check3}(13,:));
%                         grouping_sums{A}{check3,14} = sum(grouping_data2{A}{check3}(14,:));
%                         grouping_sums{A}{check3,15} = sum(grouping_data2{A}{check3}(15,:));
%                         grouping_sums{A}{check3,16} = sum(grouping_data2{A}{check3}(16,:));
%                         grouping_sums{A}{check3,17} = sum(grouping_data2{A}{check3}(17,:));
%                         grouping_sums{A}{check3,18} = sum(grouping_data2{A}{check3}(18,:));
%                         grouping_sums{A}{check3,19} = sum(grouping_data2{A}{check3}(19,:));
%                     end
% 
%                 end
%                 s_joey{A} = [];
%                 for check4 = 1:n_centroids
%                     s_joey{A} = [s_joey{A}; grouping_sums{A}{check4}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+2*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+3*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+4*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+5*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+6*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+7*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+8*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+9*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+10*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+11*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+12*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+13*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+14*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+15*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+16*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+17*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+18*n_centroids}/size(grouping_data{A}{check4},2)];
%                 end
%             else
%                 s_joey{A};
%             end
%         % end
%         l = l+1;
%     end
    % Print plots (only works for 2D)
%      figure;
%      scatter(s_joey{A}(:,1),s_joey{A}(:,2),'filled','green');
%      hold on
%      scatter(mfccs{A}(1,:),mfccs{A}(2,:),'filled','red');
%      title(['centroids' audios{A}]);
%      hold off
     % fprintf('Number of iterations: %d\n', l-2);

%% End of old section, tow into testing section
% We are done with training, we will now do testing

% parameters for testing stft
N_t = 256; %frame and fft length
M_t = 100; %overlap

for B = 1:length(audio_test)
    [y_t, fs_t] = audioread(audio_test{B});
    
    %apply notch filter
    y_t = bandstop(y_t, [10000, 10100], fs, 'Steepness',0.85, 'StopbandAttenuation',60);
        
    y_t = y_t / max(abs(y_t));
    % zero pad the signal if it is shorter than the longest signal
    if length(y_t) < max_length
        y_t = [y_t; zeros(max_length - length(y_t), 1)];
    end

    % apply stft
    [s_t, f_t, t_t] = stft(y_t, fs_t, 'Window', hamming(N_t), 'OverlapLength', M, 'FFTLength', N);

    % apply melfb
    K_t = 20;
    m_t = melfb(K_t, N_t, fs_t);

    % calculate the spectrum (before)
    spectrum_t = abs(s_t);

    % apply melfb to the spectrum (after)
    melSpectrum_t = m_t * spectrum_t(1:size(m_t,2),:);

    % apply MFCC on the Spectrum
    mfccs_t{B} = mfcc_test(melSpectrum_t, K_t);

    % delete the first K of mfccs
    mfccs_t{B}(1,:) = [];
    
    d_t{B} = cell(length(audio_test),1);
    matching_array_t{B} = mfccs_t{B};
    % matching_array_t{B}(3:end,:) = []; % Will delete this when we go to 19 dimensions

    for A_2 = 1:length(audios)
        % find distances

        % d_t{B}{A_2} = disteu(matching_array_t{B},transpose(s_joey{A_2}));
        d_t{B}{A_2} = disteu(matching_array_t{B},grouping_sums2_ari{A_2});

        for A_3 = 1:length(d_t{B}{A_2})
            minimum_distance{B}{A_2}{A_3} = min(d_t{B}{A_2}(A_3,:));
        end
       
        average_distance{B}{A_2} = mean(cell2mat(minimum_distance{B}{A_2}));
    end
    
    [min_values{B}, min_indices{B}] = min(cell2mat(average_distance{B}));

end

for C = 1:length(audio_test)
    final_answer = sprintf('test speaker %d should be with trainee speaker %d.',C,cell2mat(min_indices(:,C)));
    disp(final_answer)
end


