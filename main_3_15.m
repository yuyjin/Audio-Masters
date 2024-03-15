clear; 
clc; 
close all;
%  list audio train files
audios = {'s1.wav','s2.wav','s3.wav','t4.wav','t5.wav','t6.wav'};

% parameters for stft
N = 516; %frame and fft length
M = 200; %overlap
dimensions = 2;
D = cell(length(audios),1);

% load and apply stft on audios
% s: 3-D array
% f: vector samples
% t: vector Lu
% y: audio data matrix
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

    figure;
    scatter(mfccs{1}(1,:),mfccs{1}(2,:),'filled','red');
    % next we use LGB to find clusters
    % (may also cluster numbers---not just Lu/6)
    first_centroid_x{A} = mean(mfccs{A}(1,:));
    first_centroid_y{A} = mean(mfccs{A}(2,:));
    
    % select the number of centroids
    n_centroids = 20;
    m = 1;
    e = [];
    s1 = cell(n_centroids,1);
   
    while m <= n_centroids
         e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids)];
         m = m+1;
    end
    e = real(e) + imag(e);
    e = e*((max(mfccs{A}(2,:)) - min(mfccs{A}(2,:)))/4)*0.8; %scaling e. The way we scale e can be adjusted
    
    %% updated find new centroids

    centroid{A} = e + [first_centroid_x{A} first_centroid_y{A}];
    D{A} = [];
    D{A} = [9]; % 9 is chosen as a random number that is greater than 0.01, but not too big
    l = 2;
    epsilon{A} = 9; % 9 is chosen as a random number that is greater than 0.01
    while epsilon{A} > 0.01
        for j = 1:length(mfccs{A}(1,:))
            matching_array{A} = mfccs{A};
            matching_array{A}(3:end,:) = [];
            grouping_data{A} = cell(n_centroids,1);
            grouping_sums{A} = cell(n_centroids,2);
            % if epsilon > 0.01
            if l == 2
                disteuclid{A} = disteu(matching_array{A},transpose(centroid{A}));
            else
                disteuclid{A} = disteu(matching_array{A},transpose(s_joey{A}));
            end
            see = transpose(disteuclid{A});
            D{A} = [D{A}; (1/length(mfccs{A}(2,:)))*sum(min(disteuclid{A}))];
            epsilon{A} = (D{A}(l-1)-D{A}(l))/D{A}(l);
            if epsilon{A} > 0.01
                [~, index{A}] = min(transpose(disteuclid{A}));
                for check = 1:n_centroids
                    lim=1;
                    index{A}
                    for check2 = 1:length(index{1})
                        lim;
                        if index{A}(:,check2) == check
                            grouping_data{A}{check} = [grouping_data{A}{check},matching_array{A}(:,check2)];
                        else
                            grouping_data{A}{check} = [grouping_data{A}{check}];
                        end
                        lim=lim+1;
                    end
                end
                % test = sum(grouping_data{A}{1}(1,:))
                for check3 = 1:n_centroids
                    if isempty(grouping_data{A}{check3})
                        grouping_sums{A}{check3,1} = [];
                        grouping_sums{A}{check3,2} = [];
                    else
                        grouping_sums{A}{check3,1} = sum(grouping_data{A}{check3}(1,:));
                        grouping_sums{A}{check3,2} = sum(grouping_data{A}{check3}(2,:));
                    end

                end
                s_joey{A} = [];
                for check4 = 1:n_centroids
                    s_joey{A} = [s_joey{A}; grouping_sums{A}{check4}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+n_centroids}/size(grouping_data{A}{check4},2)];
                end
            else
                s_joey{A};
            end
        end
        l = l+1;
    end
     figure;
     scatter(s_joey{A}(:,1),s_joey{A}(:,2),'filled','green');
     hold on
     scatter(mfccs{A}(1,:),mfccs{A}(2,:),'filled','red');
     title(['centroids' audios{A}]);
     hold off
     fprintf('Number of iterations: %d\n', l-2);
end
%% Testing
% We are done with training, we will now do testing
% list audio test files
audio_test = {'s1.wav','s2.wav','s3.wav','s4.wav','s5.wav','s6.wav'};
mfccs_t = cell(length(audio_test),1);

% parameters for testing stft
N_t = 516; %frame and fft length
M_t = 200; %overlap
D = cell(length(audio_test),1);

for B = 1:length(audio_test)
    [y_t, fs_t] = audioread(audio_test{B});
    [s_t, f_t, t_t] = stft(y_t, fs_t, 'Window', hamming(N_t), 'OverlapLength', M_t, 'FFTLength', N_t);

    % apply melfb
    K_t = 20;
    m_t = melfb(K_t, N_t, M_t);
        
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

    figure;
    scatter(mfccs_t{1}(1,:),mfccs_t{1}(2,:),'filled','red');
    
    first_centroid_x{B} = mean(mfccs_t{B}(1,:));
    first_centroid_y{B} = mean(mfccs_t{B}(2,:));
    
    % select the number of centroids
    %n_centroids = 8;
    m = 1;
    e = [];
    s1 = cell(n_centroids,1);
   
    while m <= n_centroids
         e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids)];
         m = m+1;
    end
    e = real(e) + imag(e);
    e = e*((max(mfccs_t{B}(2,:)) - min(mfccs_t{B}(2,:)))/4)*0.8; %scaling e. The way we scale e can be adjusted
   
    %% updated find new centroids

    centroid{B} = e + [first_centroid_x{B} first_centroid_y{B}];
    D{B} = [];
    D{B} = [9]; % 9 is chosen as a random number that is greater than 0.01, but not too big
    l = 2;
    epsilon{B} = 9; % 9 is chosen as a random number that is greater than 0.01
    while epsilon{B} > 0.01
        for j = 1:length(mfccs_t{B}(1,:))
            matching_array{B} = mfccs_t{B};
            matching_array{B}(3:end,:) = [];
            grouping_data{B} = cell(n_centroids,1);
            grouping_sums{B} = cell(n_centroids,2);
            % if epsilon > 0.01
            if l == 2
                disteuclid{B} = disteu(matching_array{B},transpose(centroid{B}));
            else
                disteuclid{B} = disteu(matching_array{B},transpose(ss_joey{B}));
            end
            see = transpose(disteuclid{B});
            D{B} = [D{B}; (1/length(mfccs_t{B}(2,:)))*sum(min(disteuclid{B}))];
            epsilon{B} = (D{B}(l-1)-D{B}(l))/D{B}(l);
            if epsilon{B} > 0.01
                [~, index{B}] = min(transpose(disteuclid{B}));
                for check = 1:n_centroids
                    lim=1;
                    index{B};
                    for check2 = 1:length(index{1})
                        lim;
                        if index{B}(:,check2) == check
                            grouping_data{B}{check} = [grouping_data{B}{check},matching_array{B}(:,check2)];
                        else
                            grouping_data{B}{check} = [grouping_data{B}{check}];
                        end
                        lim=lim+1;
                    end
                end
                % test = sum(grouping_data{A}{1}(1,:))
                for check3 = 1:n_centroids
                    if isempty(grouping_data{B}{check3})
                        grouping_sums{B}{check3,1} = [];
                        grouping_sums{B}{check3,2} = [];
                    else
                        grouping_sums{B}{check3,1} = sum(grouping_data{B}{check3}(1,:));
                        grouping_sums{B}{check3,2} = sum(grouping_data{B}{check3}(2,:));
                    end

                end
                ss_joey{B} = [];
                for check4 = 1:n_centroids
                    ss_joey{B} = [ss_joey{B}; grouping_sums{B}{check4}/size(grouping_data{B}{check4},2) grouping_sums{B}{check4+n_centroids}/size(grouping_data{B}{check4},2)];
                end
            else
                ss_joey{B};
            end
        end
        l = l+1;
    end
     figure;
     scatter(ss_joey{B}(:,1),ss_joey{B}(:,2),'filled','green');
     hold on
     scatter(mfccs_t{B}(1,:),mfccs_t{B}(2,:),'filled','red');
     title(['centroids test' audio_test{B}]);
     hold off
end
    

total_min_distances = zeros(length(audio_test), length(audios));

for B = 1:length(audio_test)
    for A = 1:length(audios)
        % Compute distance between each centroid from test and each centroid from train
        distances_each = disteu(transpose(ss_joey{B}), transpose(s_joey{A}));

        % Instead of summing all distances, focus on the minimum or average distances
        total_min_distances(B, A) = mean(min(distances_each, [], 2)); 
    end
end

for B = 1:length(audio_test)
    % Identify the speaker (training set) with the lowest total distance for each test set
    [~, identified_speaker] = min(total_min_distances(B, :));
    fprintf('The identified speaker for test %d is: %d\n', B, identified_speaker);
end
