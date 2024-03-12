clear; 
clc; 
close all;
%  list audio train files
audios = {'Zero_train1.wav','Zero_train2.wav','Zero_train3.wav','Zero_train4.wav','Zero_train5.wav','Zero_train6.wav','Zero_train7.wav','Zero_train8.wav'};

% parameters for stft
N = 516; %frame and fft length
M = 200; %overlap
dimensions = 2;
D = cell(length(audios),1)

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
    % next we use LGB to find clusters
    % (may also cluster numbers---not just Lu/6)
    first_centroid_x{A} = mean(mfccs{A}(2,:));     first_centroid_y{A} = mean(mfccs{A}(3,:));
    
    % select the number of centroids
    n_centroids = 8;
    m = 1;
    e = [];
    s1 = cell(n_centroids,1);
   
    while m <= n_centroids
         e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids)];
         m = m+1;
    end
    e = real(e) + imag(e);
    e = e*((max(mfccs{A}(2,:)) - min(mfccs{A}(2,:)))/4); %scaling e. The way we scale e can be adjusted
   
    % new centroids
    centroid{A} = e + [first_centroid_x{A} first_centroid_y{A}];
    D{A} = [];
    D{A} = 1000; 
    l = 2;
    epsilon{A} = 9; 
    while epsilon{A} > 0.01 
    
    % match the data points with closest centroid
         matching_array{A} = cat(2, transpose(mfccs{A}(2,:)), transpose(mfccs{A}(3,:)));
         matched_points_dataset1{A} = zeros(size(matching_array, 1),2);
         matched_points_dataset2{A} = zeros(size(matching_array, 1),2);
         for i = 1:length(mfccs{A}(2,:))
             matching_points{A} = matching_array{A}(i,:);
             if l == 2
                 distances{A} = sqrt(sum((centroid{A} - matching_points{A}).^2,2));
             else
                 distances{A} = sqrt(sum((s1{A} - matching_points{A}).^2,2));
             end
             [~, index{A}] = min(distances{A});
        
             matched_points_dataset1{A}(i,:) = matching_points{A};
             if l == 2
                 matched_points_dataset2{A}(i,:) = centroid{A}(index{A},:);
             else
                 matched_points_dataset2{A}(i,:) = s1{A}(index{A},:);
             end
    
         end

         d{A} = distance_algorithm(matched_points_dataset1{A}, matched_points_dataset2{A});
         
         D{A} = [D{A};(1/length(mfccs{A}(2,:)))*sum(d{A})];
         epsilon{A} = (D{A}(l-1)-D{A}(l))/D{A}(l);
         if epsilon{A} > 0.01
             % Next we find the optimal reproduction alphabet
             grouped_data{A} = cell(n_centroids, 1);
        
             unique_pairs{A} = unique(matched_points_dataset2{A}, 'rows');
        
             for i = 1:size(unique_pairs{A}, 1)
                 indices{A} = all(matched_points_dataset2{A} == unique_pairs{A}(i,:),2);
                 grouped_data{A}{i} = matched_points_dataset1{A}(indices{A},:);
             end
       
             % Now we can find the new centroid locations
             % s{A} = [];
             % s{A} = {};
             for i = 1:numel(grouped_data{A})
                 if isempty(grouped_data{A}{i})
                     grouped_data{A}{i} = {};
                 else
                     s1{A} = [s1{A}; sum(grouped_data{A}{i}(:,1))/length(grouped_data{A}{i}) sum(grouped_data{A}{i}(:,2))/length(grouped_data{A}{i})];
                 end
             end
             s1{A}
         else
             s1{A}
         end
         l = l+1;
     end
     figure;
     scatter(s1{A}(:,1),s1{A}(:,2),'kx');
     title(['centroids' audios{A}]);
     fprintf('Number of iterations: %d\n', l-2);
end


% We are done with training, we will now do testing
% list audio test files
audio_test = {'Zero_test1.wav','Zero_test2.wav','Zero_test3.wav','Zero_test4.wav','Zero_test5.wav','Zero_test6.wav','Zero_test7.wav','Zero_test8.wav',};

% parameters for testing stft
N_t = 516; %frame and fft length
M_t = 200; %overlap

for B = 1:length(audio_test)
    [y_t, fs_t] = audioread(audio_test{B});
    [s_t, f_t, t_t] = stft(y_t, fs_t, 'Window', hamming(N_t), 'OverlapLength', M, 'FFTLength', N);

    % apply melfb
    K_t = 20;
    m_t = melfb(K_t, N_t, M_t);

    % calculate the spectrum (before)
    spectrum_t = abs(s_t);

    % apply melfb to the spectrum (after)
    melSpectrum_t = m_t * spectrum_t(1:size(m_t,2),:);

    % apply MFCC on the Spectrum
    mfccs_t{B} = mfcc_test(melSpectrum_t, K_t);

    % delete the first K of mfccs
    mfccs_t{B}(1,:) = [];
    
    d_t{B} = cell(length(audio_test),1);
    
    
    for A_2 = 1:length(audios)
        % find distances

        d_t{B}{A_2} = disteu(mfccs_t{B},transpose(s1{A_2}));
        minimum_distance{B}{A_2} = min(d_t{B}{A_2});
        average_distance{B}{A_2} = average(minimum_distance{B}{A_2})
    end

%     for A_3 = 1:length(audio_test)
%         [min_values{B}, min_indices{B}] = min(average_distance{B}{A_3}, [], 2);

end



