clear; 
clc; 
close all;
%  list audios
audios = {'s1.wav','s2.wav','s3.wav','s4.wav','s5.wav','s6.wav','s7.wav','s8.wav','s9.wav','s10.wav','s11.wav'};
cell_s = cell(length(audios), 1);
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
    [s, f, t] = stft(y, fs, 'Window', hamming(N), 'OverlapLength', M, 'FFTLength', N);

    % apply melfb
    K = 20; 
    m = melfb(K, N, fs);
        
    % calculate the spectrum(before)
    Spectrum = abs(s);

    % apply melfb to the spectrum(after)
    melSpectrum = m * Spectrum(1:size(m, 2), :);

    % apply MFCC on the Spectrum
    mfccs = mfcc_test(melSpectrum, K);

    % delete the first K of mfccs
    mfccs(1, :) = []; 
    % next we use LGB to find clusters
    % (may also cluster numbers---not just Lu/6)
    first_centroid_x = mean(mfccs(2,:));     first_centroid_y = mean(mfccs(3,:));
    
    % select the number of centroids
    n_centroids = 8;
    m = 1;
    e = [];
   
    while m <= n_centroids
         e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids)];
         m = m+1;
    end
    e = real(e) + imag(e);
    e = e*((max(mfccs(2,:)) - min(mfccs(2,:)))/4); %scaling e. The way we scale e can be adjusted
   
    % new centroids
    centroid = e + [first_centroid_x first_centroid_y];

    D = Inf; 
    l = 2;
    epsilon = Inf; 
    while epsilon > 0.01 
    
    % match the data points with closest centroid
         matching_array = cat(2, transpose(mfccs(2,:)), transpose(mfccs(3,:)));
         matched_points_dataset1 = zeros(size(matching_array, 1),2);
         matched_points_dataset2 = zeros(size(matching_array, 1),2);
         for i = 1:length(mfccs(2,:))
             matching_points = matching_array(i,:);
             if l == 2
                 distances = sqrt(sum((centroid - matching_points).^2,2));
             else
                 distances = sqrt(sum((s - matching_points).^2,2));
             end
             [~, index] = min(distances);
        
             matched_points_dataset1(i,:) = matching_points;
             if l == 2
                 matched_points_dataset2(i,:) = centroid(index,:);
             else
                 matched_points_dataset2(i,:) = s(index,:);
             end
    
         end

         d = distance_algorithm(matched_points_dataset1, matched_points_dataset2);
         D = [D;(1/length(mfccs(2,:)))*sum(d)];
         epsilon = (D(l-1)-D(l))/D(l);
         if epsilon > 0.01
             % Next we find the optimal reproduction alphabet
             grouped_data = cell(n_centroids, 1);
        
             unique_pairs = unique(matched_points_dataset2, 'rows');
        
             for i = 1:size(unique_pairs, 1)
                 indices = all(matched_points_dataset2 == unique_pairs(i,:),2);
                 grouped_data{i} = matched_points_dataset1(indices,:);
             end
       
             % Now we can find the new centroid locations
             s = [];
             for i = 1:numel(grouped_data)
                 if isempty(grouped_data{i})
                     grouped_data{i} = {};
                 else
                     s = [s; sum(grouped_data{i}(:,1))/length(grouped_data{i}) sum(grouped_data{i}(:,2))/length(grouped_data{i})];
                 end
             end
             s
         else
             s
         end
         l = l+1;
     end
     cell_s{A} = s;
     figure;
     scatter(s(:,1),s(:,2),'kx');
     title(['centroids' audios{A}]);
     fprintf('Number of iterations: %d\n', l-2);
end
