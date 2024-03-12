clear; 
clc; 
close all;
%  list audios
audios = {'Zero_train3.wav'};

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

% Plot the clustered MFCC vectors
% Plot the 2nd and 3rd MFCC coefficients
figure;
scatter(mfccs(2, :), mfccs(3, :), 'filled'); 
xlabel('Second MFCC Coefficient');
ylabel('Third MFCC Coefficient');
title('2D Plot of MFCCs');

% next we use LGB to find clusters
% (may also cluster numbers---not just Lu/6)
first_centroid_x = mean(mfccs(2,:));
first_centroid_y = mean(mfccs(3,:));
figure;
scatter(mfccs(2, :), mfccs(3, :), 'filled'); 
% scatter(first_centroid_x, first_centroid_y)
% scatter([mfccs(2,:),first_centroid_x],[mfccs(3,:),first_centroid_y],['filled','filled'],["r","b"])
xlabel('Second MFCC Coefficient');
ylabel('Third MFCC Coefficient');
title('2D Plot of MFCCs with first centroid');
hold on
scatter(first_centroid_x, first_centroid_y, 'kx');

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
        
%         disp("Grouped Data:");
%         for i = 1:numel(grouped_data)
%             fprintf('Group %d:\n',i);
%             disp(grouped_data{i});
%         end
        
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
hold on
scatter(s(:,1),s(:,2),'kx');
hold off
fprintf('Number of iterations: %d\n', l-2);
