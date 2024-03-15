clear; 
clc; 
close all;
%  list audio train files
audios = {'Zero_train1.wav','Zero_train2.wav','Zero_train3.wav','Zero_train4.wav','Zero_train5.wav','Zero_train6.wav','Zero_train7.wav','Zero_train8.wav','Zero_train9.wav','Zero_train10.wav','Zero_train11.wav'};

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

    
    % select the number of centroids
    n_centroids = 12;
    m = 1;
    e = [];
    s1 = cell(n_centroids,1);
   
    while m <= n_centroids
         % e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids)]; % for 2D space
         e = [e; cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids) 1i*sin(m*2*pi/n_centroids) cos(m*2*pi/n_centroids)];
         m = m+1;
    end
    e = real(e) + imag(e);
    e = e*((max(mfccs{A}(2,:)) - min(mfccs{A}(2,:)))/4); %scaling e. The way we scale e can be adjusted
   
    %% updated find new centroids

    centroid{A} = e + [first_centroid_x{A} first_centroid_y{A} first_centroid_z1{A} first_centroid_z2{A} first_centroid_z3{A} first_centroid_z4{A} first_centroid_z5{A} first_centroid_z6{A} first_centroid_z7{A} first_centroid_z8{A} first_centroid_z9{A} first_centroid_z10{A} first_centroid_z11{A} first_centroid_z12{A} first_centroid_z13{A} first_centroid_z14{A} first_centroid_z15{A} first_centroid_z16{A} first_centroid_z17{A}];
    D{A} = [];
    D{A} = [9]; % 9 is chosen as a random number that is greater than 0.01, but not too big
    l = 2;
    epsilon{A} = 9; % 9 is chosen as a random number that is greater than 0.01
    while epsilon{A} > 0.01
        for j = 1:length(mfccs{A}(1,:))
            matching_array{A} = mfccs{A};
            % matching_array{A}(3:end,:) = []; % Delete this when we want to leave 2D space
            grouping_data{A} = cell(n_centroids,1);
            grouping_data2{A} = cell(n_centroids,1);
            grouping_sums{A} = cell(n_centroids,19);
            
            if l == 2
                disteuclid{A} = disteu(matching_array{A},transpose(centroid{A}));
            else
                disteuclid{A} = disteu(matching_array{A},transpose(s_joey{A}));
            end
            see = transpose(disteuclid{A});
            D{A} = [D{A}; (1/length(mfccs{A}(2,:)))*sum(min(disteuclid{A}))];
            epsilon{A} = (D{A}(l-1)-D{A}(l))/D{A}(l);
            if epsilon{A} > 0.01
                [~, index{A}] = min(transpose(disteuclid{A}))
                index{A}
                for check = 1:n_centroids
                    grouping_data{A}{check} = {};
                    
                    for check2 = 1:length(index{1})
                        
                        if index{A}(:,check2) == check
                            
                            grouping_data{A}{check} = [grouping_data{A}{check}, matching_array{A}(:,check2)];
                            grouping_data2{A}{check} = cell2mat(grouping_data{A}{check});
                        else
                            grouping_data{A}{check} = [grouping_data{A}{check}];
                            grouping_data2{A}{check} = cell2mat(grouping_data{A}{check});
                        end
                        
                    end
                end
                
                for check3 = 1:n_centroids
                    if length(grouping_data2{A}{check3}) == 0
                    % if isempty(grouping_data{A}{check3})
                        grouping_sums{A}{check3,1} = [];
                        grouping_sums{A}{check3,2} = [];
                        grouping_sums{A}{check3,3} = [];
                        grouping_sums{A}{check3,4} = [];
                        grouping_sums{A}{check3,5} = [];
                        grouping_sums{A}{check3,6} = [];
                        grouping_sums{A}{check3,7} = [];
                        grouping_sums{A}{check3,8} = [];
                        grouping_sums{A}{check3,9} = [];
                        grouping_sums{A}{check3,10} = [];
                        grouping_sums{A}{check3,11} = [];
                        grouping_sums{A}{check3,12} = [];
                        grouping_sums{A}{check3,13} = [];
                        grouping_sums{A}{check3,14} = [];
                        grouping_sums{A}{check3,15} = [];
                        grouping_sums{A}{check3,16} = [];
                        grouping_sums{A}{check3,17} = [];
                        grouping_sums{A}{check3,18} = [];
                        grouping_sums{A}{check3,19} = [];

                    else
                        grouping_sums{A}{check3,1} = sum(grouping_data2{A}{check3}(1,:));
                        grouping_sums{A}{check3,2} = sum(grouping_data2{A}{check3}(2,:));
                        grouping_sums{A}{check3,3} = sum(grouping_data2{A}{check3}(3,:));
                        grouping_sums{A}{check3,4} = sum(grouping_data2{A}{check3}(4,:));
                        grouping_sums{A}{check3,5} = sum(grouping_data2{A}{check3}(5,:));
                        grouping_sums{A}{check3,6} = sum(grouping_data2{A}{check3}(6,:));
                        grouping_sums{A}{check3,7} = sum(grouping_data2{A}{check3}(7,:));
                        grouping_sums{A}{check3,8} = sum(grouping_data2{A}{check3}(8,:));
                        grouping_sums{A}{check3,9} = sum(grouping_data2{A}{check3}(9,:));
                        grouping_sums{A}{check3,10} = sum(grouping_data2{A}{check3}(10,:));
                        grouping_sums{A}{check3,11} = sum(grouping_data2{A}{check3}(11,:));
                        grouping_sums{A}{check3,12} = sum(grouping_data2{A}{check3}(12,:));
                        grouping_sums{A}{check3,13} = sum(grouping_data2{A}{check3}(13,:));
                        grouping_sums{A}{check3,14} = sum(grouping_data2{A}{check3}(14,:));
                        grouping_sums{A}{check3,15} = sum(grouping_data2{A}{check3}(15,:));
                        grouping_sums{A}{check3,16} = sum(grouping_data2{A}{check3}(16,:));
                        grouping_sums{A}{check3,17} = sum(grouping_data2{A}{check3}(17,:));
                        grouping_sums{A}{check3,18} = sum(grouping_data2{A}{check3}(18,:));
                        grouping_sums{A}{check3,19} = sum(grouping_data2{A}{check3}(19,:));
                    end

                end
                s_joey{A} = [];
                for check4 = 1:n_centroids
                    s_joey{A} = [s_joey{A}; grouping_sums{A}{check4}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+2*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+3*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+4*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+5*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+6*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+7*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+8*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+9*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+10*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+11*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+12*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+13*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+14*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+15*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+16*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+17*n_centroids}/size(grouping_data{A}{check4},2) grouping_sums{A}{check4+18*n_centroids}/size(grouping_data{A}{check4},2)];
                end
            else
                s_joey{A};
            end
        end
        l = l+1;
    end
    % Print plots (only works for 2D)
%      figure;
%      scatter(s_joey{A}(:,1),s_joey{A}(:,2),'filled','green');
%      hold on
%      scatter(mfccs{A}(1,:),mfccs{A}(2,:),'filled','red');
%      title(['centroids' audios{A}]);
%      hold off
     fprintf('Number of iterations: %d\n', l-2);
end
%     %% old
% 
%     % new centroids
%     centroid{A} = e + [first_centroid_x{A} first_centroid_y{A}];
%     D{A} = [];
%     D{A} = 1000; 
%     l = 2;
%     epsilon{A} = 9; 
%     while epsilon{A} > 0.01 
%     
%     % match the data points with closest centroid
%          matching_array{A} = cat(2, transpose(mfccs{A}(2,:)), transpose(mfccs{A}(3,:)));
%          matched_points_dataset1{A} = zeros(size(matching_array, 1),2);
%          matched_points_dataset2{A} = zeros(size(matching_array, 1),2);
%          for i = 1:length(mfccs{A}(2,:))
%              matching_points{A} = matching_array{A}(i,:);
%              if l == 2
%                  distances{A} = sqrt(sum((centroid{A} - matching_points{A}).^2,2));
%              else
%                  distances{A} = sqrt(sum((s1{A} - matching_points{A}).^2,2));
%              end
%              [~, index{A}] = min(distances{A});
%         
%              matched_points_dataset1{A}(i,:) = matching_points{A};
%              if l == 2
%                  matched_points_dataset2{A}(i,:) = centroid{A}(index{A},:);
%              else
%                  matched_points_dataset2{A}(i,:) = s1{A}(index{A},:);
%              end
%     
%          end
% 
%          d{A} = distance_algorithm(matched_points_dataset1{A}, matched_points_dataset2{A});
%          
%          D{A} = [D{A};(1/length(mfccs{A}(2,:)))*sum(d{A})];
%          epsilon{A} = (D{A}(l-1)-D{A}(l))/D{A}(l);
%          if epsilon{A} > 0.01
%              % Next we find the optimal reproduction alphabet
%              grouped_data{A} = cell(n_centroids, 1);
%         
%              unique_pairs{A} = unique(matched_points_dataset2{A}, 'rows');
%         
%              for i = 1:size(unique_pairs{A}, 1)
%                  indices{A} = all(matched_points_dataset2{A} == unique_pairs{A}(i,:),2);
%                  grouped_data{A}{i} = matched_points_dataset1{A}(indices{A},:);
%              end
%        
%              % Now we can find the new centroid locations
%              % s{A} = [];
%              % s{A} = {};
%              for i = 1:numel(grouped_data{A})
%                  if isempty(grouped_data{A}{i})
%                      grouped_data{A}{i} = {};
%                  else
%                      s1{A} = [s1{A}; sum(grouped_data{A}{i}(:,1))/length(grouped_data{A}{i}) sum(grouped_data{A}{i}(:,2))/length(grouped_data{A}{i})];
%                  end
%              end
%              s1{A}
%          else
%              s1{A}
%          end
%          l = l+1;
%      end
%      figure;
%      scatter(s1{A}(:,1),s1{A}(:,2),'kx');
%      title(['centroids' audios{A}]);
%      fprintf('Number of iterations: %d\n', l-2);
% end

%% Testing
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
    matching_array_t{B} = mfccs_t{B};
    % matching_array_t{B}(3:end,:) = []; % Will delete this when we go to 19 dimensions

    for A_2 = 1:length(audios)
        % find distances

        d_t{B}{A_2} = disteu(matching_array_t{B},transpose(s_joey{A_2}));

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


