clear; 
clc; 
close all;
%  list audio train files
audios = {'Zero_train1.wav','Zero_train2.wav','Zero_train3.wav','Zero_train4.wav','Zero_train5.wav','Zero_train6.wav','Zero_train7.wav','Zero_train8.wav','Zero_train9.wav','Zero_train10.wav','Zero_train11.wav'};
% audios = {'student_zero_train1.wav', 'student_zero_train2.wav', 'student_zero_train3.wav', 'student_zero_train4.wav', 'student_zero_train6.wav', 'student_zero_train7.wav', 'student_zero_train8.wav', 'student_zero_train9.wav', 'student_zero_train10.wav', 'student_zero_train11.wav', 'zero1train.wav', 'yes2train.wav', 'no3train.wav'};
% audios = {'student_zero_train1.wav', 'student_zero_train2.wav', 'student_zero_train3.wav', 'student_zero_train4.wav', 'student_zero_train6.wav', 'student_zero_train7.wav', 'student_zero_train8.wav', 'student_zero_train9.wav', 'student_zero_train10.wav', 'student_zero_train11.wav', 'student_zero_train12.wav', 'student_zero_train13.wav', 'student_zero_train14.wav', 'student_zero_train15.wav', 'student_zero_train16.wav', 'student_zero_train17.wav', 'student_zero_train18.wav', 'student_zero_train19.wav'};
% audios = {'student_twelve_train1.wav', 'student_twelve_train2.wav', 'student_twelve_train3.wav', 'student_twelve_train4.wav', 'student_twelve_train6.wav', 'student_twelve_train7.wav', 'student_twelve_train8.wav', 'student_twelve_train9.wav', 'student_twelve_train10.wav', 'student_twelve_train11.wav', 'student_twelve_train12.wav', 'student_twelve_train13.wav', 'student_twelve_train14.wav', 'student_twelve_train15.wav', 'student_twelve_train16.wav', 'student_twelve_train17.wav', 'student_twelve_train18.wav', 'student_twelve_train19.wav'};

% list audio test files
audio_test = {'Zero_test1.wav','Zero_test2.wav','Zero_test3.wav','Zero_test4.wav','Zero_test5.wav','Zero_test6.wav','Zero_test7.wav','Zero_test8.wav',};
% audio_test = {'student_zero_test1.wav', 'student_zero_test2.wav', 'student_zero_test3.wav', 'student_zero_test4.wav', 'student_zero_test6.wav', 'student_zero_test7.wav', 'student_zero_test8.wav', 'student_zero_test9.wav', 'student_zero_test10.wav', 'student_zero_test11.wav', 'zero1test.wav', 'yes2test.wav', 'no3test.wav'};
% audio_test = {'student_zero_test1.wav', 'student_zero_test2.wav', 'student_zero_test3.wav', 'student_zero_test4.wav', 'student_zero_test6.wav', 'student_zero_test7.wav', 'student_zero_test8.wav', 'student_zero_test9.wav', 'student_zero_test10.wav', 'student_zero_test11.wav', 'student_zero_test12.wav', 'student_zero_test13.wav', 'student_zero_test14.wav', 'student_zero_test15.wav', 'student_zero_test16.wav', 'student_zero_test17.wav', 'student_zero_test18.wav', 'student_zero_test19.wav'};
% audio_test = {'student_twelve_test1.wav', 'student_twelve_test2.wav', 'student_twelve_test3.wav', 'student_twelve_test4.wav', 'student_twelve_test6.wav', 'student_twelve_test7.wav', 'student_twelve_test8.wav', 'student_twelve_test9.wav', 'student_twelve_test10.wav', 'student_twelve_test11.wav', 'student_twelve_test12.wav', 'student_twelve_test13.wav', 'student_twelve_test14.wav', 'student_twelve_test15.wav', 'student_twelve_test16.wav', 'student_twelve_test17.wav', 'student_twelve_test18.wav', 'student_twelve_test19.wav'};

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
N = 512; %frame and fft length
M = 200; %overlap
D = cell(length(audios),1);

% load and apply stft on audios
% s: 3-D array
% f: vector samples
% t: vector Lu
% y: audio data matrix
for A = 1:length(audios)
    fprintf("Crunching numbers for training speaker number %d out of %d \n", A, length(audios))
    [y, fs] = audioread(audios{A}); 

%     % normalize the amplitude of each signal to 1
%     y = y / max(abs(y));
% 
%     % Zero Pad the signal if it is shorter than the longest signal
%     if length(y) < max_length
%         y = [y; zeros(max_length - length(y), 1)];
%     end
    
    % apply stft
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
    n_centroids = 32;
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
% % this is to view the centroids in 2D
%     test_plot{A} = grouping_sums2_ari{A}(1:2,:);
%     test_plot2{A} = mfccs{A}(1:2,:);
%     figure(A);
%     scatter(test_plot2{A}(1,:),test_plot2{A}(2,:),'filled','red');
%     hold on
%     scatter(test_plot{A}(1,:),test_plot{A}(2,:),'filled','green');
%     hold off
end

% We are done with training, we will now do testing

% parameters for testing stft
N_t = 512; %frame and fft length
M_t = 200; %overlap

for B = 1:length(audio_test)
    [y_t, fs_t] = audioread(audio_test{B});

%     % optional notch filter
%     y_t = bandstop(y_t, [5000, 5500], fs, 'Steepness', 0.85, 'StopbandAttenuation', 60); 
%     y_t = y_t / max(abs(y_t));
%     % zero pad the signal if it is shorter than the longest signal
%     if length(y_t) < max_length
%         y_t = [y_t; zeros(max_length - length(y_t), 1)];
%     end

    % apply stft
    [s_t, f_t, t_t] = stft(y_t, fs_t, 'Window', hamming(N_t), 'OverlapLength', M_t, 'FFTLength', N_t);

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
        d_t{B}{A_2} = disteu(matching_array_t{B},grouping_sums2_ari{A_2});

        for A_3 = 1:length(d_t{B}{A_2})
            minimum_distance{B}{A_2}{A_3} = min(d_t{B}{A_2}(A_3,:));
        end   
        average_distance{B}{A_2} = mean(cell2mat(minimum_distance{B}{A_2}));
    end
    [min_values{B}, min_indices{B}] = min(cell2mat(average_distance{B}));
end

success_rate = 0;
for C = 1:length(audio_test)
    final_answer = sprintf('test speaker %d should be with trainee speaker %d.',C,cell2mat(min_indices(:,C)));
    disp(final_answer)
    if C == cell2mat(min_indices(:,C))
        success_rate = success_rate+1;
    else
        success_rate = success_rate;
    end
end

final_success_rate = sprintf('the success rate is %d/%d',success_rate,length(audio_test));
disp(final_success_rate);



