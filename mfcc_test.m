function mfccs = mfcc_test(melSpectrum, K)
    % transmit from 256 points to 20(K) points
    % Ensure the melSpectrum does not contain any zeros since log(0) is undefined
    melSpectrum(melSpectrum == 0) = eps; 
    
    % apply the equation of mfcc
    % log of the melSpectrum
    logMelSpectrum = log(melSpectrum);
    
    % apply DCT to the loged melSpectrum
    mfccs = dct(logMelSpectrum);
    
    % keep the first K coefficients
    % (only use second and third, keep all of coefficients at first)
    mfccs = mfccs(1:K, :);
end
