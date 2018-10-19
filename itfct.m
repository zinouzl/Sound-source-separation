%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Transformée de Fourier à Court-Terme Inverse %
%               with MATLAB Implementation             %
%                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, t] = itfct(stft, h, nfft, fs)

% function: [x, t] = itfct(stft, wlen, h, wlen)
% stft - STFT matrix (time across columns, freq across rows)
% h - hop size
% nfft - number of FFT points
% fs - sampling frequency, Hz
% x - signal in the time domain
% t - time vector, s

% estimate the length of the signal
coln = size(stft, 2);
xlen = nfft + (coln-1)*h;
x = zeros(1, xlen);
x2 = x;

% form a periodic hamming window
win = hamming(nfft, 'periodic');

% perform IFFT and weighted-OLA
if rem(nfft, 2)                     % odd nfft excludes Nyquist point
    for b = 0:h:(h*(coln-1))
        % extract FFT points
        X = stft(:, 1 + b/h);
        X = [X; conj(X(end:-1:2))];
        
        % IFFT
        xprim = real(ifft(X));
        
        % weighted-OLA
        x((b+1):(b+nfft)) = x((b+1):(b+nfft)) + (xprim.*win)';
    end
else                                % even nfft includes Nyquist point
    for b = 0:h:(h*(coln-1))
        % extract FFT points
        X = stft(:, 1+b/h);
        X = [X; conj(X(end-1:-1:2))];
        
        % IFFT
        xprim = real(ifft(X));
        
        % weighted-OLA
        x((b+1):(b+nfft)) = x((b+1):(b+nfft)) + (xprim.*win)';
        x2((b+1):(b+nfft)) = x2((b+1):(b+nfft)) + (win.^2)';
    end
end

x = x./x2;

% W0 = sum(win.^2);                   % find W0
% x = x.*h/W0;                        % scale the weighted-OLA

% calculate the time vector
actxlen = length(x);                % find actual length of the signal
t = (0:actxlen-1)/fs;               % generate time vector

end