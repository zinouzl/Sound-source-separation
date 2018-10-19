clear all; close all;

[x fs] = audioread('lead_vocal.wav');
[y,fss] = audioread('sunrise-Mix_Bal_Pan.wav');
Lx = [];

x1 = (x(:,1)+x(:,2))/2;
size(x,1)/fs
y1 = (y(:,1)+y(:,2))/2;
wlen = 1024;
nlap = wlen/2;      % 50% de wlen
nfft = 1024;


% %% TFCT prof
 h = 50;
ma = 1;
k=20*fs;
% tfctr = [];
while (k<=size(x,1))

x2 = x1(ma:1:k);
y2 = y1(ma:1:k);
ma = k+1;
k=k+(20*fs);
[tfct, f, t] = stft(x2, wlen, h, nfft, fs);
% tfctr =[tfctr,tfct] ;
[tfct1, f, t] = stft(y2, wlen, h, nfft, fs);

 V =gpuArray(tfct);
 T = gpuArray(tfct1);

 V = abs(V);
 T = abs(T);
%% Calcul des paramètres : 
S = 10;                   % Nombres de bases
[K N] = size(tfct);
W0 = rand(K,S,'gpuArray')+1;         % Dictionnaire
H0 = rand(S,N,'gpuArray')+1;         % Activation

%bases = 1:S;

F = W0;
H = H0;
%bases = 1:S;
err = 0;
err1 = 0;
O = ones(K,N,'gpuArray');

%figure()
disp('training..')
for i = 1:100
   
    
    H = H .* (F'*( V./(F*H+eps))) ./ (F'*O);
% update dictionaries
    F = F .* ((V./(F*H+eps))*H') ./(O*H');
   % err(i) = norm(F*H - V);
    
%     subplot(4, 4, [6,7,8,10,11,12,14,15,16]);
%     imagesc(t,f,20*log10(V));
%     colorbar
%     colormap(jet)
%     axis xy;
%     
%     subplot(4, 4, [5,9,13]);
%     imagesc(bases,f,20*log10(abs(F)));
%     colormap(jet)
%     axis xy;
% 
%     subplot(4, 4, [2,3,4]);
%     imagesc(t,bases,abs(H));
%     colormap(jet)
%     axis xy;

    %pause();
    
   % disp(['iteration : ' num2str(i)]);
end
%figure()
%plot(err);
for i = 1:size(F,2)
F(:,i) = F(:,i)/sum(F(:,i));
disp('end of training')
end
mu = 100;
mu =gpuArray(mu);
H1 = rand(K,S+30,'gpuArray')+1;
U = rand(S+30,N,'gpuArray')+1;
G = rand(S,N,'gpuArray')+1;
for i = 1:size(H1,2)
H1(:,i) = H1(:,i)/sum(H1(:,i));
disp('start factorization')
end
for i=1:300

   H1 = H1 .* ((T*U')./(2*mu.*(F*(F'*H1))+((F*G+H1*U)*U')));
   
   U = U .* ((H1'*T)./(H1'*(F*G+H1*U)));
   
   G = G .* ((F'*T)./(F'*((F*G)+(H1*U))));
 
  % err1(i) = norm((F*G+H1*U) - T);



%disp(['iteration : ' num2str(i)]);
end
%figure()
%plot(err1)
disp('finishing factoriz')
lamda = angle(tfct);
sound = F * G ;
sound = sound.*exp(1i*lamda);
disp('start gathering');
sound1 = gather(sound);
disp('finishing')
[X, t] = itfct(sound1, h, nfft, fs);
X =  medfilt1(X);
%X = Wiener(X,fs);
Lx=[Lx X];
end
audiowrite('test.wav',Lx,fs);
