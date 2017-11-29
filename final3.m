filename = 'testftse.csv';
ftse_data = importdata(filename);
ftse=ftse_data.data;
ftse_index=ftse(:,2);
ftse_vol=ftse(:,3);


%normolization
meanftse=mean(ftse_index);
stdftse=std(ftse_index);
ftse_index=(ftse_index-meanftse)/stdftse;


meanftsevol=mean(ftse_vol);
stdftsevol=std(ftse_vol);
ftse_vol=(ftse_vol-meanftsevol)/stdftsevol;

%train the models;
X=ftse_index;
figure(1)
plot(X);
p=20;
Xtr=X(1:1000);
Xts=X(1001:1264);
Ntr=length(Xtr);
train_sample=zeros(Ntr-p,p);
train_target=zeros(Ntr-p,1);
for i=1:(Ntr-p)
    train_sample(i,:)=Xtr(i:i+p-1)';
    train_target(i)=Xtr(i+p,1);
end
b=ones(Ntr-p,1);
train_sample=[train_sample b];

%add the volume as indicator
voltr=ftse_vol(1:1000);
volts=ftse_vol(1001:1264);
voltr=voltr(p:Ntr-1);

train_sample=[train_sample voltr];

% %linear predictor
% wtr = inv(train_sample'*train_sample)*train_sample'*train_target;
% Nts=length(Xts);
% fhts=zeros(264,1);
% initial=X(1001-p:1000)';
% test_sample=[initial 1];
% for j=1:Nts
%     fhts(j) = test_sample*wtr;
%     test_sample(1:p-1)=test_sample(2:p);
%     test_sample(p)=fhts(j);
% end
% errortslinear=sum((Xts-fhts).^2);
% figure(2);
% plot(fhts);
% hold on;
% plot(Xts);


%neuron net work
[net]=feedforwardnet(40);
[net]=train(net,train_sample',train_target');
Nts=length(Xts);
initial=X(1001-p:1000)';
fhtsann=zeros(264,1);
test_sample=initial;
test_sample=[initial 1];
test_sample=[test_sample ftse_vol(1000)];
for j=1:Nts
    fhtsann(j) = net(test_sample');
    test_sample(1:p-1)=test_sample(2:p);
    test_sample(p)=fhtsann(j);
    test_sample(p+2)=volts(j);
end
errortsANN=sum((Xts-fhtsann).^2);
plot(Xts);
hold on;
plot(fhtsann,'g');
