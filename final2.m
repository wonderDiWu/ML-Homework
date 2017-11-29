figure(1)
plot(T, X);
p=40;
Xtr=X(1:1502);
Xts=X(1502:2001);
Ntr=length(Xtr);
train_sample=zeros(Ntr-p,p);
train_target=zeros(Ntr-p,1);
for i=1:(Ntr-p)
    train_sample(i,:)=Xtr(i:i+p-1)';
    train_target(i)=Xtr(i+p);
end
b=ones(Ntr-p,1);
train_sample=[train_sample b];
%linear predictor
wtr = inv(train_sample'*train_sample)*train_sample'*train_target;
Nts=length(Xts);
fhts=zeros(500,1);
initial=X(1502-p:1501)';
test_sample=[initial 1];
% test_sample=initial;
for j=1:Nts
    fhts(j) = test_sample*wtr;
    test_sample(1:p-1)=test_sample(2:p);
    test_sample(p)=fhts(j);
end
errortslinear=sum((Xts-fhts).^2);
figure(2);
plot(fhts,'r');
hold on;
plot(Xts);

%neuron net work
[net]=feedforwardnet(15);
[net]=train(net,train_sample',train_target');
fhtsann=zeros(500,1);
test_sample=initial;
test_sample=[initial 1];
for j=1:Nts
    fhtsann(j) = net(test_sample');
    test_sample(1:p-1)=test_sample(2:p);
    test_sample(p)=fhtsann(j);
end
errortsANN=sum((Xts-fhtsann).^2);
plot(fhtsann,'g');


%feedback the output
cons=zeros(1000,1);
Ncons=length(cons);
initcons=X(2002-p:2001)';
cons_sample=initial;
cons_sample=[initial 1];
for k=1:Ncons
    cons(k) = net(cons_sample');
    cons_sample(1:p-1)=cons_sample(2:p);
    cons_sample(p)=cons(k);
end
figure(3);
plot(cons);
