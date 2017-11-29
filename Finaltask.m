m1=[0 3]';
m2=[2 1]';
c1=[2 1;1 2];
c2=[1 0;0 1];
numgrid=50;
xrange=linspace(-6,6,numgrid);
yrange=linspace(-6,6,numgrid);
pw1x=zeros(numgrid,numgrid);
for i=1:numgrid
    for j=1:numgrid
       x=[xrange(i) yrange(j)]';
       w1(i,j)=mvnpdf(x,m1,c1)+0.2;
       w2(i,j)=mvnpdf(x,m2,c2)+0.2;
       pxw1=mvnpdf(x,m1,c1);
       pxw2=mvnpdf(x,m2,c2);
       pw1x(i,j)=pxw1*0.5/(pxw1*0.5+pxw2*0.5);
    end
end
% pw1x=(pw1x >0.5)+0;
% pw2x=(pw1x <0.5)+0;
% figure(1);
% surface(xrange,yrange,w1,pw1x);
% surface(xrange,yrange,w2,pw1x);
% set(gca,'ztick',[])

hold on;
syms x1 x2;
g1=-0.5*([x1;x2]-m1)'*inv(c1)*([x1;x2]-m1)-log(2*pi)-0.5*log(det(c1));
g2=-0.5*([x1;x2]-m2)'*inv(c2)*([x1;x2]-m2)-log(2*pi)-0.5*log(det(c2));
g=g1-g2
p=ezplot(g);
set(p,'Color','red');
set(p,'LineWidth',2);
x1=mvnrnd(m1,c1,200);
x2=mvnrnd(m2,c2,200);
plot(x1(:,1),x1(:,2),'bx',x2(:,1),x2(:,2),'ro');
% mymap=[1,0,0;0,1,0];
% colormap(mymap);
xlabel('X1','fontsize',16);
ylabel('X2','fontsize',16);
% title('Bayesian decision boundary with C1!=C2','fontsize',20)
hold off;


% %neuron net work
X=[x1;x2];
Y=ones(400,1);
Y(201:400)=0;
[net] = feedforwardnet(40);
[net] = train(net,X',Y');
% [output] = net(X');
% netresults=zeros(numgrid,numgrid);
x=zeros(numgrid*numgrid,2);
iter=1;
for i=1:numgrid
    for j=1:numgrid
%         x=[xrange(i) yrange(j)];
%         netresults(i,j)= net(x');
        x(iter,:)= [xrange(i) yrange(j)];
        iter=iter+1;
    end
end
netresults=net(x');
netresults=reshape(netresults,50,50);
netresults=netresults;
figure(2);
contour(xrange,yrange,netresults,[0.1 0.5 1],'ShowText','on');
hold on;
plot(x1(:,1),x1(:,2),'bx',x2(:,1),x2(:,2),'ro');