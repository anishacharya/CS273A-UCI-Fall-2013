clear all;
%question 1.a%
data= load('mcycle80.txt');
X=data(:,1);
y=data(:,2);
[X y]=shuffleData(X,y);
[X_tr X_te y_tr y_te]=splitData(X,y,0.75);
%question 1.b%
l=linearRegress(X_tr,y_tr);
l1=linearRegress(X_te,y_te);
xs=[0:0.01:2]';
ys=predict(l,xs);
figure, plot(xs,ys);
hold on;
scatter(X_tr,y_tr);
hold on;
y_tr_Hat=predict(l,X_tr);
y_te_Hat=predict(l1,X_te);
Mse=[(transpose(y_te_Hat-y_te)*(y_te_Hat-y_te))/(0.25*length(X))]; 
Mse1=[(transpose(y_tr_Hat-y_tr)*(y_tr_Hat-y_tr))/(0.75*length(X))];

% %#1.c%
ii=0;
for d=[1, 3, 5, 7, 10, 18]
XtrP = fpoly(X_tr,d, false); % create polynomial features up to given degree, no "1" feature
[XtrP, M,S] = rescale(XtrP); % often a good idea to scale the features
lr = linearRegress( XtrP, y_tr ); % create and train model

xsP= fpoly(xs,d,false);
[xsP] = rescale(xsP,M,S);

ysP=predict(lr,xsP);

ytrP=predict(lr,XtrP);

XteP = fpoly(X_te,d, false); % create polynomial features up to given degree, no "1" feature
[XteP] = rescale(XteP,M,S); % often a good idea to scale the features

yteP=predict(lr,XteP);

figure, scatter(X_tr,y_tr);
hold on;
plot(xs,ysP);
ii=ii+1;
M_se(ii)=[(transpose(yteP-y_te)*(yteP-y_te))/(0.25*length(X))]; 
M_se1(ii)=[(transpose(ytrP-y_tr)*(ytrP-y_tr))/(0.75*length(X))];
% 
end
figure; semilogy([1 3 5 7 10 18],log(M_se),'r-',[1 3 5 7 10 18],log(M_se1));

%#1.d
alpha=logspace(-6,1,15);
ii=0;
for e=1:15
    d=18;
XtrP = fpoly(X_tr,d, false); % create polynomial features up to given degree, no "1" feature
[XtrP, M,S] = rescale(XtrP); % often a good idea to scale the features
lr = linearRegress( XtrP, y_tr,alpha(e) ); % create and train model

xsP= fpoly(xs,d,false);
[xsP] = rescale(xsP,M,S);

ysP=predict(lr,xsP);

ytrP=predict(lr,XtrP);

XteP = fpoly(X_te,d, false); % create polynomial features up to given degree, no "1" feature
[XteP] = rescale(XteP,M,S); % often a good idea to scale the features

yteP=predict(lr,XteP);

figure, scatter(X_tr,y_tr);
hold on;
plot(xs,ysP);
ii=ii+1;
M_se20(ii)=[(transpose(yteP-y_te)*(yteP-y_te))/(0.25*length(X))]; 
M_se21(ii)=[(transpose(ytrP-y_tr)*(ytrP-y_tr))/(0.75*length(X))];
end
figure; semilogy(logspace(-6,1,15),(M_se20),'r-',logspace(-6,1,15),(M_se21));






