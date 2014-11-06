//Show the two classes in a scatter plot and verify that one is linearly
//separable.


iris=load('data/iris.txt'); % load the text fil
X = iris(:,1:2); Y=iris(:,end); % get first two fea
XA = X(Y<2,:); YA=Y(Y<2); % get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0); % get class 1 vs 2
[X0 Y0]=find(Y == 0);
[X1 Y1]=find(Y == 1);
[X2 Y2]=find(Y == 2);
XA_0=X(X0,1:2);
XA_1=X(X1,1:2);
scatter(XA_0(:,1),XA_0(:,2),'r');
hold on;
scatter(XA_1(:,1),XA_1(:,2),'b');
XB_1=X(X1,1:2);
XB_2=X(X2,1:2);
figure, scatter(XB_1(:,1),XB_1(:,2),'r');
hold on;
scatter(XB_2(:,1),XB_2(:,2),'b');

///////////
///Build a linear classifier using the basic perceptron algorithm
///////
rand('state',0); randn('state',0); % ensure reproducible
pc = perceptClassify(XA,YA,1e-4,5000); % train
plot2DLinear(pc,XA,YA);
mean( YA ~= predict(pc,XA) ),
figure, plot2DLinear(pc,XA,YA);
rand('state',0); randn('state',0); % ensure reproducible
pc = perceptClassify(XB,YB,1e-4,5000); % train
mean( YB ~= predict(pc,XB) ),
figure, plot2DLinear(pc,XB,YB);


l= linearRegress(XA,YA)
rand('state',0); randn('state',0); % ensure reproducible
pc = perceptClassify(XA,YA,1e-4,5000); % train
mean( YB ~= predict(pc,XA) ),
figure, plot2DLinear(pc,XA,YA);

rand('state',0); randn('state',0); % ensure reproducible
pc = perceptClassify(XB,YB,1e-4,5000); % train
mean( YB ~= predict(pc,XB) ),
figure, plot2DLinear(pc,XB,YB);













