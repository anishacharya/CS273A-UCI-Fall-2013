phyData = load('C:\MS 1st Year Fall 2013\machine learning\CS273A\Project\data_kddcup04\/phy_train.dat');
X = phyData(1:50000,3:end);
Y = phyData(1:50000,2);
[X Y] = shuffleData(X,Y);
[X1 Xtest y1 ytest] = splitData(X,Y,.75);
l=78;
N=37500;
kernel='rbf';
kpar1=0.1;
kpar2=0;
C=2;
tol=0.001;
steps=100000;
eps=10^(-10);
method=1;
[alpha, w0, w, evals, stp, glob] = SMO2(X1', y1',kernel, kpar1, kpar2, C, tol, steps, eps, method);
X_sup=X1(:,alpha'~=0);
alpha_sup=alpha(alpha~=0)';
y_sup=y1(alpha~=0);

for i=1:N
    t=sum((alpha_sup.*y_sup).*...
        CalcKernel(X_sup',X1(:,i)',kernel,kpar1,kpar2)')-w0;
    if(t>0)
        out_train(i)=1;
    else
        out_train(i)=-1;
    end
end
Pe1=sum(out_train.*y1<0)/length(y1)
