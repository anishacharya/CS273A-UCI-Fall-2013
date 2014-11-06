function plot2DLinear(obj, Xtrain, Ytrain)
% plot2DLinear(obj, Xtrain,Ytrain)
%
plot a linear classifier (data and decision boundary) when features
Xtrain are 2-dim
%
wts are 1x3, wts(0)+wts(2)*X(1)+wts(3)*X(2)
%
[n,d] = size(Xtrain);
if (d~=2) error('Sorry -- plot2DLinear only works on 2D data...'); end;
u=unique (Ytrain);class0 = find(Ytrain==u(1));
class1= find(Ytrain==u(2));
Xplt = linspace(min(Xtrain(:,1)),max(Xtrain(:,1)),200);
plot(Xtrain(class0,1),Xtrain(class0,2),'bo',...
Xtrain(class1,1),Xtrain(class1,2),'gs',...
Xplt,-obj.wts(1)/obj.wts(3) - obj.wts(2)/obj.wts(3).*Xplt,'r-');
% TODO: Plot each class in a different color
%
along with the linear decision boundary of the predictor
drawnow;
% ensures plot is updated immediately
