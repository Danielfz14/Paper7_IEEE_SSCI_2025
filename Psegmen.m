function Fobj=Psegmen(x1,x2,x3,x4,im)
warning off
load('precomputed_data.mat')
Th1=[round(x1),round(x2)];
Th2=[round(x3),round(x4)];
fobj = "Renyi";
level1=2;
H2D = precomputedData(im).H2D;
Fobj =-feval(fobj,Th1,Th2,H2D,level1); 
if abs(Fobj)==inf
    Fobj=10000;
end
end

