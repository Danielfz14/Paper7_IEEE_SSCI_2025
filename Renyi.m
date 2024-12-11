
function [Fitness] = Renyi(th1,th2,H2D,level)
alpha = 0.5;

for i = 1:level
    if i==1
        pk(i) = sum(sum(H2D(1:th2(i),1:th1(i))));
        Fitness(i) = (1/(1-alpha)).*log(sum(sum((H2D(1:th2(i),1:th1(i))/pk(i)).^(alpha))));
        Matriz = (H2D(1:th2(i),1:th1(i))/pk(i)).^(alpha);
    else
        pk(i) = sum(sum(H2D(th2(i-1)+1:th2(i),th1(i-1)+1:th1(i))));
        Fitness(i) = (1/(1-alpha)).*log(sum(sum((H2D(th2(i-1)+1:th2(i),th1(i-1)+1:th1(i))/pk(i)).^(alpha))));
        Matriz = (H2D(th2(i-1)+1:th2(i),th1(i-1)+1:th1(i))/pk(i)).^(alpha);
    end
    
    
end

Fitness = nansum(Fitness);

end

        
        