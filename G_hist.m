function H2D = G_hist(I,level)

level = level;
fobj = "Renyi"; %MCE_Rec;
Im = I;
[m,n]=size(Im);


%% 2D histogram

% Contrast enhancement
Im_E = adapthisteq(Im,'NumTiles',[8 8],'ClipLimit',0.005);
%Im_E = Im_E*256;
Im_E = im2uint8(Im_E);

% Gaussian filter
% sigma = 1.5;
% Im_E = imgaussfilt(Im,sigma);


% Calculate 2D histogram
[m,n]=size(Im);
fxy=zeros(256,256);
for i=1:m
    for j=1:n
        c=Im(i,j);
        d=(Im_E(i,j));
        fxy(c+1,d+1)=fxy(c+1,d+1)+1;
    end
end
H2D=fxy/(m*n); 
end
