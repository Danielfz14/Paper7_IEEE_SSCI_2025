imageDir = '...\Skin_images\';
maskDir = '...\Skin_images_masks\';

imageFiles = dir(fullfile(imageDir, '*.jpg'));
maskFiles = dir(fullfile(maskDir, '*.png'));

precomputedData = struct('H2D', []);

for i = 1:length(imageFiles)
    img = imread(fullfile(imageDir, imageFiles(i).name));
    img_gray = rgb2gray(img);
    
    s1 = 50;
    s2 = 50;
    Icomp = imcomplement(img_gray);
    Se = strel('disk', s1);
    Isc = imopen(Icomp, Se);
    Se2 = strel('disk', s2);
    Ift = imclose(Isc, Se2);
    TH = Icomp - Ift;
    level = graythresh(TH);
    Bin = imbinarize(TH, level);
    img_inpainted = inpaintCoherent(img, Bin);
    gray_img = rgb2gray(img_inpainted);
    
    H2D = G_hist(gray_img, 2);
    precomputedData(i).H2D = H2D;
end

save('precomputed_data.mat', 'precomputedData');
disp('Datos precomputados guardados exitosamente.');
