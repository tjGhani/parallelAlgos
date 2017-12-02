function [ratio lung abnormality] = findArea(bwLung,lungs)

    image1 = bwLung;
    image2 = im2bw(lungs, 0.4);
    %imshow([image1(:,:,1) image2(:,:,1)])
    %waitforbuttonpress
    
    areafill = regionprops(image1,'area');
    areafill = cat(1,areafill.Area);
    areafill = sum(areafill);
    lung = areafill;
    
    areaext = regionprops(image2,'area');
    areaext = cat(1,areaext.Area);
    areaext = sum(areaext);
    abnormality = areaext; 
    
    ratio = areaext/areafill;
end