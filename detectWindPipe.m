function [lungSegmented, whiteLungs, lungsOnBlack] = detectWindPipe(dMed)

    bwScan = dMed == 0;
    bwScan = bwScan == 0;                   % black background, white body, black lungs
    
    regions = regionprops(bwScan, 'Area');    %area of white connected comps 
    areas = cat(1, regions.Area);
    reduce = bwareaopen(bwScan, max(areas)-1);       %remove components with area less than 60000
    reduceInvert = reduce==0;
    whiteLungs = reduceInvert & ~bwareaopen(reduceInvert, 60000) & bwareaopen(reduceInvert, 750); %changed from 200 to 350
    whiteLungs = imclose(whiteLungs, ones(10));
    whiteLungs = imfill(whiteLungs, 'holes');

    % eliminate unwanted segments outside lungs and inside body
    body = imfill(imclose(double(reduceInvert==0),ones(3)));
    label_bwScan = bwlabel(bwScan);

    bwScanProps = regionprops(label_bwScan);
    bodyProps = regionprops(edge(body), 'PixelList');

    borderPixels = cat(1, bodyProps.PixelList);
    centroidPixels = cat(1, bwScanProps.Centroid);

    pixels = cat(1, centroidPixels, borderPixels);
    data = pdist(pixels);

    begin = 1;
    last = 0;
    distCentroid(1) = 0;
    for i=1:length(centroidPixels)
       last = last + length(pixels)-i;
       distCentroid(i) = min(data(begin:last));
       if (distCentroid(i) == 1)
           distCentroid(i) = min(data(begin:(last-1)));
           last = last-1;
       end
       begin = last + 1;
    end
    
    if length(distCentroid)>1 | distCentroid(1)~=0
        ind = find(distCentroid<20 & distCentroid>1);
        if (ind~=0)
            for i = 1:length(ind)
                tempVar(:,:,i) = label_bwScan~=ind(i);
                if i==1
                    tempCombined = tempVar(:,:,i);
                elseif i>1
                    tempCombined = tempCombined.*tempVar(:,:,i);
                end
            end
            whiteLungs = tempCombined.*whiteLungs;
        end
    end
    % end elimination of unwanted segments
    
    %bodyWhite = double(reduceInvert) + dMed;
    lungsOnBlack = double(whiteLungs).*dMed;
    
    boundary = (imdilate(edge(double(whiteLungs)),ones(2)));
    
    gbChannel = dMed;
    gbChannel(boundary) = 0;
    rChannel = dMed;
    rChannel(boundary) = 1;
    
    lungSegmented = cat(3, rChannel, gbChannel, gbChannel);
    %imshow(lungSegmented);

end