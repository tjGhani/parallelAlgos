function [detectedBH blackHole boundary area] = detectBlackHoles(lungWindow, originalOnWhite)

    detectedBH = 0;

    origWhiteBW = im2bw(originalOnWhite,0.05);
    regions = regionprops(~origWhiteBW,'Area');
    blackness = bwareaopen(~(origWhiteBW),300);
    blackAreaRegions = regionprops(blackness,'Centroid','Area');
    blackCentroids = cat(1, blackAreaRegions.Centroid);
    blackAreas = cat(1, blackAreaRegions.Area);

    remove = 0;
    for i = 1:size(blackCentroids,1)
        if blackCentroids(i,1)>200 & blackCentroids(i,1)<300
            remove = i;
        end
    end

    if remove ~= 0
        blackness = bwareaopen(blackness, blackAreas(remove)+1);
    end

    if(find(blackness)>0)
        detectedBH = 1;
    end

    boundary = (imdilate(edge(imclose(double(blackness),ones(4))),ones(2)));
	
	area = regionprops(imfill(boundary, 'holes'), 'Area');
	area = cat(1, area.Area);
	area = sum(area);
	
    gbChannel = lungWindow;
    gbChannel(boundary) = 0;
    rChannel = lungWindow;
    rChannel(boundary) = 1;
    
    blackHole = cat(3, rChannel, gbChannel, gbChannel);
    
end