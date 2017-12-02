function [detectedWM whiteMass boundary area] = whiteMassDetection(lungWindow, originalOnBlack)

    detectedWM = 0;
    
    whiteMass = im2bw(originalOnBlack,0.5);
    reduce = bwareaopen(whiteMass, 150);
    body = double(reduce) .* (originalOnBlack);
    boundary = (imdilate(edge(double(body)),ones(1)));
	
	area = regionprops(imfill(boundary, 'holes'), 'Area');
	area = cat(1, area.Area);
	area = sum(area);
	
    if(find(boundary)>0)
        detectedWM = 1;
    end

    gbChannel = lungWindow;
    gbChannel(boundary) = 0;
    rChannel = lungWindow;
    rChannel(boundary) = 1;

    whiteMass = cat(3, rChannel, gbChannel, gbChannel);

end