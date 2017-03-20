%色彩空间KNN

%将图片上的点显示在色彩空间中
clear;
clc;
clf;
src = imread('6.png');
src = imresize(src,128.0/size(src,1));% 归一化大小
RGB = boolean(zeros(255,255,255));% RGB空间（1-255）

%图片框范围
figure(1)
imshow(src);
[x0,y0] = ginput(2);
x1 = max(1,int32(x0));
y1 = max(1,int32(y0));

ii = 1;
for i = x1(1,1):x1(2,1)
    for j = y1(1,1):y1(2,1)
        x(ii,1) = i;%选取区域的坐标
        y(ii,1) = j;%选取区域的坐标
        ii = ii + 1;
    end
end

%RGB空间赋值
for i = 1:size(x,1)
    R(i) = src(y(i),x(i),1);%提取区域RGB
    G(i) = src(y(i),x(i),2);
    B(i) = src(y(i),x(i),3);
    R(i) = max(1,R(i));G(i) = max(1,G(i));B(i) = max(1,B(i));
    RGB(R(i),G(i),B(i)) = 1;    
end

%膨胀RGB空间中的点
dilateSize = 5;%膨胀半径
RGB_dilate = boolean(zeros(255,255,255));
for i = 1:255
    for j = 1:255
        for k = 1:255
            if RGB(i,j,k) == 1
                    Rkernel = max(1,i-dilateSize):min(255,i+dilateSize);%膨胀核
                    Gkernel = max(1,j-dilateSize):min(255,j+dilateSize);
                    Bkernel = max(1,k-dilateSize):min(255,k+dilateSize);
                    RGB_dilate(Rkernel,Gkernel,Bkernel) = ones(size(Rkernel,2),size(Gkernel,2),size(Bkernel,2));
            end
        end
    end
end

%腐蚀RGB空间中的点
erodeSize = 3;%腐蚀半径
RGB_erode = boolean(zeros(255,255,255));
for i = 1+erodeSize:255-erodeSize
    for j = 1+erodeSize:255-erodeSize
        for k = 1+erodeSize:255-erodeSize
            if RGB_dilate(i,j,k) ==1
                Rkernel = i-erodeSize:i+erodeSize;
                Gkernel = j-erodeSize:j+erodeSize;
                Bkernel = k-erodeSize:k+erodeSize;
                if sum(sum(sum(RGB_dilate(Rkernel,Gkernel,Bkernel)))) == (2*erodeSize+1)^3
                    RGB_erode(i,j,k) = 1;
                end
            end
        end
    end
end

%显示RGB空间图像
figure(2)
hold on;
grid on;
for i = 1:255
    for j = 1:255
        for k = 1:255
            if RGB_erode(i,j,k) == 1
                    plot3(i,j,k,'ob');
            end
        end
    end
end
xlabel('R');
ylabel('G');
zlabel('B');
title('RGB');
% axis([0,256,0,256,0,256]);
view(30,30);

%进行图像分割
figure(3)
dst = zeros(size(src,1),size(src,2));
for i = 1:size(src,1)
    for j = 1:size(src,2)
        srcR = max(1,src(i,j,1));srcG = max(1,src(i,j,2));srcB = max(1,src(i,j,3));
        dst(i,j) = RGB_erode(srcR,srcG,srcB);
    end
end
imshow(dst);

