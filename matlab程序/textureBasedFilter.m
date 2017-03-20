%纹理特征
src0=imread('6.png');  
%src = imresize(src,128.0/size(src,1));% 归一化大小
gray = rgb2gray(src0);
%ycbcr = rgb2ycbcr(src);

% L3 = [1,2,1];
% E3 = [-1,0,1];
% S3 = [-1,2,-1];
L3 = [1,4,6,4,1]/16;
%L3 = [-1,2,0,-2,1];
E3 = [-1,-2,0,2,1];
S3 = [1,-1,1,-1,1];

M = zeros(5,5,9);
M(:,:,1) = L3'* L3;
M(:,:,2) = E3'* L3;
M(:,:,3) = S3'* L3;
M(:,:,4) = L3'* E3;
M(:,:,5) = E3'* E3;
M(:,:,6) = S3'* E3;
M(:,:,7) = L3'* S3;
M(:,:,8) = E3'* S3;
M(:,:,9) = S3'* S3;

E = zeros(size(gray,1),size(gray,2),9);
E(:,:,1) = abs(imfilter(gray,M(:,:,1)));
E(:,:,2) = abs(imfilter(gray,M(:,:,2)));
E(:,:,3) = abs(imfilter(gray,M(:,:,3)));
E(:,:,4) = abs(imfilter(gray,M(:,:,4)));
E(:,:,5) = abs(imfilter(gray,M(:,:,5)));
E(:,:,6) = abs(imfilter(gray,M(:,:,6)));
E(:,:,7) = abs(imfilter(gray,M(:,:,7)));
E(:,:,8) = abs(imfilter(gray,M(:,:,8)));
E(:,:,9) = abs(imfilter(gray,M(:,:,9)));
% E(:,:,10) = abs(imfilter(window(:,:,2,s,r),M(:,:,1)))));
% E(:,:,11) = abs(imfilter(window(:,:,3,s,r),M(:,:,1)))));

dst = zeros(size(E,1),size(E,2),3);
% for i=1:9
%     dst(:,:,1) = dst(:,:,1)+E(:,:,i);
%     dst(:,:,2) = dst(:,:,1);
%     dst(:,:,3) = dst(:,:,1);
% end
dst(:,:,1) = E(:,:,5);
dst(:,:,2) = E(:,:,6);
dst(:,:,3) = E(:,:,9);

%dst = uint8(dst/8);
aa = uint8(E(:,:,5)+E(:,:,6)+E(:,:,9));
imshow(aa);


%将图片上的点显示在色彩空间中
src = dst;




%---------正样本-----------
% %图片取点
% figure(1)
% imshow(src);
% [x0,y0] = ginput;
% x = int32(x0);
% y = int32(y0);

%图片框范围
figure(1)
imshow(src);
[x0,y0] = ginput(2);
x1 = int32(x0);
y1 = int32(y0);

ii = 1;
for i = x1(1,1):x1(2,1)
    for j = y1(1,1):y1(2,1)
        x(ii,1) = i;%选取区域的坐标
        y(ii,1) = j;%选取区域的坐标
        ii = ii + 1;
    end
end

%RGB色彩空间中显示
figure(2)
hold on;
grid on;
for i = 1:size(x,1)
    R(i) = src(y(i),x(i),1);%提取区域RGB
    G(i) = src(y(i),x(i),2);
    B(i) = src(y(i),x(i),3);
end
plot3(R,G,B,'ob');
xlabel('R');
ylabel('G');
zlabel('B');
title('RGB');
% axis([0,256,0,256,0,256]);
view(30,30);

%---------负样本-----------
%图片框范围
figure(1)
[x0,y0] = ginput(2);
x1 = int32(x0);
y1 = int32(y0);

ii = 1;
for i = x1(1,1):x1(2,1)
    for j = y1(1,1):y1(2,1)
        xn(ii,1) = i;
        yn(ii,1) = j;
        ii = ii + 1;
    end
end

%RGB色彩空间中显示
figure(2)
for i = 1:size(xn,1)
    Rn(i) = src(yn(i),xn(i),1);
    Gn(i) = src(yn(i),xn(i),2);
    Bn(i) = src(yn(i),xn(i),3);
end
plot3(Rn,Gn,Bn,'or');
