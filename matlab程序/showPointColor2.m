%将图片上的点显示在色彩空间中
clear;
clc;
clf;
src = imread('xihu.png');
src = imresize(src,128.0/size(src,1));% 归一化大小
RGB = boolean(zeros(255,255,255));% RGB空间（1-255）

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

%显示RGB空间图像
figure(2)
hold on;
grid on;
for i = 1:255
    for j = 1:255
        for k = 1:255
            if RGB(i,j,k) == 1
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

%归一化RGB
figure(3);
hold on;
grid on;
dr = double(R);
dg = double(G);
db = double(B);

r = dr./(dr+dg+db);
g = dg./(dr+dg+db);
b = db./(dr+dg+db);

plot3(r,g,b,'ob');
xlabel('r');
ylabel('g');
zlabel('b');
title('归一化RGB');
% axis([0,256,0,256,0,256]);
view(30,30);

%HSV色彩空间中显示
figure(4)
hold on;
grid on;

% cform = makecform('srgb2lab'); 
% hsv = applycform(src, cform);%lab
% hsv = rgb2ycbcr(src);%yCbCr
hsv = rgb2hsv(src);
for i = 1:size(x,1)
    H(i) = hsv(y(i),x(i),1);
    S(i) = hsv(y(i),x(i),2);
    V(i) = hsv(y(i),x(i),3);
end
plot3(H,S,V,'ob');
xlabel('H');
ylabel('S');
zlabel('V');
title('HSV');
% axis([0,1,0,1,0,1]);
view(30,30);

%YCbCr色彩空间中显示
figure(5)
hold on;
grid on;

ycbcr = rgb2ycbcr(src);
for i = 1:size(x,1)
    Y(i) = ycbcr(y(i),x(i),1);
    Cb(i) = ycbcr(y(i),x(i),2);
    Cr(i) = ycbcr(y(i),x(i),3);
end
plot3(Y,Cb,Cr,'ob');
xlabel('Y');
ylabel('Cb');
zlabel('Cr');
title('YCbCr');
% axis([0,1,0,1,0,1]);
view(30,30);
%---------负样本-----------
%图片框范围
figure(1)
[x0,y0] = ginput(2);
x1 = max(1,int32(x0));
y1 = max(1,int32(y0));

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
    Rn(i) = max(1,Rn(i));Gn(i) = max(1,Gn(i));Bn(i) = max(1,Bn(i));
    if RGB(Rn(i),Gn(i),Bn(i)) == 0
        plot3(Rn(i),Gn(i),Bn(i),'oy');
    else
        plot3(Rn(i),Gn(i),Bn(i),'or');
    end
end


%归一化RGB
figure(3);
drn = double(Rn);
dgn = double(Gn);
dbn = double(Bn);

rn = drn./(drn+dgn+dbn);
gn = dgn./(drn+dgn+dbn);
bn = dbn./(drn+dgn+dbn);

plot3(rn,gn,bn,'oy');

%HSV色彩空间中显示
figure(4)
for i = 1:size(xn,1)
    Hn(i) = hsv(yn(i),xn(i),1);
    Sn(i) = hsv(yn(i),xn(i),2);
    Vn(i) = hsv(yn(i),xn(i),3);
end
plot3(Hn,Sn,Vn,'oy');

%YCbCr色彩空间中显示
figure(5)
for i = 1:size(xn,1)
    Yn(i) = ycbcr(yn(i),xn(i),1);
    Cbn(i) = ycbcr(yn(i),xn(i),2);
    Crn(i) = ycbcr(yn(i),xn(i),3);
end
plot3(Yn,Cbn,Crn,'oy');

%---------第三样本-----------
%图片框范围
figure(1)
[x0,y0] = ginput(2);
x1 = max(1,int32(x0));
y1 = max(1,int32(y0));

ii = 1;
for i = x1(1,1):x1(2,1)
    for j = y1(1,1):y1(2,1)
        xn2(ii,1) = i;
        yn2(ii,1) = j;
        ii = ii + 1;
    end
end

%RGB色彩空间中显示
figure(2)
for i = 1:size(xn2,1)
    Rn2(i) = src(yn2(i),xn2(i),1);
    Gn2(i) = src(yn2(i),xn2(i),2);
    Bn2(i) = src(yn2(i),xn2(i),3);
    Rn2(i) = max(1,Rn2(i));Gn2(i) = max(1,Gn2(i));Bn2(i) = max(1,Bn2(i));
    if RGB(Rn2(i),Gn2(i),Bn2(i)) == 0
        plot3(Rn2(i),Gn2(i),Bn2(i),'og');
    else
        plot3(Rn2(i),Gn2(i),Bn2(i),'or');
    end
end
