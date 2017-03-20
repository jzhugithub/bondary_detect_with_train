%查看各种颜色空间分量的Matlab程序实现
a=imread('2.png');  
%可以通过下面的程序看一幅图的RGB三个通道 
R=a(:,:,1);
G=a(:,:,2);
B=a(:,:,3);

figure(1);
subplot(2,2,1);imshow(a);title('原始图像'); 
subplot(2,2,2);imshow(R);title('R分量图像');
subplot(2,2,3);imshow(G);title('G分量图像');
subplot(2,2,4);imshow(B);title('B分量图像');

figure(2);
dr = double(R);
dg = double(G);
db = double(B);

r = dr./(dr+dg+db);
g = dg./(dr+dg+db);
b = db./(dr+dg+db);
subplot(2,2,1);imshow(a);title('原始图像'); 
subplot(2,2,2);imshow(r);title('归一化R分量图像');
subplot(2,2,3);imshow(g);title('归一化G分量图像');
subplot(2,2,4);imshow(b);title('归一化B分量图像');

figure(3);
a2 = rgb2hsv(a);
H=a(:,:,1);
S=a(:,:,2);
V=a(:,:,3);
subplot(2,2,1);imshow(a);title('原始图像'); 
subplot(2,2,2);imshow(H);title('H分量图像');
subplot(2,2,3);imshow(S);title('S分量图像');
subplot(2,2,4);imshow(V);title('V分量图像');

figure(4);
a3 = rgb2ycbcr(a);
Y = a(:,:,1);
Cr = a(:,:,2);
Cb = a(:,:,3);
subplot(2,2,1);imshow(a);title('原始图像'); 
subplot(2,2,2);imshow(Y);title('Y分量图像');
subplot(2,2,3);imshow(Cr);title('Cr分量图像');
subplot(2,2,4);imshow(Cb);title('Cb分量图像');

figure(5);
cform = makecform('srgb2lab'); 
lab = applycform(a, cform);
L=lab(:,:,1);
A=lab(:,:,2);
B=lab(:,:,3);
subplot(2,2,1);imshow(a);title('原始图像'); 
subplot(2,2,2);imshow(L);title('L分量图像');
subplot(2,2,3);imshow(A);title('A分量图像');
subplot(2,2,4);imshow(B);title('B分量图像');

