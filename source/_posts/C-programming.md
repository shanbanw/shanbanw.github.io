title: "C语言学习"
date: 2016-12-17 18:40:08
tags: C/C++
---
C语言备忘录
<pre class="brush: cpp;">
#include &lt;iostream&gt;
using namespace std;

int main(){
	cout&lt;&lt"hello, world"&lt;&lt;endl;
	return 0;
}
</pre>

<!-- more -->

**函数**
<pre class="brush: cpp;">
r=sqrt(100.0); //求平方根
k=pow(x, y);  //求幂 x^y
i=strlen(str1); //求字符串长度
v=strcmp(str1, str2); //比较两个字符串大小
n=atoi(str1); //字符串转换为整数 “123” to 123
</pre>

**函数原型(Signature)** = 返回值类型+函数名+参数类型，里面的参数名（形参）可以不写。

<pre class="brush: cpp; highlight: [3];">
#include &lt;iostream&gt;
using namespace std;
float max(float, float);
int main(){
    cout&lt;&lt;max(3, 4);
    return 0;
}
float max(float a, float b){
    if(a &gt; b)
        return a;
    else
        return b;
}
</pre>

**参数传递：**实参与形参具有不同的存储单元，函数调用时，系统给形参分配存储单元，并将实参对应的值传递给形参，形参的值改变不会影响实参。

**变量的作用域：**根据变量在程序中作用范围的不同，可以将变量分为

* 局部变量： 在函数内或块内定义，只在这个函数或块内起作用的变量；
+ 全局变量： 在所有函数外定义的变量，它的作用域是从定义变量的位置开始到本程序文件结束；
- 当全局变量与局部变量同名时，局部变量将在自己作用域内有效，它将屏蔽同名的全局变量。

<pre class="brush: cpp; highlight: [3,4,7];">
#include &lt;iostream&gt;
using namespace std;
int a=0, b=0;  //全局变量
void exchange(int a, int b){ //局部变量a, b
    int p;
    if (a &lt; b){
        p=a; a=b; b=p;  //a, b交换，不会影响前面a, b的值
    }
}
int main(){
    cin&gt;&gt;a&gt;&gt;b;
    exchange(a, b);
    cout&lt;&lt;a&lt;&lt;" "&lt;&lt;b&lt;&lt;endl;
    return 0;
}
</pre>

数组名是一个常量，用来存储数组在内存中的地址。

<pre class="brush: cpp;">
#include &lt;iostream&gt;
using namespace std;
void change(int a[]){
    a[0]=30; a[1]=50;
}
int main(){
    int a[2]={3,5};
    change(a);  //数组名作为参数是把数组的地址copy给了形参，这里将会改变数组元素的值
    cout&lt;&lt;a[0]&lt;&lt;" "&lt;&lt;a[1]&lt;&lt;endl;
    return 0;
}
</pre>

**递归**

<pre class="brush: cpp;">
#include &lt;iostream&gt;
using namespace std;
int recur(){
    char c;
    c = cin.get();
    if (c != '\n')
        recur();
    cout&lt;&lt;c;
    return 0;
}

int main(){
    recur();
    return 0;
}
</pre>

输入abcd回车，将会输出回车dcba。每次输入不为回车，程序调用recur，当遇到回车符时，程序将不会再调用自身，从最后一个recur开始向下执行cout输出。

**指针**

取址运算符& 指针运算符*

<pre class="brush: cpp;">
int a=10;
cout&lt;&lt;&a;  //输出a在内存中的地址
cout&lt;&lt;*&a; //输出10
int *pointer; //定义一个指向整型变量的指针变量，里面存储一个地址，初始化需要赋值一个地址
</pre>

**指针变量：** 用于存放指针（某个变量的地址）的变量

数组名代表数组首元素的地址，即相当于指向数组第一个元素的指针。**数组名是常量，不能赋值。**

用指针变量访问数组元素，给指针赋值数组的地址，指针可以像数组名一样访问数组元素。

<pre class="brush: cpp;highlight: [12];">
#include &lt;iostream&gt;
using namespace std;
int main(){
    int a[5]={10,11,12,13,14};
    int *p =NULL;
    cout&lt;&lt;a&lt;&lt;endl; //输出地址
    p = a;
    cout&lt;&lt;p&lt;&lt;endl; //输出地址
    cout&lt;&lt;*p&lt;&lt;endl; //输出a[0] 10
    cout&lt;&lt;*p++&lt;&lt;endl;  //输出a[0] 10 后置++优先级高于*，但后置++是先用元素再自加 现在p指向a[1]
    cout&lt;&lt;*p++&lt;&lt;endl;  //同上 输出a[1] 11
    cout&lt;&lt;*p+1&lt;&lt;"=="&lt;&lt;p[1]&lt;&lt;"=="&lt;&lt;a[3]&lt;&lt;endl;  //输出a[3] 13
    return 0;
}
</pre>

递归实现斐波那契数列

<pre class="brush: cpp; highlight: [11];">
#include &lt;iostream&gt;
using namespace std;
void feb(int n, int a[]){
    if (n==1){
        a[0]=1;a[1]=1;
    }
    if (n==2){
        a[0]=1;a[1]=1;
    }
    else{
        feb(n-1, a);
        int tmp = a[0]; a[0]=a[0]+a[1];a[1]=tmp;
    }
}
int main(){
    int a[2]={0};
    feb(10, a);
    cout&lt;&lt;"n=10: "&lt;&lt;a[0]&lt;&lt;endl;  //55
    return 0;
}
</pre>

指向二维数组的指针

<pre class="brush: cpp; highlight: [6];">
#include &lt;iostream&gt;
using namespace std;
int main(){
    int a[3][4]={1,2,3,4,5,6,7,8,9,10,11,12};
    int *p;
    for (p=&a[0][0]; p&lt;&a[0][0]+12;p++){
        cout&lt;&lt;p&lt;&lt;" "&lt;&lt;*p&lt;&lt;endl;
    }
    return 0;
}
</pre>

对于二维数组a，a相当于指向a[3][4]的“第一个元素”的指针，所谓的“第一个元素”是指一个“包含4个int型元素的一维数组”，所以a相当于一个“包含4个int型元素的一维数组”的地址。假如定义一个指针p，要使p=a，那么p的“基类型”应该是“包含4个int型元素的一维数组”，可写为int (*p)[4]，如下。

<pre class="brush: cpp; highlight: [3];">
int a[3][4]={1,2,3,4,5,6,7,8,9,10,11,12};
int (*p)[4];
p=a;  //赋值，*(*(p+i)+j) == p[i][j] == a[i][j]
</pre>

分析\*(*(p+i)+j)

* p指向一个“包含4个int型元素的一维数组”；
* p+i是第i+1个“包含4个int型元素的一维数组”的地址；
* p+i等价于&a[i]；
* *(p+i)等价于a[i];
* a[i]+j就是&a[i][j]，而&a[i]+j是&a[i+j];
* *(p+i)+j等价于a[i]+j，也就是&a[i][j]；
* 所以*(*(p+i)+j)等价于a[i][j]。

除非sizeof，Alignof，&三个操作符作用于数组名，或者赋值为一个字符串，数组名将会被转换为指向首元素的指针。即int a[4]，假如int占4个字节，那么a+1将会返回数组首元素的地址加上4的地址；然而&a是被当作更高一级的指向整个数组的指针，所以&a+1将会返回数组首元素的地址加上4乘以数组长度的地址。

<pre class="brush: cpp; highlight: [8];">
#include &lt;iostream&gt;
using namespace std;
int main(){
    int a[4]={1,3,5,7};
    cout&lt;&lt;a&lt;&lt;endl;
    cout&lt;&lt;a+1&lt;&lt;endl;  //跨越一个int，即4个字节
    cout&lt;&lt;&a&lt;&lt;endl;
    cout&lt;&lt;&a+1&lt;&lt;endl;  //跨越整个数组，本例中的16个字节
    cout&lt;&lt;*(&a)&lt;&lt;endl; //*(&a)相当于a
    return 0;
}
</pre>

>数组名相当于指向数组第一个元素的指针，对于二维数组，第一个元素将是一个数组；
>&E相当于把E的管辖范围上升了一个级别，体现在+1时跨越的字节数；
>\*E相当于把E的管辖范围下降了一个级别，最小的时候是代表一个数值，int a[2]={1,2}; *a表示1.

指向字符串的指针

<pre class="brush: cpp;">
int main(){
    char a[] = {'h','e','l','l','o','\0'};
    char *p = a;
    cout&lt;&lt;a&lt;&lt;endl;   // hello
    cout&lt;&lt;p&lt;&lt;endl;  // hello
    cout&lt;&lt;static_cast&lt;void*&gt;(a)&lt;&lt;endl;  //输出地址
    cout&lt;&lt;static_cast&lt;void*&gt;(p)&lt;&lt;endl;  //输出地址
    
    //p="ABC";
    //cout&lt;&lt;p&lt;&lt;endl;  
}
</pre>

**符号常量** const int a = 10; int const a = 10;

指向符号常量的指针 const int *p; 只是不能更改指针所指向的量的数值。

**静态局部变量：**函数中局部变量的值在函数调用结束后不消失而保留原值，即其占用的存储单元不释放，在下一次该函数调用时，仍可以继续使用该变量。

<pre class="brush: cpp;">
static int value = 20;
</pre>

指针用作函数参数时为了防止对所指内容进行修改，可以用const来“限制”指针的功能；当指针用作函数返回值时，必须确保函数返回的地址是有意义的，返回全局变量或者静态局部变量的地址。

**结构体**

一种数据类型

<pre class="brush: cpp;">
struct student{
    int id;
    char name[20];
    float score;
};  //分号
student student1,student2;
student student3 = {3,{'m','i','k','e','\0'}, 82.1};
student3.id = student3.id+3;
student1=student3;
student *one = student3; //指向结构体的指针
cout&lt;&lt;(*one).id&lt;&lt;" "&lt;&lt;(*one).name;
cout&lt;&lt;one-&gt;id&lt;&lt;" "&lt;&lt;one-&gt;name; //指向运算符，指针操作结构体和其他的都一样。
student myclass[2]={1,{'q','\0'},10,2,{'w','\0'},2};
</pre>

动态申请内存空间

<pre class="brush: cpp;">
int *pint = new int(1024);  //new 创建内存空间，并返回地址
delete pint;
int *pia = new int[4];
delete [] pia;
</pre>






