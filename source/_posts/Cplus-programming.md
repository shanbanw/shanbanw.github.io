title: "C++程序设计"
date: 2016-12-18 16:07:56
tags: C/C++
---

函数指针

程序运行期间，每个函数都会占用一段连续的内存空间。函数名就是该函数所占内存空间的起始地址。可以把函数的起始地址赋给一个指针变量，使该指针变量指向该函数，然后通过指针变量就可以调用这个函数。这种指向函数的指针变量称为“函数指针”。

<pre class="brush: cpp;">
//类型名 （*指针变量名）(参数类型1， 参数类型2， ...);
int (*pf)(int, char);

//通过函数指针调用它指向的函数
//函数指针名（实参表）;
</pre>

<!--more -->

命令行参数

<pre class="brush: cpp;toolbar:false;">
int main(int argc, char *argv[]){
//argc代表程序启动时命令行参数的个数。可执行程序本身的文件名也是一个参数，因此argc的值至少为1.
//argv指针数组，其中每个元素都是char *类型的指针，该指针指向一个字符串，这个字符串里存放着命令行参数。例如，argv[0]指向的字符串就是第一个命令行参数，即可执行程序的文件名，argv[1]指向第二个命令行参数，argv[2]指向第三个命令行参数.
}
</pre>

位运算

<pre class="brush: cpp;">
//按位与“&”，按位或“|“，按位异或“^”，非“~”
//左移运算符 &lt;&lt;高位丢弃 低位补零 比乘法快
//右移运算符 &gt;&gt;低位丢弃，对于有符号的int等高位为1就补1，高位为零就补零。
</pre>

引用

定义引用时一定要初始化成引用某个变量，初始化后它就一直引用该变量，不会再引用别的变量。只能引用变量，不能引用常量和表达式。<br>
函数返回值写成引用，非引用的函数返回值不可以作为左值使用。
<pre class="brush: cpp;">
int n=4;
int &r=n; //r引用了n，相当于n的一个别名
const int & t=n; //常引用 不能通过其修改引用的内容
</pre>

动态内存分配

<pre class="brush: cpp;">
int * pn = new int;
* pn = 5;
delete pn;
int * pm = new int[10];
delete [] pm;
</pre>

内联函数

应对函数被反复执行很多次的情况。编译器处理对内联函数的调用语句时，是将整个函数的代码插入到调用语句处，而不会产生调用函数的语句。在函数定义前面加“inline"关键字即可定义内联函数。

<pre class="brush: cpp;">
inline int Max(int a, int b){
    if (a>b) return a;
    return b;
}
</pre>

函数重载

一个或多个函数，名字相同，返回值类型相同，参数个数或参数类型不同，编译器根据调用语句中的实参个数和类型判断调用哪个函数。不能确定调用哪个函数时出现”二义性“错误。

**类**

对象的大小等于所有成员变量的大小之和，
每个对象各有自己的存储空间。
对象之间可以用”=“进行赋值。

缺省为私有成员

<pre class="brush: cpp;">
class classname{
    private:   //只能在成员函数内被访问
        //私有属性和函数
    public:   //可以在任何地方被访问
        //公有属性和函数
    protected:   //继承
        //保护属性和函数
}; //分号
</pre>

对象成员的访问权限

* 类的成员函数内部，可以访问：
    * 当前对象的全部属性，函数
    * 同类其他对象的全部属性和函数
* 类的成员函数以外的地方，
    * 只能访问该类对象的公有成员。

内联成员函数

inline + 成员函数(声明在类内，定义在类外)
整个函数体出现在类定义内部

成员函数重载及参数缺省

构造函数

名字与类名相同，可以有参数，不能有返回值（void也不行），如果定义类时没写构造函数，则编译器生成一个默认的无参数的构造函数，不做任何操作。对象生成时构造函数自动被调用，一个类可以有多个构造函数。

复制构造函数

只有一个参数，即同类对象的引用。如果没有定义复制构造函数，编译器默认生成一个。

<pre class="brush: cpp;">
X::X( X&)
X::X(const X &) //能以常量对象作为参数
</pre>

类型转换构造函数：只有一个参数，不是复制构造函数，建立一个临时对象。

析构函数

名字与类名相同，在前面加”~“，没有参数和返回值，一个类最多只有一个析构函数

静态成员变量和静态成员函数

在声明前加static，为所有对象共享，不需要通过对象就能访问。sizeof不会计算静态成员变量。静态成员变量本质上是全局变量，哪怕一个对象都不存在，类的静态成员变量也存在。静态成员函数本质上是全局函数，因此函数里面不能访问非静态成员变量，也不能调用非静态成员函数。

必须在定义类的文件中对静态成员变量进行一次初始化。

设置静态成员是为了将和某些类紧密相关的全局变量和函数写到类里面，看上去像一个整体，易于维护和理解。

<pre class="brush: cpp;">
 class CMyclass{
     int n;
     static int s;
 }; //sizeof(CMyclass)只计算int，所以等于4
 int CMyclass::s = 0; //初始化
 CMyclass::s; //类名::成员名
 CMyclass r; r.s; //对象名.成员名
 CMyclass *p=&r; p-&gt;s; //指针-&gt;成员名
 CMyclass & ref = r; ref.s; // 引用.成员名
</pre>

成员对象和封闭类

成员对象：一个类的成员变量是另一个类的对象。

包含成员对象的类叫封闭类（encclosing）。

友元（Friend）

一个类的**友元函数**可以访问该类的私有成员<br>
A是B的**友元类**，A的成员函数可以访问B的私有成员，友元类之间不能传递，不能继承

this指针

指向成员函数所作用的对象。静态成员函数不能使用this指针！

<pre class="brush: cpp;">
class A{
        int i;
    public:
        void hello_1(){cout&lt;&lt;"hello"&lt;&lt;endl;}
        //void hello_1(A * this){cout&lt;&lt"hello"&lt;&lt;endl;}
        void hello_2(){cout&lt;&lt;i&lt;&lt;"hello"&lt;&lt;endl;}
        //void hello_2(A * this){cout&lt;&lt;this-&gt;i&lt;&lt;"hello"&lt;&lt;endl;
};
int main(){
    A *p =NULL;
    p-&gt;hello_1(); //输出hello hello_1(p)
    p-&gt;hello_2(); //出错
    return 0;
}
</pre>

常量对象 常量成员函数和常引用

如果不希望某个对象的值被改变，则定义对象的时候前面加const关键字。const Demo obj; //常量对象<br>
在类的成员函数说明后面加const，该成员函数成为常量成员函数。在常量成员数不能修改成员变量的值（除静态成员变量外），也不能调用同类的非常量成员函数（静态成员函数除外）。void Sample::GetValue() const;<br>
常量成员函数重载：两个成员函数，名字和参数都一样，但一个是const，一个不是，算重载。<br>
常引用：不能通过常引用修改其引用的变量。对象作为函数的参数时，生成该参数需要调用复制构造函数，效率比较低，用指针作参数，代码又不好看，这时可以用对象的**常引用**作为参数，就能确保不会修改对象的值。

**运算符重载**

运算符重载的实质是函数重载<br>
在程序编译时，把含运算符的表达式转换为对<font color="green">运算符函数</font>的调用，把运算符的操作数转换给运算符函数的参数，运算符被多次重载时，根绝实参的类型决定调用哪个运算符函数。<br>
重载为普通函数时，<font color="blue">参数个数为运算符目数</font>；重载为类成员函数时，<font color="blue">参数个数为运算符目数减一</font>。<br>
<pre class="brush: cpp;">
返回值类型 operator 运算符 (形参表){

}
</pre>

赋值运算符‘=’重载

只能重载为<font color="red">成员函数</font><br>
返回值类型不能是void<br>
<pre class="brush: cpp;">
//一个长度可变的字符串类String
// 包含一个char *类型的成员变量，指向动态分配的存储空间
//该存储空间用于存放'\0'结尾的字符串
class String{
    private:
        char * str;
    public:
        //构造函数，初始化str为NULL
        String (): str(NULL){}
        //c_str返回一个常量指针,确保不会通过这个指针修改对象的值
        const char * c_str(){return str;}
        //实现String S1; S1="hello";能够成立
        char  *operator = (const char * s);
        //实现String S1,S2; S1= "this"; S2="that"; S1=S2;这里如果默认调用复制构造函数只能进行潜复制，存在很大问题
        String & operator=(const String & s);
        //String s1; s1="hello";String s2(s1);会调用复制构造函数，存在潜复制的问题
        String(String & s);
        ~String();
};
//重载'='
char * String::operator=(const char *s){
    if(str) delete [] str; //this指针
    if(s){ //s不为NULL才执行拷贝
        str = new char[strlen(s)+1];
        strcpy(str, s);
    }
    else
        str = NULL;
    return str;
}
String & operator=(const String & s){
    if(str == s.str) return * this; //String s; s= "Hello”; s=s;
    if(str) delete [] str;
    if(s.str){  //s.str不为NULL才会执行拷贝
        str = new char[strlen(s.str)+1];
        strcpy(str, s.str);
    }
    else
        str = NULL;
    return * this;
}
//初始化的时候调用
String::String(String & s){
    if(s.str){
        str = new char[strlen(s.str)+1];
        strcpy(str, s.str);
    }
    else
        str = NULL;
}
String s2 = "hello"; //出错，初始化，调用构造函数，但这个例子里面没有调用
</pre>

运算符重载为友元函数
>成员函数不能满足使用要求，例如不能通过对象调用成员函数实现某种操作
>普通函数不能访问类的私有成员

<pre class="brush: cpp; highlight=[6];">
class Complex{
        double real, imag;
    public:
        Complex(double r,double i): real(r), imag(i){};
        Complex operator+(double r);
        friend Complex operator+(double r, const Complex & c);
};
//能够计算c+5，但不能计算5+c
Complex Complex::operator+(double r){
    return Complex(real + r, imag);
}
//实现5+c的友元函数
Complex operator+ (double r, const Complex & c){
    return Complex(c.real+r, c.imag);
}
</pre>

流插入和流提取运算符的重载

cout是在iostream中定义的ostream类的对象。

<pre class="brush: cpp;">
//重载为ostream的成员函数
ostream & ostream::operator&lt;&lt;(int n){
    ...  //输出n的代码
    return * this;
}
ostream & ostream::operator&lt;&lt;(const char * s){
    ...  //输出s的代码
    return * this;
}
cout&lt;&lt;5&lt;&lt;"this"; //函数调用形式如下
cout.operator&lt;&lt;(5).operator&lt;&lt;("this");

//一个例子
#include &lt;iostream&gt;
#include &lt;string&gt;
#include &lt;cstdlib&gt;
using namespace std;
class Complex {
        double real, imag;
    public:
        Complex(double r=0,double i=0):real(r),imag(i){}
        friend ostream & operator&lt;&lt;(ostream & os, const Complex & c);
        friend istream & operator&gt;&gt;(istream & is, Complex & c);
};
ostream & operator&lt;&lt;(ostream &os, const Complex & c){
    //以“a+bi"的形式输出
    os&lt;&lt;c.real&lt;&lt;"+"&lt;&lt;c.imag&lt;&lt;"i";
    return os;
}
istream & operator&gt;&gt;(istream & is,Complex & c){
    string s;
    //将"a+bi"作为字符串读入，不能有空格
    is&gt;&gt;s;
    int pos = s.find("+",0);
    string sTmp = s.substr(0,pos); //分离出实部
    //atof库函数能将const char *指针指向的内容转换成float
    c.real = atof(sTmp.c_str());
    sTmp = s.substr(pos+1, s.length()-pos-2); //分离虚部
    c.imag = atof(sTmp.c_str());
    return is;
}
</pre>

自增/自减运算符重载

前置运算符作为一元运算符重载，后置作为二元运算符重载（多写一个参数，具体无意义）<br>

类型强制转换运算符重载时，不能写返回值类型，实际上其返回值类型是类型强制转换运算符代表的类型<br>

<pre class="brush: cpp; highlight=[8];">
class CDemo{
    private:
        int n;
    public:
        CDemo(int i=0):n(i){}
        CDemo operator++(); //前置形式
        CDemo operator++(int); //后置形式
        operator int(){return n;}  //强制类型转换运算符重载，(int)s;等效于s.int();
        friend CDemo operator--(CDemo &); //全局函数
        friend CDemo operator--(CDemo &, int); //全局函数
};
CDemo CDemo::operator++(){//前置
    n++;
    return *this;
}
CDemo CDemo::operator(int k){
    CDemo tmp(*this);
    n++;
    return tmp;
}
CDemo operator--(CDemo &d){
    d.n--;
    return d;
}
CDemo operator--(CDemo & d,int){
    CDemo tmp(d);
    d.n--;
    return tmp;
}
</pre>

运算符重载注意事项：
>C++不允许定义新的运算符
>重载后运算符的含义应该符合日常习惯
>运算符重载不改变运算符的优先级
>以下运算符不能被重载: . .* :: ?: sizeof
>重载运算符(), [], ->, =时，重载函数必须声明为类的成员函数


继承和派生

派生类有基类所有的成员函数和成员变量，但派生类的成员函数不能访问基类的私有成员<br>
<pre class="brush: cpp;">
class 派生类名: public 基类名{
}；
</pre>
派生类对象的体积 等于基类对象体积加上派生类自己的成员变量的体积。<br>

继承关系和复合关系

复合关系：类C中有成员变量k，k是类D的对象。<br>
例如几何形体程序中，需要写点类，也需要写圆类，因为圆不是点，所以圆类不能继承点类，而是每一个圆对象都包含一个点对象，这个点对象就是圆心。<br>
<pre class="brush: cpp;">
class CPoint{
    double x,y;
    friend class CCircle; //便于CCircle类操作其圆心
};
class CCircle{
    double r;
    CPoint center;
};
</pre>

基类和派生类同名成员，一般情况下基类和派生类不定义同名的成员变量
<pre class="brush: cpp;">
class base{
        int j;
    public:
        int i;
        void func();
};
class derived: public base{
    public:
        int i;
        void access();
        void func();
};
void derived::access(){
    j=5; //error
    i=5; //derived类的i
    base::i = 5; //基类的i 域作用符
    func(); 
    base::func();
}
derived obj;
obj.i=1;
obj.base::i=1;  //基类
</pre>

访问范围说明符：protected
基类的protected成员：可以被下列函数访问
>基类的成员函数
>基类的友元函数
>派生类的成员函数可以访问“当前对象”的基类的保护成员

派生类构造函数
执行派生类构造函数之前，先执行基类的构造函数
派生类交代基类初始化，具体形式如下：
构造函数名(形参表)：基类名(基类构造函数实参表){}

虚函数

在类的定义中，前面有virtual关键字的成员函数就是虚函数。<br>
virtual关键字只用在类定义的函数声明中，写函数体时不用。<br>
构造函数和静态成员函数不能是虚函数。<br>
<pre class="brush: cpp;">
class base{
    virtual int get();
};
int base::get(){}
</pre>

多态

派生类的指针可以赋给基类指针。<br>
通过基类指针调用基类和派生类中的同名“虚函数”时：
>若该指针指向一个基类的对象，那么被调用是基类的虚函数；
>若该指针指向一个派生类的对象，那么被调用的是派生类的虚函数。

派生类的对象可以赋给基类引用，调用同名虚函数时作用同上。

多态的关键在于通过基类指针或引用调用一个虚函数时，编译时不确定到底调用的是基类还是派生类的函数，运行时才确定，这叫动态联编。

虚析构函数

通过基类指针删除派生类对象时，只调用基类的析构函数，会使派生类析构函数不被调用；<br>
把基类的析构函数声明为virtual，
>派生类的析构函数virtual可以不进行声明
>通过基类的指针删除派生类对象时，首先调用派生类的析构函数，然后调用基类的析构函数
类如果定义了虚函数，最好将析构函数也定义成虚函数

纯虚函数：没有函数体的虚函数
<pre class="brush: cpp; highlight=[5];">
classA{
    private:
        int a;
    public:
        virtual void Print() = 0; //纯虚函数
        void fun(){cout << "fun";}
};
A a; //错，A是抽象类，不能创建对象
A * pa;  //OK，可以定义抽象类的指针和引用
pa = new A; // 错，A是抽象类，不能创建对象
</pre>

抽象类：包含纯虚函数的类，主要用于实现多态
>只能作为基类
>不能创建抽象类的对象
>抽象类的指针和引用，由抽象类派生出来的类的对象

抽象类的成员函数内可以调用纯虚函数，构造函数和析构函数不能调用纯虚函数<br>
如果一类从抽象类派生而来，它实现了基类中的所有纯虚函数，才能成为非抽象类。

文件操作

ifstream, ofstream, fstream3个类用于文件操作，统称为文件流类。

使用/创建文件的基本流程：打开文件，读写文件，关闭文件。
<pre class="brush: cpp;">
#include <fstream> //包含头文件
ofstream outFile("clients.dat", ios::out|ios::binary); //打开文件
//ofstream 是fstream中定义的类
//outFile 自定义的ofstream类的对象
//ios::out 输出到文件，删除原有内容
//ios::输出到文件，保留原有内容，在尾部添加
//ios::binary 以二进制文件格式打开文件

ofstream fout;  //先创建ofstream对象，再用open函数打开
fout.open("test.out",ios::out|ios::binary);
if(!fout){cerr&lt;&lt;"File open error!"&lt;&lt;endl;};
</pre>

文件的读写指针
>对于输入文件，有一个读指针
>对于输出文件，有一个写指针
>对于输入输出文件，有一个读写指针
>标记文件操作的当前位置，该指针在哪里，读写操作就在哪里进行

函数tellp取得写指针的位置，seekp移动写指针
函数tellg取得读指针的位置，seekg移动读指针

函数模板

算法实现一遍，适用于多种数据结构
<pre class="brush: cpp;">
template&lt;class 类型参数1, class 类型参数2, ...&gt;
返回值类型 模板名(形参表){
    函数体
}
//交换两个变量值的函数
template&lt;class T&gt;
void Swap(T & x, T & y){
    T tem = x;
    x = y;
    y = tmp;
}
</pre>

模板函数调用在普通函数之后。可以在函数模板中使用多个类型参数，来避免二义性。

类模板

定义一批相似的类，定义类模板，生成不同的类
<pre class="brush: cpp;">
template &lt;类型参数表&gt;
class 类模板名{
    成员函数和成员变量
};
//类模板里的成员函数，如在类模板外定义
template&lt;类型参数表&gt;
返回值类型 类模板名&lt;类型参数表&gt;::成员函数名(参数表){
    
}
//用类模板定义对象
类模板名 &lt;真实类型参数表&gt; 对象名(构造函数实际参数表)；
类模板名&lt;真实类型参数表&gt; 对象名;
</pre>

string类

是一个模板类，定义如下：<br>
<pre class="brush: cpp;">
typedef basic_string&lt;char&gt; string;
//string对象初始化
string s1("hello");
string s2(8,'x');
string month = "March";
</pre>
使用string类要包含头文件&lt;string&gt;

输入输出<br>
                +---ifstream<br>
    +---istream-<br>
ios-            +---iostream---&gt;fstream<br>
    +---ostream-<br>
                +---ofstream<br>

标准模板库STL

用于存放各种类型的数据的数据结构，都是类模板，分为三种：
>顺序容器 vector, deque, list
>关联容器 set, multiset, map, multimap
>容器适配器 stack, queue, priority_queue

顺序容器<br>
vector 头文件<vector> 动态数组<br>
deque  头文件<deque>  双向队列<br>
list   头文件<list>   双向链表<br>

关联容器
>元素是排序的，用于查找
>通常以平衡二叉树方式实现，插入和检索时间都是O(log(N))

set/multiset  头文件<set> <br>
set集合，不允许相同的元素，multiset允许相同元素

map/multimap  头文件<map> <br>
两者区别在于是否允许键有相同元素

容器适配器<br>
stack   头文件<stack>  栈<br>
queue   头文件<queue>  队列<br>
priority_queue 头文件<queue>  优先队列。最高优先级元素总是第一个出列

迭代器
>用于指向顺序容器和关联容器中的元素
>用法和指针类似
>有const和非const两种
>通过迭代器可以读取它指向的元素
>通过非const迭代器还能修改其指向的元素

容器类名::iterator 变量名；<br>
容器类名::const_iterator 变量名；<br>

访问一个迭代器指向的元素: * 迭代器变量名

算法是一个个函数模板，大多数在<algorithm>中定义

函数对象

若一个类重载了运算符"()"，则该类的对象就成为函数对象。
<pre class="brush: cpp;">
class CMyAverage{  //函数对象类
    public:
        double operator() (int a1, int a2,int a3){
            return (double)(a1+a2+a3)/3;
        }
};

CMyAverage average;  //函数对象
cout&lt;&lt;average(3,2,3);  //average.operator()(3,2,3)
</pre>









