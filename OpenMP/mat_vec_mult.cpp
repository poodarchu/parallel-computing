#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

using namespace std;

#define MatrixOrder 1024
#define FactorIntToDouble 1.1; //使用rand（）函数产生int型随机数，将其乘以因子转化为double型；

double firstParaMatrix [MatrixOrder] [MatrixOrder] = {0.0};  
double secondParaMatrix [MatrixOrder] [MatrixOrder] = {0.0};
double matrixMultiResult [MatrixOrder] [MatrixOrder] = {0.0};

//计算matrixMultiResult[row][col]
double calcuPartOfMatrixMulti(int row,int col)
{
    double resultValue = 0;
    for(int transNumber = 0 ; transNumber < MatrixOrder ; transNumber++) {
        resultValue += firstParaMatrix [row] [transNumber] * secondParaMatrix [transNumber] [col] ;
    }
    return resultValue;
}

void matrixInit()
{
    #pragma omp parallel for num_threads(64)
    for(int row = 0 ; row < MatrixOrder ; row++ ) {
        for(int col = 0 ; col < MatrixOrder ;col++){
            srand(row+col);
            firstParaMatrix [row] [col] = ( rand() % 10 ) * FactorIntToDouble;
            secondParaMatrix [row] [col] = ( rand() % 10 ) * FactorIntToDouble;
        }
    }
}

void matrixMulti()
{
    #pragma omp parallel for num_threads(64)
    for(int row = 0 ; row < MatrixOrder ; row++){
        for(int col = 0; col < MatrixOrder ; col++){
            matrixMultiResult [row] [col] = calcuPartOfMatrixMulti (row,col);
        }
    }
}

int main()  
{ 
    matrixInit();

    clock_t t1 = clock(); //开始计时；
    matrixMulti();
    clock_t t2 = clock(); //结束计时
    cout<<"time: "<<t2-t1<<endl; 
    
    return 0;  
} 
