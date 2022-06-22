#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include<stdlib.h>
#include<string.h>
using namespace std;
#define MIN(a,b)  ((a)<(b)?(a):(b))

int main(int argc, char* argv[])
{
	long long     count;        /* Local prime count */
	double elapsed_time; /* Parallel execution time */
	long long     first;        /* Index of first multiple */
	long long     global_count; /* Global prime count */
	long long     high_value;   /* Highest value on this proc */
	int    i;
	int    id;           /* Process ID number */
	int    index;        /* Index of current prime */
	long long     low_value;    /* Lowest value on this proc */
	char* marked;       /* Portion of 2,...,'n' */
	long long     n;            /* Sieving from 2, ..., 'n' */
	int    p;            /* Number of processes */
	long long     proc0_size;   /* Size of proc 0's subarray */
	long long     prime;        /* Current prime */
	long long     size;         /* Elements in 'marked' */

	MPI_Init(&argc, &argv);

	/* Start the timer */

	MPI_Comm_rank(MPI_COMM_WORLD, &id);//得到本进程的进程号 [0, p - 1]
	MPI_Comm_size(MPI_COMM_WORLD, &p);//得到参加运算的进程个数
	MPI_Barrier(MPI_COMM_WORLD);//阻止调用直到communicator中所有进程完成调用
	elapsed_time = -MPI_Wtime();

	//参数个数为2 ： 文件名和问题规模
	if (argc != 2) {
		if (!id) printf("Command line: %s <m>\n", argv[0]);
		MPI_Finalize();
		exit(1);
	}

	//问题规模 [2, n]
	n = atoi(argv[1]);

	/* Figure out this process's share of the array, as
	   well as the integers represented by the first and
	   last array elements */

	   // 分配数据块
	low_value = 2 + id * (n - 1) / p;//进程第一个数
	high_value = 1 + (id + 1) * (n - 1) / p;//进程最后一个数
	size = (high_value - low_value) + 1;//处理的数组大小

	/* Bail out if all the primes used for sieving are
	   not all held by process 0 */

	proc0_size = (n - 1) / p; //process 0的数据范围

	//因为0号节点是根节点 是从0号节点的数据范围里寻找到最小的尚未被标记的素数然后进行广播 所以必须保证 2- sqrt(n)在0号节点的数据范围内
	// 否则会输出不正确 
	if ((2 + proc0_size) < (int)sqrt((double)n)) {
		if (!id) printf("Too many processes\n");
		MPI_Finalize();
		exit(1);
	}

	/* Allocate this process's share of the array. */
	//为当前节点分配标记数组
	marked = (char*)malloc(size);

	//分配失败
	if (marked == NULL) {
		printf("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit(1);
	}

	//默认全为素数
	for (i = 0; i < size; i++) marked[i] = 0;
	//0号节点的index从0开始 用于寻找下一个未被标记的最小素数
	if (!id) index = 0;
	//第一个未被标记的最小素数
	prime = 2;
	do {
		//从当前素数的平方开始寻找  如果current prime^2大于该节点的数据左端点 减去low_value即为第一个倍数  
		if (prime * prime > low_value)
			first = prime * prime - low_value;
		else {
			if (!(low_value % prime)) first = 0;//该节点的左端点大于prime^2 从左端点开始寻找 如果low_value是第一个倍数  first = 0
			else first = prime - (low_value % prime);//如果low_value不是第一个倍数 计算first
		}
		for (i = first; i < size; i += prime) marked[i] = 1;//从第一个倍数开始 在该节点的数据范围内 标记current prime的倍数
		//0号节点寻找下一个最小的未被标记的素数
		if (!id) {
			while (marked[++index]);
			prime = index + 2;
		}
		//0号节点将找到的下一个最小的未被标记的素数广播给其他节点
		if (p > 1) MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} while (prime * prime <= n);//继续进行标记的条件
	count = 0;//标记结束后  计算该节点内未被标记的元素个数
	for (i = 0; i < size; i++)
		if (!marked[i]) count++;
	//当有多个进程时 其他节点将所计算得到的count发送给0号节点 并计算之和
	if (p > 1) MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	else //如果只有一个进程 不需要进行规约  总的素数个数就是0号节点所计算得到的count
	{
		global_count = count;
	}

	/* Stop the timer */

	elapsed_time += MPI_Wtime();

	/* Print the results */

	if (!id) {
		printf("There are %d primes less than or equal to %d\n", global_count, n);
		printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
	}
	MPI_Finalize();
	return 0;
}
