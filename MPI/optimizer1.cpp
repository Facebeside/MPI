//�Ż�һ��ȥ��ż��
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

	MPI_Comm_rank(MPI_COMM_WORLD, &id);//�õ������̵Ľ��̺� [0, p - 1]
	MPI_Comm_size(MPI_COMM_WORLD, &p);//�õ��μ�����Ľ��̸���
	MPI_Barrier(MPI_COMM_WORLD);//��ֹ����ֱ��communicator�����н�����ɵ���
	elapsed_time = -MPI_Wtime();

	//��������Ϊ2 �� �ļ����������ģ
	if (argc != 2) {
		if (!id) printf("Command line: %s <m>\n", argv[0]);
		MPI_Finalize();
		exit(1);
	}

	//�����ģ [2, n]
	n = atoi(argv[1]);

	/* Figure out this process's share of the array, as
	   well as the integers represented by the first and
	   last array elements */

    // �������ݿ�
	long long  N = (n - 3) / 2 + 1;//ȥ��ż����������ܸ���
	long long  low_index = id * N / p; //��һ�����±�
	long long  high_index = (id + 1) * N / p - 1;//���һ�����±�
	low_value = 3 + 2 * low_index;//���̵�һ����
	high_value = 3 + 2 * high_index;//�������һ����
	size = (high_value - low_value) / 2 + 1;//����������С

	/* Bail out if all the primes used for sieving are
	   not all held by process 0 */

	proc0_size = N / p; //process 0�����ݷ�Χ

	//��Ϊ0�Žڵ��Ǹ��ڵ� �Ǵ�0�Žڵ�����ݷ�Χ��Ѱ�ҵ���С����δ����ǵ�����Ȼ����й㲥 ���Ա��뱣֤ 2- sqrt(n)��0�Žڵ�����ݷ�Χ��
	// ������������ȷ 
	if ((3 + 2 * (proc0_size - 1)) < (long long )sqrt((long double)n)) {
		if (!id) printf("Too many processes\n");
		MPI_Finalize();
		exit(1);
	}

	/* Allocate this process's share of the array. */
	//Ϊ��ǰ�ڵ����������
	marked = (char*)malloc(size);

	//����ʧ��
	if (marked == NULL) {
		printf("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit(1);
	}

	//Ĭ��ȫΪ����
	for (i = 0; i < size; i++) marked[i] = 0;
	//0�Žڵ��index��0��ʼ ����Ѱ����һ��δ����ǵ���С����
	if (!id) index = 0;
	//��һ��δ����ǵ���С����
	prime = 3;
	do {
		//�ӵ�ǰ������ƽ����ʼѰ��  ���current prime^2���ڸýڵ��������˵� ��ȥlow_value�ټ��뼴Ϊ��һ���������±�
		if (prime * prime > low_value)
			first = (prime * prime - low_value) / 2;
		else {
			if (!(low_value % prime)) first = 0;//�ýڵ����˵����prime^2 ����˵㿪ʼѰ�� ���low_value�ǵ�һ������  first = 0
			else if (low_value % prime % 2)//���low_value���ǵ�һ������ ����first
			{
				first = (prime - low_value % prime) / 2;
			}
			else
			{
				first = prime - low_value % prime / 2;
			}
		}
		for (i = first; i < size; i += prime) marked[i] = 1;//�ӵ�һ��������ʼ �ڸýڵ�����ݷ�Χ�� ���current prime�ı���
		//0�Žڵ�Ѱ����һ����С��δ����ǵ�����
		if (!id) {
			while (marked[++index]);
			prime = 2 * index + 3; //���ݼ��� �±�index����ֵvalue��Ӧ�Ĺ�ϵΪ��value = 2 * index + 3
		}
		//0�Žڵ㽫�ҵ�����һ����С��δ����ǵ������㲥�������ڵ�
		if (p > 1) MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} while (prime * prime <= n);//�������б�ǵ�����
	count = 0;//��ǽ�����  ����ýڵ���δ����ǵ�Ԫ�ظ���
	for (i = 0; i < size; i++)
		if (!marked[i]) count++;
	//���ж������ʱ �����ڵ㽫������õ���count���͸�0�Žڵ� ������֮��
	if (p > 1) MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	else //���ֻ��һ������ ����Ҫ���й�Լ  �ܵ�������������0�Žڵ�������õ���count
	{
		global_count = count;
	}

	/* Stop the timer */

	elapsed_time += MPI_Wtime();

	/* Print the results */

	if (!id) {
		//��Ϊ�����Ǵ�3��ʼɸѡ�ģ�û�д�2 ��ʼ����2��������ż���������������+1
		printf("There are %d primes less than or equal to %d\n", global_count + 1, n);
		printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
	}
	MPI_Finalize();
	return 0;
}
