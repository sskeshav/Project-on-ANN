#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#define TOL pow(10,-2)

int main()
{
	int L,M,N,P;
	int i,j,k,p;    //for iterations
	int counter=1;
	
	srand(time(NULL));
		
	FILE *ip, *op1, *op2;

		ip  = fopen("Input File.txt","r");
		op1  = fopen("Output File.txt","w");
		op2 = fopen("Final Result.txt","w");

	double I[100][100],IH[100][100],OH[100][100],IO[100][100],OO[100][100],TO[100][100];
	double V[100][100],W[100][100],delta_W[100][100],delta_V[100][100];
	double e,MSE;
	double min_I[100],max_I[100],min_TO[100],max_TO[100];
	
	fscanf(ip,"%d%d%d%d",&P,&L,&M,&N);
	fprintf(op2,"Total No of Patterns (P)=%d\n",P);
	fprintf(op2,"Total No of Input Neurons (L)=%d\n",L);	
	fprintf(op2,"Total No of Hidden Neurons (M)=%d\n",M);	
	fprintf(op2,"Total No of Output Neurons (N)=%d\n",N);
	
//Scanning Inputs for Input layer

	for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			fscanf(ip,"%lf",&I[i][p]);
		}
	}
	
//Printing Inputs for Input layer

	fprintf(op2,"I matrix of order %dX%d :\n",L,P);
	
	for(i=1;i<=L;i++)
	{
		for(p=1;p<=P;p++)
		{
			fprintf(op2,"I[%d][%d]=%lf\t",i,p,I[i][p]);
		}
		fprintf(op2,"\n");
	}
	
//Scanning & Printing Target output for Output layer

	fprintf(op2,"\nTarget Output matrix of order %dX%d :\n",P,N);
	
	for(p=1;p<=P;p++)
	{
		for(k=1;k<=N;k++)
		{
			fscanf(ip,"%lf",&TO[k][p]);
			fprintf(op2,"TO[%d][%d]:%lf\t",k,p,TO[k][p]);
		}
		fprintf(op2,"\n");
	}

//Normalization of Inputs for Input layer

	for(i=1;i<=L;i++)
	{
		max_I[i]=-1000;min_I[i]=1000;
		
		for(p=1;p<=P;p++)
		{
			if(I[i][p]>max_I[i])
			max_I[i]=I[i][p];
			if(I[i][p]<min_I[i])
			min_I[i]=I[i][p];
		}
	}
	
	for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			I[i][p]=0.1+0.8*((I[i][p]-min_I[i])/(max_I[i]-min_I[i]));
		}
	}
	
	fprintf(op2,"\nNormalized I matrix of order %dX%d :\n",L,P);
	
	for(i=1;i<=L;i++)
	{
		for(p=1;p<=P;p++)
		{
			fprintf(op2,"%f\t",I[i][p]);
		}
		fprintf(op2,"\n");
	}


//Normalization of Target output for Output layer

for(k=1;k<N+1;k++)
	{
		max_TO[k]=-1000;min_TO[k]=1000;
		for(p=1;p<=P;p++)
		{
			if(TO[k][p]>max_TO[k])
			max_TO[k]=TO[k][p];
			if(TO[k][p]<min_TO[k])
			min_TO[k]=TO[k][p];			
		}
	}
	
	for(p=1;p<=P;p++)
	{
		for(k=1;k<=N;k++)
		{
			TO[k][p]=-0.1+(1.0*((TO[k][p]-min_TO[k])/(max_TO[k]-min_TO[k])));
		}
	}

	fprintf(op2,"\nNormalized Target Ouput matrix of order %dX%d :\n",P,N);
	
	for(p=1;p<=P;p++)
	{
		for(k=1;k<N+1;k++)
		{
			fprintf(op2,"TO[%d][%d]:%lf\t",k,p,TO[k][p]);
		}
		fprintf(op2,"\n");
	}
	
//Define V

	fprintf(op2,"\nV matrix of order %dX%d :\n",L+1,M);

	for(i=0;i<L+1;i++)
	{
		for(j=1;j<=M;j++)
		{
			if(i==0)
			{
				V[i][j]=0;
			}
			else
			{
				V[i][j]=1.0*rand()/RAND_MAX;
			}
		}
	}
	
	for(i=0;i<=L;i++)
	{
		for(j=1;j<=M;j++)
		{
			fprintf(op2,"V[%d][%d]:%f\t",i,j,V[i][j]);
		}
		fprintf(op2,"\n");
	}
	fprintf(op2,"\n");

//Define W

	fprintf(op2,"\nW matrix of order %dX%d :\n",M+1,N);
	
	for(j=0;j<M+1;j++)
	{
		for(k=1;k<=N;k++)
		{
			if(j==0)
			{
				W[i][j]=0;
			}
			else
			{
			W[j][k]=1.0*rand()/RAND_MAX;
			}
		}
	}
	
	for(j=0;j<M+1;j++)
	{
		for(k=1;k<=N;k++)
		{
			fprintf(op2,"W[%d][%d]:%f\t",j,k,W[j][k]);
		}
		fprintf(op2,"\n");
	}

//Do-While loop TRAINING of Patterns

	do
	{		
		//Calculation for forward pass
		
		for(p=1;p<=P-9;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+(1.0);
				OH[j][p]=1/(1+exp(-IH[j][p]));
				IH[j][p]=0;
			}
			
		}
		//fprintf(op1,"\n");
		
		//Calculation for Output of Output layer
		for(p=1;p<=P-9;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<M+1;j++)
				{
					IO[k][p]=IO[k][p]+OH[j][p]*W[j][k];
				}
				IO[k][p]=IO[k][p]+1.0;
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
				IO[k][p]=0;
			}
		}
		//fprintf(op1,"\n");
		
		//Calculations del_W_jk
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				delta_W[j][k]=0;
				for(p=1;p<=P-9;p++)
				{
					delta_W[j][k]=delta_W[j][k]+((0.3/P)*(TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*OH[j][p]);
				}
				fprintf(op1,"delta_W[%d][%d]=%lf\t",j,k,delta_W[j][k]);
			}
			fprintf(op1,"\n");
		}

		fprintf(op1,"\n");
		
		//Calcualtions delta_V_ij
		for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				delta_V[i][j]=0;
				for(p=1;p<=P-9;p++)
				{
					for(k=1;k<=N;k++)
					{
						delta_V[i][j]=delta_V[i][j]+((0.3/(P*N))*((TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*W[j][k]*OH[j][p]*(1-OH[j][p])*I[i][p]));
					}
				}
				fprintf(op1,"delta_V[%d][%d]=%lf\t",i,j,delta_V[i][j]);
			}
			fprintf(op1,"\n");
		}
		
		//Calcualtion for e
		MSE=0;
		for(p=1;p<=P-9;p++)
		{
			for(k=1;k<=N;k++)
			{
				e=pow((TO[k][p]-OO[k][p]),2)/2;
				MSE=MSE+e;
			}
		}
		MSE=MSE/P;
		fprintf(op1,"\nMSE=%f\tIteration=%d\n",MSE,counter);
		
		//Updating values of Vij
		for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				V[i][j]=V[i][j]+delta_V[i][j];
				fprintf(op1,"V[%d][%d]:%f\t",i,j,V[i][j]);
			}

			fprintf(op1,"\n");
		}
		fprintf(op1,"\n");
		
		//Updating values of Wjk
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				W[j][k]=W[j][k]+delta_W[j][k];
				fprintf(op1,"W[%d][%d]:%f\t",j,k,W[j][k]);
			}
			fprintf(op1,"\n");
		}
		
	printf("\nIteration %d completed",counter);
	
	counter++;

	
	}while(MSE>TOL);

	for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				fprintf(op2,"V[%d][%d]:%f\t",i,j,V[i][j]);
			}
			fprintf(op2,"\n");
		}
		fprintf(op2,"\n");
		
		//Updating values of Wjk
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				fprintf(op2,"W[%d][%d]:%f\t",j,k,W[j][k]);
			}
			fprintf(op2,"\n");
		}
		
	//TESTING of Patterns
	
		//Calculation for forward pass
		for(p=17;p<=25;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+1.0;
				OH[j][p]=1/(1+exp(-IH[j][p]));
				fprintf(op2,"\nIH[%d][%d]:%f\tOH[%d][%d]:%f",j,p,IH[j][p],j,p,OH[j][p]);
				IH[j][p]=0;
			}
			
		}

		fprintf(op2,"\n\n");
		
		//Calculation for Output of Output layer
		for(p=17;p<=25;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<=M+1;j++)
				{
					IO[k][p]=IO[k][p]+OH[j][p]*W[j][k];
				}
				IO[k][p]=IO[k][p]+1.0;
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
				fprintf(op2,"\nIO[%d][%d]:%f\tOO[%d][%d]:%f\tTO[%d][%d]:%f\tError =%f",k,p,IO[k][p],k,p,OO[k][p],k,p,TO[k][p],fabs(OO[k][p]-TO[k][p]));
				IO[k][p]=0;
			}
		}
	
	fclose(ip);
	fclose(op1);
	fclose(op2);

	return 0;
}
