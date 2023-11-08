/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 09/29/2021
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
CALI_CXX_MARK_FUNCTION;

int sizeOfMatrix;
if (argc == 2)
{
    sizeOfMatrix = atoi(argv[1]);
}
else
{
    printf("\n Please provide the size of the matrix");
    return 0;
}
int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */
double	a[sizeOfMatrix][sizeOfMatrix],           /* matrix A to be multiplied */
	b[sizeOfMatrix][sizeOfMatrix],           /* matrix B to be multiplied */
	c[sizeOfMatrix][sizeOfMatrix];           /* result matrix C */
MPI_Status status;
double worker_receive_time,       /* Buffer for worker recieve times */
   worker_calculation_time,      /* Buffer for worker calculation times */
   worker_send_time = 0;         /* Buffer for worker send times */
double whole_computation_time,    /* Buffer for whole computation time */
   master_initialization_time,   /* Buffer for master initialization time */
   master_send_receive_time = 0; /* Buffer for master send and receive time */
/* Define Caliper region names */
const char* whole_computation = "whole_computation";
const char* master_initialization = "master_initialization";
const char* master_send_recieve = "master_send_recieve";
const char* worker_recieve = "worker_recieve";
const char* worker_calculation = "worker_calculation";
const char* worker_send = "worker_send";

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;

MPI_Comm worker_comm;
MPI_Comm_split(MPI_COMM_WORLD, (taskid == MASTER) ? MPI_UNDEFINED : 0, taskid, &worker_comm);

    
// WHOLE PROGRAM COMPUTATION PART STARTS HERE
CALI_MARK_BEGIN(whole_computation);
double whole_computation_time_start = MPI_Wtime();

// Create caliper ConfigManager object
cali::ConfigManager mgr;
mgr.start();

/**************************** master task ************************************/
   if (taskid == MASTER)
   {
   
      // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE

      printf("mpi_mm has started with %d tasks.\n",numtasks);
      printf("Initializing arrays...\n");

      CALI_MARK_BEGIN(master_initialization); // Don't time printf
      double master_initialization_start_time = MPI_Wtime();

      for (i=0; i<sizeOfMatrix; i++)
         for (j=0; j<sizeOfMatrix; j++)
            a[i][j]= i+j;
      for (i=0; i<sizeOfMatrix; i++)
         for (j=0; j<sizeOfMatrix; j++)
            b[i][j]= i*j;
            
      //INITIALIZATION PART FOR THE MASTER PROCESS ENDS HERE
      CALI_MARK_END(master_initialization);
      double master_initialization_end_time = MPI_Wtime();
      master_initialization_time = master_initialization_end_time - master_initialization_start_time;
      //SEND-RECEIVE PART FOR THE MASTER PROCESS STARTS HERE
      CALI_MARK_BEGIN(master_send_recieve);
      double master_send_receive_time_start = MPI_Wtime();

      /* Send matrix data to the worker tasks */
      averow = sizeOfMatrix/numworkers;
      extra = sizeOfMatrix%numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++)
      {
         rows = (dest <= extra) ? averow+1 : averow;   	
         printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&a[offset][0], rows*sizeOfMatrix, MPI_DOUBLE, dest, mtype,
                   MPI_COMM_WORLD);
         MPI_Send(&b, sizeOfMatrix*sizeOfMatrix, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + rows;
      }

      /* Receive results from worker tasks */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&c[offset][0], rows*sizeOfMatrix, MPI_DOUBLE, source, mtype, 
                  MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
      }
      
      //SEND-RECEIVE PART FOR THE MASTER PROCESS ENDS HERE
      CALI_MARK_END(master_send_recieve);
      double master_send_receive_time_end = MPI_Wtime();
      master_send_receive_time = master_send_receive_time_end - master_send_receive_time_start;


      /* Print results - you can uncomment the following lines to print the result matrix */
      /*
      printf("******************************************************\n");
      printf("Result Matrix:\n");
      for (i=0; i<sizeOfMatrix; i++)
      {
         printf("\n"); 
         for (j=0; j<sizeOfMatrix; j++) 
            printf("%6.2f   ", c[i][j]);
      }
      printf("\n******************************************************\n");
      printf ("Done.\n");
      */
      
      
   }


/**************************** worker task ************************************/
   if (taskid > MASTER)
   {
      //RECEIVING PART FOR WORKER PROCESS STARTS HERE
      CALI_MARK_BEGIN(worker_recieve);
      double worker_receive_time_start = MPI_Wtime();


      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&a, rows*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&b, sizeOfMatrix*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      
      //RECEIVING PART FOR WORKER PROCESS ENDS HERE
      CALI_MARK_END(worker_recieve);
      double worker_receive_time_end= MPI_Wtime();
      worker_receive_time = worker_receive_time_end - worker_receive_time_start;

      //CALCULATION PART FOR WORKER PROCESS STARTS HERE
      CALI_MARK_BEGIN(worker_calculation);
      double worker_calc_time_start = MPI_Wtime();

      for (k=0; k<sizeOfMatrix; k++)
         for (i=0; i<rows; i++)
         {
            c[i][k] = 0.0;
            for (j=0; j<sizeOfMatrix; j++)
               c[i][k] = c[i][k] + a[i][j] * b[j][k];
         }
         
      //CALCULATION PART FOR WORKER PROCESS ENDS HERE
      CALI_MARK_END(worker_calculation);
      double worker_calc_time_end = MPI_Wtime();
      worker_calculation_time = worker_calc_time_end - worker_calc_time_start;

      //SENDING PART FOR WORKER PROCESS STARTS HERE
      CALI_MARK_BEGIN(worker_send);
      double worker_send_start_time = MPI_Wtime();


      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&c, rows*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

      //SENDING PART FOR WORKER PROCESS ENDS HERE
      CALI_MARK_END(worker_send);
      double worker_send_end_time = MPI_Wtime();
      worker_send_time = worker_send_end_time - worker_send_start_time;
   }

double whole_computation_time_end = MPI_Wtime();
CALI_MARK_END(whole_computation);
whole_computation_time = whole_computation_time_end - whole_computation_time_start;
   adiak::init(NULL);
   adiak::user();
   adiak::launchdate();
   adiak::libraries();
   adiak::cmdline();
   adiak::clustername();
   adiak::value("num_procs", numtasks);
   adiak::value("matrix_size", sizeOfMatrix);
   adiak::value("program_name", "master_worker_matrix_multiplication");
   adiak::value("matrix_datatype_size", sizeof(double));

   double worker_receive_time_max,
      worker_receive_time_min,
      worker_receive_time_sum,
      worker_recieve_time_average,
      worker_calculation_time_max,
      worker_calculation_time_min,
      worker_calculation_time_sum,
      worker_calculation_time_average,
      worker_send_time_max,
      worker_send_time_min,
      worker_send_time_sum,
      worker_send_time_average = 0; // Worker statistic values.

   /* USE MPI_Reduce here to calculate the minimum, maximum and the average times for the worker processes.
   MPI_Reduce (&sendbuf,&recvbuf,count,datatype,op,root,comm). https://hpc-tutorials.llnl.gov/mpi/collective_communication_routines/ */
   if (taskid != MASTER) {
      MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, worker_comm);
      MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, worker_comm);
      MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, worker_comm);

      MPI_Reduce(&worker_calculation_time, &worker_calculation_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, worker_comm);
      MPI_Reduce(&worker_calculation_time, &worker_calculation_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, worker_comm);
      MPI_Reduce(&worker_calculation_time, &worker_calculation_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, worker_comm);

      MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, worker_comm);
      MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, worker_comm);
      MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, worker_comm);

   }
   
   if (taskid == 0)
   {
      // Master Times
      printf("******************************************************\n");
      printf("Master Times:\n");
      printf("Whole Computation Time: %f \n", whole_computation_time);
      printf("Master Initialization Time: %f \n", master_initialization_time);
      printf("Master Send and Receive Time: %f \n", master_send_receive_time);
      printf("\n******************************************************\n");

      // Add values to Adiak
      adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
      adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
      adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

      // Must move values to master for adiak
      mtype = FROM_WORKER;
      MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_recieve_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

      adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
      adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
      adiak::value("MPI_Reduce-worker_recieve_time_average", worker_recieve_time_average);
      adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
      adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
      adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
      adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
      adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
      adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
   }
   else if (taskid == 1)
   { // Print only from the first worker.
      // Print out worker time results.
      
      // Compute averages after MPI_Reduce
      worker_recieve_time_average = worker_receive_time_sum / (double)numworkers;
      worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
      worker_send_time_average = worker_send_time_sum / (double)numworkers;

      printf("******************************************************\n");
      printf("Worker Times:\n");
      printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
      printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
      printf("Worker Receive Time Average: %f \n", worker_recieve_time_average);
      printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
      printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
      printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
      printf("Worker Send Time Max: %f \n", worker_send_time_max);
      printf("Worker Send Time Min: %f \n", worker_send_time_min);
      printf("Worker Send Time Average: %f \n", worker_send_time_average);
      printf("\n******************************************************\n");

      mtype = FROM_WORKER;
      MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_recieve_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_send_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }

   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();
}