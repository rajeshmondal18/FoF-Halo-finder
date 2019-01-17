#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<fftw3.h>
#include<omp.h>

#include "nbody.h"

//*******************************************************************************
//                    global  variables from Nbody_comp 
//*******************************************************************************

float  vhh, // Hubble parameter in units of 100 km/s/Mpc
  vomegam, // Omega_matter; total matter density (baryons+CDM) parameter
  vomegalam, // Cosmological Constant 
  vomegab, //Omega_baryon
  sigma_8_present ,//  Last updated value of sigma_8 (Presently WMAP)
  vnn; // Spectral index of primordial Power spectrum

long N1,N2,N3;// box dimension (grid) 
int NF, // Fill every NF grid point 
  Nbin; // Number of bins to calculate final P(k) (output)

float   LL; // grid spacing in Mpc

long    MM; // Number of particles
// global variables  (calculated )

int zel_flag=1, // memory allocation for zel is 3 times that for nbody
  fourier_flag;//for fourier transfrom
float  DM_m, // Darm matter mass of simulation particle in 10^10 M_sun h^-1 unit
  norm, // normalize Pk
  pi=M_PI;

io_header    header1;

// arrays for storing data
float ***ro; // for density/potential
fftwf_plan p_ro; // for FFT
fftwf_plan q_ro; // for FFT

float Z; //redshift of the output
char file[100], file1[100], file2[100], num[8], num1[8], num2[8];

//*******************************************************************************
//                    done global variables from Nbody_comp 
//*******************************************************************************


//*******************************************************************************
//                        global variables for FOF
//*******************************************************************************
float Lfof=0.2;   // linking length in Grid units                     

int count, // number of elements in a cluster 
  Nmin=10, // minimum particles  in a clusters (count >= Nmin)
  left_flag[3], // =1 if cluster touches left boundary else =0
  right_flag[3]; // =1 if cluster touches right boundary else =0

//*******************************************************************************
//                     done global variables for FOF
//*******************************************************************************

//*******************************************************************************
//                  new variable type definition for FOF
//       stores particle position,velocity and pointer to next particle 
//*******************************************************************************
struct particle
{
  float x[3];
  float v[3];
  struct particle *next;
};
//*******************************************************************************
//                  done new variable type definition for FOF
//*******************************************************************************

//*******************************************************************************
//            more global variables for FOF - pointers to particles
//*******************************************************************************  

struct particle ****grid; // grid of pointers to particle linked lists 
struct particle *data; // store entire particle positio, and velocity 
struct particle *cluster; // head of  linked list of particles in a clusterer
struct particle *cctest; // current particle in the cluster linked list 
struct particle *gptest; // previous particle in linked list attached to  grid
struct particle *gctest; // current particle in linked list attached to  grid
struct particle *tail; // last particle in the cluster linked list 
struct particle *tmp; // temporary pointer

//*******************************************************************************
//         done more global variables for FOF - pointers to particles  
//*******************************************************************************

//*******************************************************************************
//              allocate memory for n1xn2xn3 3D grid of pointers
//*******************************************************************************

struct particle ****allocate_particle_pointer_3d(long n1,long n2,long n3)
{
  long ii,jj,kk;
  long size,index;
  struct particle ****p1, **p2;
  
  p1=(struct particle ****) malloc (n1 * sizeof(struct particle ***));
  
  for(ii=0;ii<n1;ii++)
    p1[ii]=(struct particle ***) malloc (n2 *  sizeof(struct particle **));
  
  size=n1*n2*n3;
  
  if(!(p2 = (struct particle **) malloc (size*sizeof(struct particle*))))
    {
      printf("Error in allocate particle pointer 3d");
      exit(0);
    }
  
  for(ii=0;ii<n1;ii++)
    for(jj=0;jj<n2;jj++)
      {
	index=ii*n2*n3+jj*n3;
	p1[ii][jj]=p2 + index;
      }
  
  return(p1);
}

//*******************************************************************************
//             done allocate memory for n1xn2xn3 3D grid of pointers
//*******************************************************************************


//*******************************************************************************
//  distance between 2 particles with periodic bondary conditions(in grid units)
//*******************************************************************************

float dist (struct particle *a, struct particle *b)
{
  int i;
  long N;
  float dx[3] ;
  
  for (i=0;i<3;i++)
    {
      N=N1*(1-i)*(2-i)/2 + N2*i*(2-i) + N3*i*(i-1)/2;
      dx[i] = fabs(a->x[i] - b->x[i]) ;
      dx[i] = (dx[i] > (N - 1)*LL) ? (N*LL - dx[i]) : dx[i] ;
    }
  
  return(sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]));
}

//*******************************************************************************
//   done find distance between 2 particles with periodic bondary conditions
//*******************************************************************************


//*******************************************************************************
//         move particle from grid linked list  to cluster linked list
//*******************************************************************************

void move(long p, long q, long r)      // p,q,r -> grid point
{
  tail->next = gctest;
  
  if(gptest == NULL) // move first particle in grid linked list
    {
      grid[p][q][r] = (grid[p][q][r])->next;
      gctest = grid[p][q][r];
    }
  else // move the particle pointed to by gptest
    {
      gptest->next = gctest->next;
      gctest = gptest->next;
    }
  tail = tail->next;
  tail->next = NULL;
}

//***********************************************************************************
//       done move particle from grid linked list  to cluster linked list
//***********************************************************************************

//***********************************************************************************
//  for all elements in cluster  linked list  - find friends in grid linked list  and
//                 use move() to move friend from grid to cluster
//***********************************************************************************

void findfriends()
{
  int u, v, w;
  long  i, j, k, ib[3], N;
  cctest = cluster;   // cctest is pointer to current particle in cluster linked list. start at cluster head
  tail = cluster;   // last element of cluster linked list
  tail->next = NULL;
  
  while(cctest != NULL)
    {
      count++; // number of elements in cluster
      
      for (i=0;i<3;i++)
	{
	  ib[i] = (long)floor((cctest->x[i])/NF);   // grid point corresponding to cctest
	  
	  
	  // check if element is in first or last grid - for calculating cluster center of mass
	  N=N1*(1-i)*(2-i)/2 + N2*i*(2-i) + N3*i*(i-1)/2;
	  
	  
	  if(ib[i]< 1)
	    left_flag[i] = 1;
	  if(ib[i] >(N-2))
	    right_flag[i] = 1;
	  
	  // done checking
	}
      
      // loop over neighbouring grid points for ib[]
      
      for (u=-1;u<=1;u++)
	{
	  i = (ib[0] + u + N1/NF)%(N1/NF);
	  for (v=-1;v<=1;v++)
	    {
	      j = (ib[1] + v + N2/NF)%(N2/NF);
	      for (w=-1;w<=1;w++)
		{
		  k = (ib[2] + w + N3/NF)%(N3/NF);
		  
		  //   find friend in linked attached to  grid point (i,j,k)
		  
		  gptest = NULL;    // previous position on grid linked list
		  gctest = grid[i][j][k];  // current  position on grid linked list
		  
		  while(gctest != NULL)
		    {
		      if(dist(cctest, gctest) < (Lfof*NF)) // check if freind
			move(i, j, k);   // move to cluster if friend;  and increment gptest and gctest
		      else
			{
			  gptest = gctest;   // increment gptest and gctest if not friend
			  gctest = gctest->next;
			}
		    }
		}
	    }
	}
      cctest = cctest->next;   // increment to next element on cluster
    }
}
//***********************************************************************************
//                                done findfriends
//***********************************************************************************

//***********************************************************************************
//        read unsorted halo catalogue and write sorted halo catalogue
//***********************************************************************************

int cmpfunc (const void* a, const void* b) // compare two variable for shorting
{
  return(*(int*)b - *(int*)a);
}


void write_fof(long t, float **clust)
{
  float  tmp1[7];
  long i, j, k;
  int dummy;

  FILE *read, *fp1;

  strcpy(file1,"clusters_");
  sprintf(num1,"%3.3f",Z);
  strcat(file1,num1);

  read = fopen(file1,"r");

  for(i=0; i<t; i++)
    fscanf(read,"%f\t%f\t%f\t%f\t%f\t%f\t%f", &clust[i][0], &clust[i][1], &clust[i][2], &clust[i][3], &clust[i][4], &clust[i][5], &clust[i][6]);
  
  fclose(read);
  
  //--------------------sort clusters by mass------------------------------------
  
  qsort(clust[0], t , sizeof(tmp1), cmpfunc);  // sort clusters by mass
  
  //-------------------------- sort done--------------------------------------
  
  strcpy(file2,"halo_catalogue_");
  sprintf(num2,"%3.3f",Z);
  strcat(file2,num2);
  
  fp1 = fopen(file2,"w");
  
  //---------------------------write header----------------------------------------
  
  fwrite(&dummy,sizeof(dummy),1,fp1);
  fwrite(&header1,sizeof(io_header),1,fp1);
  fwrite(&dummy,sizeof(dummy),1,fp1);
  
  // header  written
  
  // writing total cluster
  fwrite(&dummy,sizeof(dummy),1,fp1);
  fwrite(&t,sizeof(long),1,fp1);
  fwrite(&dummy,sizeof(dummy),1,fp1);
  
  // writing data
  fwrite(&dummy,sizeof(dummy),1,fp1);
  
  for(i=0; i<t; i++)
    fwrite(&clust[i][0],sizeof(float),7,fp1);
  
  fwrite(&dummy,sizeof(dummy),1,fp1);
  
  fclose(fp1);
}

//***********************************************************************************
//                       done write sorted halo catalogue
//***********************************************************************************

//************************************************************************************
//                                 main
//***********************************************************************************

main()
{
  int l,ii,jj;
  long i, j, k, ia[3], N;
  long ll, totstar, totcluster;
  long int seed;
  float vaa, x_v[6], dummy_xv[6]={-1.,-1.,-1.,-1.,-1.,-1.};
  int output_flag, in_flag;
  float **rra, **vva;
  
  float **clust;
  
  FILE  *inp;
  int Noutput;
  float *nz;
  
  double clusterx[3], clusterv[3];
  
  double t,T=omp_get_wtime();
  
  /*---------------------------------------------------------------------------*/
  /* Read input parameters for the simulation from the file "input.nbody_comp" */
  /*---------------------------------------------------------------------------*/
  inp=fopen("input.nbody_comp","r");
  fscanf(inp,"%*ld%*d");
  fscanf(inp,"%*f%*f%*f%*f");
  fscanf(inp,"%*f%*f");
  fscanf(inp,"%*ld%*ld%*ld%*ld%*f");
  fscanf(inp,"%*f%*d%*d%*d");
  fscanf(inp,"%*f");  /* time step, final scale factor*/
  fscanf(inp,"%d",&Noutput);
  
  nz=(float*)calloc(Noutput,sizeof(float)); // array to store Noutput
  
  for(ii=0;ii<Noutput;ii++)
    fscanf(inp,"%f",&nz[ii]);
  
  fclose(inp);
  
  /*-----------------------------------------------------------*/
  
  for(jj=0;jj<Noutput;jj++)
    {
      t=omp_get_wtime();
      totstar = 0;
      totcluster = 0;
      
      strcpy(file,"output.nbody_");
      sprintf(num,"%3.3f",nz[jj]);
      strcat(file,num);
      
      read_output(file,1,&seed,&output_flag,&in_flag,rra,vva,&vaa); // only read header
      if(jj==0)
  	{
  	  rra= allocate_float_2d(MM,3);
  	  vva= allocate_float_2d(MM,3);
  	}
      read_output(file,2,&seed,&output_flag,&in_flag,rra,vva,&vaa); // read data
      
      NF=(int)round(pow(1.*N1*N2*N3/MM,1./3.));  // if 1, particle in every NF^3 grid point
      
      printf("(%3.3f)nbody output_flag=%d, NF=%d\n", nz[jj], output_flag, NF);
      
      /*-----stores particles position, velocity and pointer to next particle -----------*/
      
      if(jj==0)
  	data = (struct particle *) calloc (MM, sizeof(struct particle)); // memory allocation for the data

      
      /*----------------copy  particle position and velocity to 'data'-------------------*/
      
      for(ll=0;ll<MM;ll++)
  	{
  	  for(l=0; l<3; l++)
  	    {
  	      data[ll].x[l] = rra[ll][l];
  	      data[ll].v[l] = vva[ll][l];
  	    }
  	  data[ll].next= NULL;
  	}
      
      //***********************************************************************************
      /*-----------------done copy  particle position and velocity ----------------------*/
      
      //***********************************************************************************
      //                        grid for linked lists
      //***********************************************************************************
      if(jj==0)
        grid = allocate_particle_pointer_3d(N1/NF, N2/NF, N3/NF); // memory allocation for all the grid points
      
      /*------------------initialization of grid-------------------*/
      for (i=0;i<N1/NF;i++)
  	for (j=0;j<N2/NF;j++)
  	  for (k=0;k<N3/NF;k++)
  	    grid[i][j][k]=NULL;
      
      
      //***********************************************************************************
      //                           make grid linked list
      //***********************************************************************************
      
      for(ll=0;ll<MM;ll++)
  	{
  	  for(i=0;i<3;++i)
  	    ia[i]=(long)floor((data[ll].x[i])/NF); // new grid units
	  
  	  data[ll].next = grid[ia[0]][ia[1]][ia[2]];  // attach to grid linked list
  	  grid[ia[0]][ia[1]][ia[2]] = &data[ll];
  	}
      
      //***********************************************************************************
      //                         done make grid linked list
      //***********************************************************************************
      
      //***********************************************************************************
      
      FILE  *dat1,*dat2;
      //dat1 = fopen("clust_element","w");

      strcpy(file1,"clusters_");
      sprintf(num1,"%3.3f",nz[jj]);
      strcat(file1,num1);

      dat2 = fopen(file1,"w");
      
      //***********************************************************************************
      
      for (i=0;i<N1/NF;i++)
  	for (j=0;j<N2/NF;j++)
  	  for (k=0;k<N3/NF;k++)
  	    while (grid[i][j][k] != NULL)
  	      {
  		count = 0; // number of elements in a cluster
		
  		for(l=0;l<3;l++)
  		  {
  		    left_flag[l] = 0;
  		    right_flag[l] = 0;
  		    clusterx[l] = 0.0;
  		    clusterv[l] = 0.0;
  		  }
		
  		/*******************************************/
		
  		cluster = grid[i][j][k];   // first element of grid linked list sent to cluster head
  		grid[i][j][k] = grid[i][j][k]->next;
		
  		findfriends(); // find friends starting from  cluster head
		
  		if (count >= Nmin)
  		  {
  		    /***** for cluster position and velocity calculation *****/
  		    tmp=cluster;
  		    while(tmp != NULL)
  		      {
  			for(l=0;l<3;l++)
  			  {
  			    N=N1*(1-l)*(2-l)/2 + N2*l*(2-l) + N3*l*(l-1)/2;
			    
  			    /***** for cluster position calculation *****/
  			    // wrap around if cluster spans 2 boundaries
			    
  			    if(left_flag[l] !=0 && right_flag[l] !=0)
  			      clusterx[l] += (double)((tmp->x[l] < N/2.0) ? (tmp->x[l]+N) : tmp->x[l]);
  			    else
  			      clusterx[l] += (double)tmp->x[l];
			    
  			    clusterv[l] += (double)tmp->v[l]; // calculating cluster velocity
			    
			    
  			    /* if(output_flag != 1) */
  			    /*   { */
  			    /*     x_v[l]=tmp->x[l]*LL*1000.0*vhh;   //coordinates in kpc/h */
  			    /*     x_v[l+3]=tmp->v[l]*LL*vhh*100./vaa;   //peculiar velocities in km/sec */
  			    /*   } */
  			    /* else */
  			    /*   { */
  			    /*     x_v[l]=tmp->x[l];   //coordinates in N-body grid unit */
  			    /*     x_v[l+3]=tmp->v[l]; */
  			    /*   } */
  			  }
  			/* fwrite(&x_v[0],sizeof(float),6,dat1);   //for cluster element printing */
			
  			tmp=tmp->next;
  		      }
  		    /* fwrite(&dummy_xv[0],sizeof(float),6,dat1);  */
		    
  		    /***** for cluster position calculation *****/
  		    for(l=0;l<3;l++)
  		      {
  			N=N1*(1-l)*(2-l)/2 + N2*l*(2-l) + N3*l*(l-1)/2;
  			clusterx[l] = clusterx[l]/(1.0*count) + N;
  			clusterx[l] = clusterx[l] - N*floor(clusterx[l]/N); //Periodic boundary conditions appliying
			
  			clusterv[l] = clusterv[l]/(1.0*count);
			
  			if(output_flag != 1)
  			  {
  			    clusterx[l]=clusterx[l]*LL*1000.0*vhh;   //coordinates in kpc/h
  			    clusterv[l]=clusterv[l]*LL*vhh*100./vaa;   //peculiar velocities in km/sec
  			  }
  		      }
  		    fprintf(dat2,"%e %e %e %e %e %e %e\n", count*DM_m, clusterx[0], clusterx[1], clusterx[2], clusterv[0], clusterv[1], clusterv[2]);
		    
  		    totcluster++ ;
  		  }
  		totstar = totstar + count ;
  	      }
      /* fclose(dat1); */
      if( totstar!=MM)
  	printf("Particle number not matching. totstar=%ld and MM=%ld\n",totstar,MM);
      
      fclose(dat2);
      
      printf("Total number of clusters=%ld\n",totcluster);
      
      /*-----------------------------------------------------------*/
      Z=nz[jj];
      
      clust = allocate_float_2d(totcluster,7);
      
      write_fof(totcluster, clust);   // print sorted halo catalogue
      
      free(clust);
      
      /*-----------------------------------------------------------*/
      printf("(for %3.3f) Time taken = %dhr %dmin %dsec\n",nz[jj],(int)((omp_get_wtime()-t)/3600), (int)((omp_get_wtime()-t)/60)%60, (int)(omp_get_wtime()-t)%60);
    }
  free(rra);
  free(vva);
  free(data);
  free(grid);
  
  system("rm clusters*");
      
  printf("done FoF. Total time taken = %dhr %dmin %dsec\n",(int)((omp_get_wtime()-T)/3600), (int)((omp_get_wtime()-T)/60)%60, (int)(omp_get_wtime()-T)%60);
}
