#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

//RUNNING MEDIAN CODE START
typedef float Item;
typedef struct Mediator_t
{
  Item* data;  //circular queue of values
  int*  pos;   //index into `heap` for each value
  int*  heap;  //max/median/min heap holding indexes into `data`.
  int   N;     //allocated size.
  int   idx;   //position in circular queue
  int   minCt; //count of items in min heap
  int   maxCt; //count of items in max heap
} Mediator;
 
/*--- Helper Functions ---*/
 
//returns 1 if heap[i] < heap[j]
inline int mmless(Mediator* m, int i, int j)
{
  return (m->data[m->heap[i]] < m->data[m->heap[j]]);
}
 
//swaps items i&j in heap, maintains indexes
int mmexchange(Mediator* m, int i, int j)
{
  int t = m->heap[i];
  m->heap[i]=m->heap[j];
  m->heap[j]=t;
  m->pos[m->heap[i]]=i;
  m->pos[m->heap[j]]=j;
  return 1;
}
 
//swaps items i&j if i<j;  returns true if swapped
inline int mmCmpExch(Mediator* m, int i, int j)
{
  return (mmless(m,i,j) && mmexchange(m,i,j));
}
 
//maintains minheap property for all items below i.
void minSortDown(Mediator* m, int i)
{
  for (i*=2; i <= m->minCt; i*=2)
    {  if (i < m->minCt && mmless(m, i+1, i)) { ++i; }
      if (!mmCmpExch(m,i,i/2)) { break; }
    }
}
 
//maintains maxheap property for all items below i. (negative indexes)
void maxSortDown(Mediator* m, int i)
{
  for (i*=2; i >= -m->maxCt; i*=2)
    {  if (i > -m->maxCt && mmless(m, i, i-1)) { --i; }
      if (!mmCmpExch(m,i/2,i)) { break; }
    }
}
 
//maintains minheap property for all items above i, including median
//returns true if median changed
inline int minSortUp(Mediator* m, int i)
{
  while (i>0 && mmCmpExch(m,i,i/2)) i/=2;
  return (i==0);
}
 
//maintains maxheap property for all items above i, including median
//returns true if median changed
inline int maxSortUp(Mediator* m, int i)
{
  while (i<0 && mmCmpExch(m,i/2,i))  i/=2;
  return (i==0);
}
/*--- Public Interface ---*/
 
 
//creates new Mediator: to calculate `nItems` running median. 
//mallocs single block of memory, caller must free.
Mediator* MediatorNew(int nItems)
{
  int size = sizeof(Mediator)+nItems*(sizeof(Item)+sizeof(int)*2);
  Mediator* m=  malloc(size);
  m->data= (Item*)(m+1);
  m->pos = (int*) (m->data+nItems);
  m->heap = m->pos+nItems + (nItems/2); //points to middle of storage.
  m->N=nItems;
  m->minCt = m->maxCt = m->idx = 0;
  while (nItems--)  //set up initial heap fill pattern: median,max,min,max,...
    {  m->pos[nItems]= ((nItems+1)/2) * ((nItems&1)?-1:1);
      m->heap[m->pos[nItems]]=nItems;
    }
  return m;
}
 
 
//Inserts item, maintains median in O(lg nItems)
void MediatorInsert(Mediator* m, Item v)
{
  int p = m->pos[m->idx];
  Item old = m->data[m->idx];
  m->data[m->idx]=v;
  m->idx = (m->idx+1) % m->N;
  if (p>0)         //new item is in minHeap
    {  if (m->minCt < (m->N-1)/2)  { m->minCt++; }
      else if (v>old) { minSortDown(m,p); return; }
      if (minSortUp(m,p) && mmCmpExch(m,0,-1)) { maxSortDown(m,-1); }
    }
  else if (p<0)   //new item is in maxheap
    {  if (m->maxCt < m->N/2) { m->maxCt++; }
      else if (v<old) { maxSortDown(m,p); return; }
      if (maxSortUp(m,p) && m->minCt && mmCmpExch(m,1,0)) { minSortDown(m,1); }
    }
  else //new item is at median
    {  if (m->maxCt && maxSortUp(m,-1)) { maxSortDown(m,-1); }
      if (m->minCt && minSortUp(m, 1)) { minSortDown(m, 1); }
    }
}
 
//returns median item (or average of 2 when item count is even)
Item MediatorMedian(Mediator* m)
{
  Item v= m->data[m->heap[0]];
  if (m->minCt<m->maxCt) { v=(v+m->data[m->heap[-1]])/2; }
  return v;
}

void runningMedian(float* inbuffer, float* outbuffer, int window, int nsamps)
{
  int ii;
  Mediator* m = MediatorNew(window);
  for(ii=0;ii<nsamps;ii++){
    MediatorInsert(m,inbuffer[ii]);
    outbuffer[ii]= inbuffer[ii] - (float) MediatorMedian(m);
  }
}
//RUNNING MEDIAN CODE END

void runningMean(float* inbuffer,
                 float* outbuffer,
                 int window,
		 int nsamps)

{
  int ii;
  double sum = 0;
  for (ii=0;ii<nsamps;ii++){
    sum += inbuffer[ii];
    if (ii<window)
      outbuffer[ii] = inbuffer[ii] - (float) sum/(ii+1);
    else {
      outbuffer[ii] = inbuffer[ii] - (float) sum/(window+1);
      sum -= inbuffer[ii-window];
    }
  }
}

void runBoxcar(float* inbuffer,
	       float* outbuffer,
	       int window,
	       int nsamps)
{
  int ii;
  double sum = 0;
  
  


  for(ii=0;ii<window;ii++){
    sum += inbuffer[ii];
    outbuffer[ii] = sum/(ii+1);
  }

  for (ii=window/2;ii<nsamps-window/2;ii++){
    sum += inbuffer[ii+window/2];
    sum -= inbuffer[ii-window/2];
    outbuffer[ii] = sum/window;
  }

  for(ii=nsamps-window;ii<nsamps;ii++){
    outbuffer[ii] = sum/(nsamps-ii);
    sum -= inbuffer[ii];
  }
}


void downsampleTim(float* inbuffer,
		   float* outbuffer,
		   int factor,
		   int newLen)
{
  int ii,jj;
#pragma omp parallel for default(shared) private(jj,ii)
  for(ii=0;ii<newLen;ii++){
    for(jj=0;jj<factor;jj++)
      outbuffer[ii]+=inbuffer[(ii*factor)+jj];
    //outbuffer[ii]/=(float) factor;
  }
}


void foldTim(float* buffer,
	     double* result,
	     int* counts, 
	     double tsamp,
	     double period,
	     double accel,
	     int nsamps,
	     int nbins,
	     int nints)
{
  int ii,phasebin,subbint,factor1;
  float factor2,tj;
  float c = 299792458.0;
  float tobs;
  
  tobs = nsamps*tsamp;
  factor1 = (int) ((nsamps/nints)+1);

  for(ii=0;ii<nsamps;ii++){
    tj = ii*tsamp;
    phasebin = abs(((int)(nbins*tj*(1+accel*(tj-tobs)/(2*c))/period + 0.5)))%nbins;
    subbint = (int) ii/factor1;
    result[(subbint*nbins)+phasebin]+=buffer[ii];
    counts[(subbint*nbins)+phasebin]++;
  }
}

void rfft(float* buffer,
          float* result,
          int size)
{
  fftwf_plan plan;
  int ii;
  plan = fftwf_plan_dft_r2c_1d(size, buffer, (fftwf_complex*) result,FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

void resample(float* input,
	      float* output,
	      int nsamps,
	      float accel,
	      float tsamp)
{
  int index,ii;
  int nsamps_by_2 = nsamps/2;
  float partial_calc = (accel*tsamp) / (2 * 299792458.0);
  float tot_drift = partial_calc * pow(nsamps_by_2,2);
  int last_bin = 0;
  for (ii=0;ii<nsamps;ii++){
    index = ii + partial_calc * pow(ii-nsamps_by_2,2) - tot_drift;
    output[index] = input[ii];
    if (index - last_bin > 1)
      output[index-1] = input[ii];
    last_bin = index;
  }
}


