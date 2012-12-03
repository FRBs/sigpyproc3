#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <complex.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>
 
/*----------------------------------------------------------------------------*/

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }
float median(float arr[], int n)
{
  int low, high;
  int median;
  int middle, ll, hh;

  low = 0;
  high = n - 1;
  median = (low + high) / 2;
  for (;;) {
    if (high <= low)          /* One element only */
      return arr[median];

    if (high == low + 1) {    /* Two elements only */
      if (arr[low] > arr[high])
        ELEM_SWAP(arr[low], arr[high]);
      return arr[median];
    }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])
      ELEM_SWAP(arr[middle], arr[high]);
    if (arr[low] > arr[high])
      ELEM_SWAP(arr[low], arr[high]);
    if (arr[middle] > arr[low])
      ELEM_SWAP(arr[middle], arr[low]);

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low + 1]);

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
      do
        ll++;
      while (arr[low] > arr[ll]);
      do
        hh--;
      while (arr[hh] > arr[low]);

      if (hh < ll)
        break;

      ELEM_SWAP(arr[ll], arr[hh]);
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]);

    /* Re-set active partition */
    if (hh <= median)
      low = ll;
    if (hh >= median)
      high = hh - 1;
  }
}
#undef ELEM_SWAP


void ccfft(float* buffer,
           float* result,
           int size)
{
  fftwf_plan plan;
  fftw_complex* tempinput;

  plan = fftwf_plan_dft_1d(size, (fftwf_complex*) buffer,
			   (fftwf_complex*) result, FFTW_BACKWARD,
			   FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

void ifft(float* buffer,
	  float* result,
          int size)
{
  fftwf_plan plan;
  plan = fftwf_plan_dft_c2r_1d(size, (fftwf_complex*) buffer,
			       result,FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

void formSpecInterpolated(float* fftbuffer,
	      float* specbuffer,
	      int nsamps)
{
  int ii;
  float i,r,a,b;
  float rl=0.0, il=0.0;
  for(ii=0;ii<nsamps;ii++){
    r = fftbuffer[2*ii];
    i = fftbuffer[2*ii+1];
    a = pow(r,2)+pow(i,2);
    b = (pow((r-rl),2) + pow((i-il),2))/2.;
    specbuffer[ii] = sqrt(fmax(a,b));
    rl=r;
    il=i;
  }
}

void formSpec(float* fftbuffer,
	      float* specbuffer,
	      int points)
{
  int ii;
  float i,r;
  for(ii=0;ii<points;ii+=2){
    r = fftbuffer[ii];
    i = fftbuffer[ii+1];
    specbuffer[ii/2] = sqrt(pow(r,2)+pow(i,2));
  }
}

void rednoise(float* fftbuffer,
	      float* outbuffer,
	      float* oldinbuf,
	      float* newinbuf,
	      float* realbuffer,
	      int nsamps,
	      float tsamp,
	      int startwidth,
	      int endwidth,
	      float endfreq)
{
  int ii;
  int binnum = 1;
  int bufflen=startwidth;
  int rindex,windex;
  int numread_new,numread_old;
  float slope,mean_new,mean_old;
  float T = nsamps*tsamp; 
  
  //Set DC bin to 1.0
  outbuffer[0] = 1.0;
  outbuffer[1] = 0.0;
  windex=2;
  rindex=2;

  //transfer bufflen complex samples to oldinbuf
  for(ii=0;ii<2*bufflen;ii++)
    oldinbuf[ii] = fftbuffer[ii+rindex];
  numread_old = bufflen;
  rindex+=2*bufflen;

  //calculate powers for oldinbuf
  for (ii=0;ii<numread_old;ii++){
    realbuffer[ii] = 0;
    realbuffer[ii] =
      oldinbuf[ii*2] * oldinbuf[ii*2] +
      oldinbuf[ii*2+1] * oldinbuf[ii*2+1];
  }
  
  //calculate first median of our data and determine next bufflen  
  mean_old = median(realbuffer, numread_old) / log(2.0);
  binnum += numread_old;
  bufflen = startwidth*log(binnum);

  while(rindex/2<nsamps){
    if(bufflen>nsamps-rindex/2)
      numread_new = nsamps-rindex/2;
    else
      numread_new = bufflen;
    
    for(ii=0;ii<2*numread_new;ii++)
      newinbuf[ii] = fftbuffer[ii+rindex];
    rindex += 2*numread_new;

    for (ii = 0; ii < numread_new; ii++) {
      realbuffer[ii] = 0;
      realbuffer[ii] =
	newinbuf[ii*2] * newinbuf[ii*2] +
	newinbuf[ii*2+1] * newinbuf[ii*2+1];
    }
    
    mean_new = median(realbuffer, numread_new) / log(2.0);
    slope = (mean_new - mean_old) / (numread_old + numread_new);
    
    for (ii = 0; ii < numread_old; ii++) {
      outbuffer[ii*2+windex] = 0.0;
      outbuffer[ii*2+1+windex] = 0.0;
      outbuffer[ii*2+windex]=oldinbuf[ii*2]/
	sqrt(mean_old+slope*((numread_old+numread_new) / 2.0 - ii));
      outbuffer[ii*2+1+windex]=oldinbuf[ii*2+1]/
	sqrt(mean_old+slope*((numread_old+numread_new) / 2.0 - ii));
    }
    windex+=2*numread_old;
    
    binnum += numread_new;
    if ((binnum * 1.0) / T < endfreq)
      bufflen = startwidth * log(binnum);
    else
      bufflen = endwidth;
    numread_old = numread_new;
    mean_old = mean_new;
    
    for (ii = 0; ii < 2*numread_new; ii++) {
      oldinbuf[ii] = 0;
      oldinbuf[ii] = newinbuf[ii];
    }
  }
  for (ii = 0; ii < 2*numread_old; ii++) {
    outbuffer[ii+windex] = oldinbuf[ii] / sqrt(mean_old);
  }
}

void conjugate(float* specbuffer,
	       float* outbuffer,
	       int size)
{
  int ii;
  int out_size = 2*size-2;
  memcpy(outbuffer,specbuffer,size*sizeof(float));
  for (ii=0;ii<size-2;ii+=2){
    outbuffer[out_size-1-ii] = -1.0*specbuffer[ii+1];
    outbuffer[out_size-2-ii] = specbuffer[ii];
  }
}

void sumHarms(float* specbuffer,
	      float* sumbuffer,
	      int* sumarray,
	      int* factarray,
	      int nharms,
	      int nsamps,
	      int nfoldi)
{
  int ii,jj,kk;
  for(ii=nfoldi;ii<nsamps-(nharms-1);ii+=nharms){
    for(jj=0;jj<nharms;jj++){
      for(kk=0;kk<nharms/2;kk++){
	sumbuffer[ii+jj] += specbuffer[ factarray[kk] + sumarray[jj*nharms/2+kk]] ;
      }
    }
    for(kk=0;kk<nharms/2;kk++){
      factarray[kk]+=2*kk+1;
    }
  }										 
}

void multiply_fs(float* selfbuffer,
		 float* otherbuffer,
		 float* outbuffer,
		 int size)
{
  int ii;
  float sr,si,or,oi;
  for (ii=0;ii<size;ii+=2){
    sr = selfbuffer[ii];
    si = selfbuffer[ii+1];
    or = otherbuffer[ii];
    oi = otherbuffer[ii+1];
    outbuffer[ii]   = sr*or - si*oi;
    outbuffer[ii+1] = sr*oi + si*or; 
  }
}
