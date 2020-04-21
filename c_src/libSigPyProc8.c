#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>
 
/*----------------------------------------------------------------------------*/

unsigned char getRand(float mean, float std){
  unsigned char randval;
  randval = (std*normDist() + mean);
  if(randval>255)
    randval = 255;
  else if(randval<0)
    randval = 0;
  return randval;
}

void getTim(unsigned char* inbuffer,float* outbuffer,int nchans,int nsamps,int index){
  int ii,jj,val;
#pragma omp parallel for default(shared) private(jj,ii)
  for (ii=0;ii<nsamps;ii++){
    for (jj=0;jj<nchans;jj++){
      outbuffer[index+ii]+=inbuffer[(nchans*ii)+jj];
    }
  }
}

void getBpass(unsigned char* inbuffer,float* outbuffer,int nchans,int nsamps){
  int ii,jj;
#pragma omp parallel for default(shared) private(jj,ii) 
  for (jj=0;jj<nchans;jj++){
    for (ii=0;ii<nsamps;ii++){
      outbuffer[jj]+=inbuffer[(nchans*ii)+jj];
    }
  }
}

void dedisperse(unsigned char* inbuffer,
		float* outbuffer,
		int* delays,
		int maxdelay,
		int nchans,
		int nsamps,
		int index )
{
  int ii,jj,var1;
#pragma omp parallel for default(shared) private(ii,jj)
  for (ii=0;ii<(nsamps-maxdelay);ii++){
    for (jj=0;jj<nchans;jj++){
      outbuffer[index+ii] += inbuffer[(ii*nchans)+(delays[jj]*nchans)+jj];
    }
  }
}

void maskChannels(unsigned char* inbuffer,
		  unsigned char* mask,
		  int nchans,
		  int nsamps)
{
  int ii,jj;
  for (ii=0;ii<nchans;ii++){
    if (mask[ii] == 0){
      for (jj=0;jj<nsamps;jj++){
	inbuffer[jj*nchans+ii] = 0.0;
      }
    }
  }
}


void subband(unsigned char* inbuffer,
	     float* outbuffer,
	     int* delays,
	     int* c_to_s,
	     int maxdelay,
	     int nchans,
	     int nsubs,
	     int nsamps)
{
  int ii,jj,out_size;
  out_size = nsubs*(nsamps-maxdelay)*sizeof(float);
  memset(outbuffer,0.0,out_size);
#pragma omp parallel for default(shared) private(ii,jj)
  for (ii=0;ii<(nsamps-maxdelay);ii++){
    for (jj=0;jj<nchans;jj++){
      outbuffer[(ii*nsubs) + c_to_s[jj]] += (float) inbuffer[(ii*nchans) + (delays[jj]*nchans) + jj];
    }
  }
}


void foldFil(unsigned char* inbuffer,
	     float* foldbuffer,
	     int* countbuffer,
	     int* delays,
	     int maxDelay,
	     double tsamp,
	     double period,
	     double accel,
	     int totnsamps,
	     int nsamps,
	     int nchans,
	     int nbins,
	     int nints,
	     int nsubs,
	     int index)
{
  int ii,jj,phasebin,subband,subint,pos1,pos2;
  float factor1,factor2,val,tj;
  int tobs;
  float c = 299792458.0;
  factor1 = (float) totnsamps/nints;
  factor2 = (float) nchans/nsubs;
  tobs = (int) (totnsamps*tsamp); 
  for (ii=0;ii<(nsamps-maxDelay);ii++){
    tj = (ii+index)*tsamp;
    phasebin = ((int)(nbins*tj*(1+accel*(tj-tobs)/(2*c))/period + 0.5))%nbins;        
    subint = (int) ((index+ii)/factor1);
    pos1 = (subint*nsubs*nbins)+phasebin;
    for (jj=0;jj<nchans;jj++){
      subband = (int) (jj/factor2);
      pos2 = pos1+(subband*nbins);
      val = inbuffer[(ii*nchans)+(delays[jj]*nchans)+jj];
      foldbuffer[pos2] += val;
      countbuffer[pos2]++; 
    }
  }
}

void downsample(unsigned char* inbuffer,
		unsigned char* outbuffer,
		int tfactor,
		int ffactor,
		int nchans,
		int nsamps)
{
  float temp;
  int ii,jj,kk,ll,pos;
  int newnsamps = nsamps/tfactor;
  int newnchans = nchans/ffactor;
  int totfactor = ffactor*tfactor;
#pragma omp parallel for default(shared) private(jj,ii,kk,ll,temp,pos) 
  for(ii=0;ii<newnsamps;ii++){
    for(jj=0;jj<newnchans;jj++){
      temp = 0;
      pos = nchans*ii*tfactor+jj*ffactor;
      for(kk=0;kk<tfactor;kk++){
        for(ll=0;ll<ffactor;ll++){
          temp += inbuffer[kk*nchans+ll+pos];
        }
      }
      outbuffer[ii*newnchans+jj] = (unsigned char) (temp/totfactor+0.5);
    }
  }
}


void getChan(unsigned char* inbuffer,
	     float* outbuffer,
	     int chan,
	     int nchans,
	     int nsamps,
	     int index)
{
  int ii;
#pragma omp parallel for default(shared) private(ii)
  for(ii=0;ii<nsamps;ii++)
    outbuffer[index+ii] = inbuffer[(ii*nchans)+chan];
}

void splitToChans(unsigned char* inbuffer,
		  float* outbuffer,
		  int nchans,
		  int nsamps,
		  int gulp)
{
  int ii,jj;
#pragma omp parallel for default(shared) private(ii,jj)
  for (ii=0;ii<nsamps;ii++){
    for (jj=0;jj<nchans;jj++){
      outbuffer[(jj*gulp)+ii] = inbuffer[(ii*nchans)+jj];
    }  
  }
}

/*
TODO: This code is very slow compared to python slicing.
*/
void splitToBands(unsigned char* inbuffer,
      unsigned char* outbuffer,
      int nchans,
      int nsamps,
      int nsub,
      int chanpersub,
      int chanstart)
{
  int ii,jj;
  int chan_end = nsub*chanpersub+chanstart;
#pragma omp parallel for default(shared) private(ii,jj)
  for (ii=0; ii<nsamps; ii++){
    for (jj=chanstart; jj<chan_end; jj++){
      outbuffer[(ii*nchans)+(jj%chanpersub*nsub + jj/chanpersub)] = inbuffer[(ii*nchans)+jj];
    }  
  }
}

/*
getStats: Computing central moments in one pass through the data, 
the algorithm is numerically stable and accurate.

Ref:
https://www.johndcook.com/blog/skewness_kurtosis/
https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
*/
void getStats(unsigned char* inbuffer,
        float* M1,
        float* M2,
        float* M3,
        float* M4,
        float* maxbuffer,
        float* minbuffer,
        long long* count,
        int nchans,
        int nsamps,
        int startflag)
        
{
  int ii,jj;
  unsigned char val;
  
  if( startflag == 0 ){
    for (jj=0;jj<nchans;jj++){
      maxbuffer[jj] = inbuffer[jj];
      minbuffer[jj] = inbuffer[jj];
    }
  }
#pragma omp parallel for default(shared) private(jj,ii) shared(inbuffer)
  for (jj=0; jj<nchans; jj++){
    double delta, delta_n, delta_n2, term1;
    for (ii=0; ii<nsamps; ii++){
      val = inbuffer[(nchans*ii)+jj];
      count[jj] += 1;
      long long n = count[jj];

      delta    = val - M1[jj];
      delta_n  = delta / n;
      delta_n2 = delta_n * delta_n;
      term1    = delta * delta_n * (n - 1);
      M1[jj]  += delta_n;
      M4[jj]  += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2[jj] - 4 * delta_n * M3[jj];
      M3[jj]  += term1 * delta_n * (n - 2) - 3 * delta_n * M2[jj];
      M2[jj]  += term1;

      if( val > maxbuffer[jj])
        maxbuffer[jj] = val;
      else if( val < minbuffer[jj] )
        minbuffer[jj] = val;
    }
  }
  
}

void invertFreq(unsigned char* inbuffer,unsigned char* outbuffer,int nchans,int nsamps)
{

  int ii,jj;
#pragma omp parallel for default(shared) private(jj,ii) shared(outbuffer,inbuffer)
  for (ii = 0; ii < nsamps; ii++){
    for (jj = 0; jj < nchans; jj++){
      outbuffer[(jj)+ii*nchans] = inbuffer[(nchans-1-jj)+ii*nchans];
    }
  } 
  
}

/*
Digitizing code taken from SigProcDigitizer.C (dspsr)
*/
void removeBandpass(unsigned char* inbuffer,
        unsigned char* outbuffer,
        float* means,
        float* stdevs,
        int nchans,
        int nsamps)
{
  int ii,jj;
  unsigned char val;
  double scale;

  float DIGI_MEAN  = 127.5;
  float DIGI_SIGMA = 6;
  float DIGI_SCALE = DIGI_MEAN / DIGI_SIGMA;
  int DIGI_MIN = 0;
  int DIGI_MAX = 255;  

#pragma omp parallel for default(shared) private(jj,ii) shared(outbuffer,inbuffer)
  for (jj = 0; jj < nchans; jj++){
    for (ii = 0; ii < nsamps; ii++){
      val = inbuffer[(nchans*ii)+jj];

      if (stdevs[jj] == 0.0)
        scale = 1.0;
      else
        scale = 1.0 / stdevs[jj];

      // Normalize the data per channel to N(0,1)
      double normval = (val - means[jj]) * scale;
      // Shift the data for digitization
      int result = ((normval * DIGI_SCALE) + DIGI_MEAN + 0.5);
      // clip the result at the limits
      if (result < DIGI_MIN)
        result = DIGI_MIN;
      
      if (result > DIGI_MAX)
        result = DIGI_MAX;
      outbuffer[(nchans*ii)+jj] = (unsigned char) result;
    }
  } 
  
}