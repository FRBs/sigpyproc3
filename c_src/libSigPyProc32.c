#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>
 
/*----------------------------------------------------------------------------*/

void getTim(float* inbuffer,float* outbuffer,int nchans,int nsamps,int index){
  int ii,jj,val;
#pragma omp parallel for default(shared) private(jj,ii) shared(outbuffer,inbuffer)
  for (ii=0;ii<nsamps;ii++){
    for (jj=0;jj<nchans;jj++){
      outbuffer[index+ii]+=inbuffer[(nchans*ii)+jj];
    }
  }
}

void getBpass(float* inbuffer,float* outbuffer,int nchans,int nsamps){
  int ii,jj;
#pragma omp parallel for default(shared) private(jj,ii) shared(outbuffer,inbuffer)
  for (jj=0;jj<nchans;jj++){
    for (ii=0;ii<nsamps;ii++){
      outbuffer[jj]+=inbuffer[(nchans*ii)+jj];
    }
  }
}

void dedisperse(float* inbuffer,
		float* outbuffer,
		int* delays,
		int maxdelay,
		int nchans,
		int nsamps,
		int index )
{
  int ii,jj;
#pragma omp parallel for default(shared) private(ii,jj) shared(outbuffer,inbuffer)
  for (ii=0;ii<(nsamps-maxdelay);ii++){
    for (jj=0;jj<nchans;jj++){
      outbuffer[index+ii] += inbuffer[(ii*nchans)+(delays[jj]*nchans)+jj];
    }
  }
}

void maskChannels(float* inbuffer,
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

void subband(float* inbuffer,
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

void foldFil(float* inbuffer,
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

void downsample(float* inbuffer,
                float* outbuffer,
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
#pragma omp parallel for default(shared) private(jj,ii,kk,ll,temp)
  for(ii=0;ii<newnsamps;ii++){
    for(jj=0;jj<newnchans;jj++){
      temp = 0;
      pos = nchans*ii*tfactor+jj*ffactor;
      for(kk=0;kk<tfactor;kk++){
        for(ll=0;ll<ffactor;ll++){
          temp += inbuffer[kk*nchans+ll+pos];
        }
      }
      outbuffer[ii*newnchans+jj] = temp/totfactor;
    }
  }  
}

void getChan(float* inbuffer,
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

void splitToChans(float* inbuffer,
                  float* outbuffer,
                  int nchans,
                  int nsamps,
		  int gulp)
{
  int ii,jj;
#pragma omp parallel for default(shared) private(ii,jj)
  for (ii=0;ii<nsamps;ii++){
    for (jj=0;jj<nchans;jj++)
      outbuffer[jj*gulp+ii] = inbuffer[(ii*nchans)+jj];
  }
}

void getStats(float* inbuffer,
              float* means,
              float* stdevs,
              float* maxbuffer,
              float* minbuffer,
              int nchans,
              int nsamps,
              int startflag)

{
  int ii,jj;
  float val;

  if( startflag == 0 ){
    for (jj=0;jj<nchans;jj++){
      maxbuffer[jj] = inbuffer[jj];
      minbuffer[jj] = inbuffer[jj];
    }
  }
#pragma omp parallel for default(shared) private(jj,ii) shared(inbuffer)
  for (jj=0;jj<nchans;jj++){
    for (ii=0;ii<nsamps;ii++){
      val = inbuffer[(nchans*ii)+jj];
      means[jj]+=val;
      stdevs[jj]+=pow(val,2);
      if( val > maxbuffer[jj])
        maxbuffer[jj]=val;
      else if( val < minbuffer[jj] )
        minbuffer[jj]=val;
    }
  }

}

void to8bit(float* inbuffer,
	    unsigned char* outbuffer,
	    unsigned char* flagbuffer,
	    float* factbuffer,
	    float* plusbuffer,
	    float* flagMax,
	    float* flagMin,
	    int nsamps,
	    int nchans)
{
  int ii,jj;
#pragma omp parallel for default(shared) private(jj,ii)
  for (ii=0;ii<nsamps;ii++){
    for (jj=0;jj<nchans;jj++){
      outbuffer[(ii*nchans)+jj] = inbuffer[(ii*nchans)+jj]/factbuffer[jj] - plusbuffer[jj]; 
      if (inbuffer[(ii*nchans)+jj] > flagMax[jj])
        flagbuffer[(ii*nchans)+jj] = 2;
      else if (inbuffer[(ii*nchans)+jj] < flagMin[jj])
        flagbuffer[(ii*nchans)+jj] = 0;
      else
        flagbuffer[(ii*nchans)+jj] = 1;
    }
  }
}

void invertFreq(float* inbuffer,float* outbuffer,int nchans,int nsamps)
{

  int ii,jj;
#pragma omp parallel for default(shared) private(jj,ii) shared(outbuffer,inbuffer)
  for (ii = 0; ii < nsamps; ii++){
    for (jj = 0; jj < nchans; jj++){
      outbuffer[(jj)+ii*nchans] = inbuffer[(nchans-1-jj)+ii*nchans];
    }
  }  

}



