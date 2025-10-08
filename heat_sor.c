/*
 * Copyright (c) 2025      High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 *
 * Authors: Kingshuk Haldar <kingshuk.haldar@hlrs.de>
 *
 */

#define _POSIX_C_SOURCE 199309L
#include"mpi.h"
#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<limits.h>

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define FREE_IF(x) if(NULL!= x) { free(x); x= NULL; }

void TheCtor() __attribute__ ((constructor));

inline static double Timespec2Seconds(const struct timespec *const ts)
{
  const double t= ((double) ts->tv_sec)+ ((double) ts->tv_nsec)* 1.0e-9;
  return t;
}
inline static double WtimeRaw()
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return Timespec2Seconds(&ts);
}

static struct timespec Timespec0;
static double Time0= -1.0;
void TheCtor()
{
  clock_gettime(CLOCK_REALTIME, &Timespec0);
  Time0= Timespec2Seconds(&Timespec0);
}
inline static double Wtime() { return WtimeRaw()- Time0; }

static char *Timespec2Pretty(struct timespec *ts)
{
  struct tm *tmobj= localtime(&(ts->tv_sec));
  static char buf[1024]; size_t buflen= sizeof(buf)/ sizeof(buf[0]);
  strftime(buf, buflen, "%d.%b.%Y %H:%M:%S", tmobj);
  sprintf(buf+ strlen(buf), ".%ld (%.9lf s)", ts->tv_nsec, Timespec2Seconds(ts)- Time0);
  return buf;
}
inline static char *DateTime0Pretty() { return Timespec2Pretty(&Timespec0); }
inline static char *DateTimePretty()
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return Timespec2Pretty(&ts);
}

MPI_Comm WComm= MPI_COMM_WORLD;
int WRank= -1, WSize= 0;

static double TempBC[4]= { -1.0, -1.0, -1.0, -1.0 }, TempIC= 0.0;
static double ProcMemMB= 0.0, XferMemMB= 0.0;
static int ProcDim[2]= { 0, 0 };
static int NumIters= 0, ShowInterval= 0;
int ReadControlFile0(const char *const fn)
{
  int fileokay= 0;
  FILE *fp= fopen(fn, "r");
  if(NULL== fp) { goto bye; }

  char line[128]= { '\0' };
  int len= (int) (sizeof(line)/ sizeof(line[0]));
  while(NULL!= fgets(line, len, fp)) {
    if(0== strlen(line)|| '#'== line[0]) { continue; }

    char key[16]= { '\0' };
    if(1!= sscanf(line, "%s", key)) { continue; }

    if(0== strcmp("procdim", key)) {
      sscanf(line, "%*s %d %d", ProcDim, ProcDim+ 1);
      continue;
    }

    if(0== strcmp("bc", key)) {
      sscanf(line, "%*s %lf %lf %lf %lf", TempBC, TempBC+ 1, TempBC+ 2, TempBC+ 3);
      continue;
    }

    if(0== strcmp("ic", key)) {
      sscanf(line, "%*s %lf", &TempIC);
      continue;
    }

    if(0== strcmp("procmem", key)) {
      sscanf(line, "%*s %lf", &ProcMemMB);
      continue;
    }

    if(0== strcmp("niters", key)) {
      sscanf(line, "%*s %d", &NumIters);
      continue;
    }

    if(0== strcmp("show", key)) {
      sscanf(line, "%*s %d", &ShowInterval);
      continue;
    }

    if(0== strcmp("xfermem", key)) {
      sscanf(line, "%*s %lf", &XferMemMB);
      continue;
    }
  }
  fclose(fp); fp= NULL;

  if(ProcDim[0]> 0&& ProcDim[1]> 0&& TempBC[0]> 0.0&& TempBC[1]> 0.0&& TempBC[2]> 0.0&& TempBC[3]> 0.0&& TempIC>= 0.0&& ProcMemMB> 0.0&&
     (ProcDim[0]* ProcDim[1]== WSize)) {
    printf(" Initial condition: %.2lf K\n", TempIC);
    printf("Boundary condition: { %.2lf, %.2lf, %.2lf, %.2lf } K\n", TempBC[0], TempBC[1], TempBC[2], TempBC[3]);
    printf(" Process dimension: %d x %d\n", ProcDim[0], ProcDim[1]);
    printf("  Req memory/ rank: %g MB\n", ProcMemMB);
    printf("     Xfermem/ rank: %g MB\n", XferMemMB);
    fileokay= 1;
  } else {
    printf("Could not find all information in control file (%s) or validate them.\n", fn);
    printf("ProcDim: %d x %d, TempBC: { %lf, %lf, %lf, %lf }, TempIC: %lf, ProcMemMB: %lf\n",
           ProcDim[0], ProcDim[1], TempBC[0], TempBC[1], TempBC[2], TempBC[3], TempIC, ProcMemMB);
    printf("Num-procs: %d\n", WSize);
    fflush(stdout);
  }

  if(0== NumIters) {
    NumIters= 100;
  }
  printf("        #Iteration: %d\n", NumIters);

  if(0== ShowInterval) {
    ShowInterval= MAX(1,NumIters/100);
  }
  printf("       Print every: %d iterations\n", ShowInterval);

 bye:
  return fileokay;
}
void ReadControlFile(const char *const fn)
{
  int fileokay= 0;
  if(0== WRank) {
    fileokay= ReadControlFile0(fn);
  }

  MPI_Bcast(&fileokay, 1, MPI_INT, 0, WComm);
  if(0== fileokay) { MPI_Abort(WComm, -1); }

  MPI_Bcast(ProcDim, 2, MPI_INT, 0, WComm);
  MPI_Bcast(TempBC, 4, MPI_DOUBLE, 0, WComm);
  MPI_Bcast(&TempIC, 1, MPI_DOUBLE, 0, WComm);
  MPI_Bcast(&ProcMemMB, 1, MPI_DOUBLE, 0, WComm);
  MPI_Bcast(&NumIters, 1, MPI_INT, 0, WComm);
  MPI_Bcast(&ShowInterval, 1, MPI_INT, 0, WComm);
  MPI_Bcast(&XferMemMB, 1, MPI_DOUBLE, 0, WComm);
}

static double *Temp= NULL;
static int NumRows= 0, NumCols= 0, NumIntElems= 0;
static int NumExtElems= 0;
static int (*Nbrs)[4]= NULL;
static double *SendBuf= NULL, *RecvBuf= NULL;
static int ProcNbrs[4];
static int ProcExtOffset[4];
static int NumXferElems= 0;
static int XferOffset[5];

int GetNumRowsOrCols(const size_t numbytes)
{
  const double nb= (double) numbytes;
  const double d= (double) sizeof(double);
  const double i= (double) sizeof(int);
  const double l= (double) sizeof(long long);

  const double dsc= sqrt(16.0* d* d+ 16.0* i* nb+ 4.0* d* nb);
  const double denom= 1.0/ (8.0* i+ 2.0* d);
  const double r1= (-4.0* d+ dsc)* denom;
  const double r2= (-4.0* d- dsc)* denom;

#if 0
  printf("roots= %.8e, %.8e\n", r1, r2);
#endif
  const int nrc= (int) MAX(r1,r2);
  return nrc;
}

void Alloc()
{
  size_t nbytes= (size_t) (ProcMemMB* 1024.0* 1024.0);
  const int nrc= GetNumRowsOrCols(nbytes);
  if(0== WRank) {
    printf(" Used memory/ rank: %.6g MB\n",
           ((double) (sizeof(double)* (nrc* nrc+ 2* (nrc+ nrc))+ sizeof(int[4])* nrc* nrc))/ (1024.0* 1024.0));
    printf("   #elements/ rank: %d^2= %d\n", nrc, nrc* nrc);
  }
  NumRows= NumCols= nrc;
  NumIntElems= NumRows* NumCols;
  NumExtElems= 2* (NumRows+ NumCols);

  Temp= (double *) malloc(sizeof(double)* (NumIntElems+ NumExtElems));
  Nbrs= (int (*)[4]) malloc(sizeof(int[4])* NumIntElems);

  nbytes= (size_t) (XferMemMB* 0.125* 1024.0* 1024.0);
  /* 0.125= 1/8= 1/ (4(nbrs) x 2(buf)) */

  NumXferElems= nbytes/ sizeof(double);
  if(NumXferElems< NumExtElems) {
    NumXferElems= NumExtElems;
  }

  SendBuf= (double *) malloc(sizeof(double)* NumXferElems* 8);
  RecvBuf= SendBuf+ NumXferElems* 4;
}
void Free()
{
  RecvBuf= NULL;
  FREE_IF(SendBuf);

  FREE_IF(Nbrs);

  FREE_IF(Temp);
}

void AssignProcNeighbours()
{
  ProcNbrs[3]= ProcNbrs[2]= ProcNbrs[1]= ProcNbrs[0]= -1;

  /* not on boundary-0 */
  if(0!= WRank/ ProcDim[1]) { ProcNbrs[0]= WRank- ProcDim[1]; }

  /* not on boundary-1 */
  if(0!= (WRank+ 1)% ProcDim[1]) { ProcNbrs[1]= WRank+ 1; }

  /* not on boundary-2 */
  if(WRank/ ProcDim[1]!= ProcDim[0]- 1) { ProcNbrs[2]= WRank+ ProcDim[1]; }

  /* not on bondary-3 */
  if(0!= WRank% ProcDim[1]) { ProcNbrs[3]= WRank- 1; }

#if defined(_DEBUG)
  /* printf("┘ ┐ ┌ └ ┼ ─ ├ ┤ ┴ ┬ │\n"); */
  for(int ip= 0; ip< WSize; ++ip) {
    if(WRank== ip) {
      printf("    ┌───┐\n");
      printf("    │%3d│\n", ProcNbrs[2]);
      printf("┌───┼───┼───┐\n");
      printf("│%3d│%3d│%3d│\n", ProcNbrs[3], ip, ProcNbrs[1]);
      printf("└───┼───┼───┘\n");
      printf("    │%3d│\n", ProcNbrs[0]);
      printf("    └───┘\n");
      fflush(stdout);
    }
    MPI_Barrier(WComm);
  }
#endif
}
#define IX(x,y) ((x)* NumCols+ (y))
void AssignNeighbourArrays()
{
  /* internal elements */
  long long offset= NumIntElems;

  /* boundary: no rank exists here, just boundary-condition */
  for(int i= 0; i< 4; ++i) {
    if(-1== ProcNbrs[i]) {
      ProcExtOffset[i]= offset;
      offset+= (0== i% 2? NumCols: NumRows);
    }
  }

  /* rank boundary: needs update */
  for(int i= 0; i< 4; ++i) {
    if(-1!= ProcNbrs[i]) {
      ProcExtOffset[i]= offset;
      offset+= (0== i% 2? NumCols: NumRows);
    }
  }

  /* offsets in send/recv buffers */

  /* init */
  for(int i= 0; i< 5; ++i) {
    XferOffset[i]= -1;
  }

  /* assign */
  offset= 0;
  for(int i= 0; i< 4; ++i) {
    if(-1!= ProcNbrs[i]) {
      XferOffset[i]= offset;
      offset+= NumXferElems;
    }
  }
  XferOffset[4]= offset;

#if defined(_DEBUGR0)
  if(0== WRank) {
    printf("0:  ProcNbrs: { %d %d %d %d }\n", ProcNbrs[0], ProcNbrs[1], ProcNbrs[2], ProcNbrs[3]);
    printf("0:XferOffset: { %d %d %d %d %d }\n", XferOffset[0], XferOffset[1], XferOffset[2], XferOffset[3], XferOffset[4]);
  }
#endif

  int ix;
  for(ix= 0; ix< NumIntElems; ++ix) {
    Nbrs[ix][3]= Nbrs[ix][2]= Nbrs[ix][1]= Nbrs[ix][0]= -1;
  }

  ix= 0;
  for(size_t ic= 0; ic< NumCols; ++ic) {
    Nbrs[ix][0]= ProcExtOffset[0]+ ic;
    ++ix;
  }

  ix= NumCols- 1;
  for(long long ir= 0; ir< NumRows; ++ir) {
    Nbrs[ix][1]= ProcExtOffset[1]+ ir;
    ix+= NumCols;
  }

  ix= NumCols* (NumRows- 1);
  for(long long ic= 0; ic< NumCols; ++ic) {
    Nbrs[ix][2]= ProcExtOffset[2]+ ic;
    ++ix;
  }

  ix= 0;
  for(long long ir= 0; ir< NumRows; ++ir) {
    Nbrs[ix][3]= ProcExtOffset[3]+ ir;
    ix+= NumCols;
  }

  for(ix= NumCols; ix< NumIntElems; ++ix) {
    Nbrs[ix][0]= ix- NumCols;
  }
  ix= 0;
  for(long long ir= 0; ir< NumRows; ++ir) {
    for(long long ic= 0; ic< NumCols- 1; ++ic) {
      Nbrs[ix][1]= ix+ 1;
      ++ix;
    }
    ++ix;
  }

  for(ix= 0; ix< NumIntElems- NumCols; ++ix) {
    Nbrs[ix][2]= ix+ NumCols;
  }
  ix= 1;
  for(long long ir= 0; ir< NumRows; ++ir) {
    for(long long ic= 1; ic< NumCols; ++ic) {
      Nbrs[ix][3]= ix- 1;
      ++ix;
    }
    ++ix;
  }

#if defined(_DEBUGR0)
  if(0== WRank) {
    printf("0:    ");
    for(long long ic= 0; ic< NumCols; ++ic) {
      printf("(%2d)", Nbrs[IX(NumRows-1,ic)][2]);
    }
    printf("    \n");

    for(long long ir= NumRows; ir> 0; --ir) {
      printf("0:(%2d)", Nbrs[IX(ir-1,0)][3]);
      for(long long ic= 0; ic< NumCols; ++ic) {
        printf("(%2lld)", IX(ir-1,ic));
      }
      printf("(%2d)\n", Nbrs[IX(ir-1,NumCols-1)][1]);
    }

    printf("0:    ");
    for(long long ic= 0; ic< NumCols; ++ic) {
      printf("(%2d)", Nbrs[IX(0,ic)][0]);
    }
    printf("    \n");
  }
#endif
}
#undef IX
void CreateNeighbourhood()
{
  AssignProcNeighbours();

  AssignNeighbourArrays();
}

void ApplyInitialConditions()
{
  for(long long ix= 0; ix< NumIntElems; ++ix) {
    Temp[ix]= TempIC;
  }
}

void ApplyBoundaryConditions()
{
  if(-1== ProcNbrs[0]) {
    for(long long ic= 0; ic< NumCols; ++ic) {
      Temp[ProcExtOffset[0]+ ic]= TempBC[0];
    }
  }
  if(-1== ProcNbrs[1]) {
    for(long long ir= 0; ir< NumRows; ++ir) {
      Temp[ProcExtOffset[1]+ ir]= TempBC[1];
    }
  }
  if(-1== ProcNbrs[2]) {
    for(long long ic= 0; ic< NumCols; ++ic) {
      Temp[ProcExtOffset[2]+ ic]= TempBC[2];
    }
  }
  if(-1== ProcNbrs[3]) {
    for(long long ir= 0; ir< NumRows; ++ir) {
      Temp[ProcExtOffset[3]+ ir]= TempBC[3];
    }
  }
}

void CopyCol(double *const dest, const double *const src)
{
  for(long long ir= 0; ir< NumRows; ++ir) {
    dest[ir]= src[ir* NumCols];
  }
}
void Buf2Xfer()
{
  if(-1!= ProcNbrs[0]) {
    memcpy(SendBuf+ XferOffset[0], Temp, sizeof(double)* NumCols);
  }
  if(-1!= ProcNbrs[1]) {
    CopyCol(SendBuf+ XferOffset[1], Temp+ NumCols- 1);
  }
  if(-1!= ProcNbrs[2]) {
    memcpy(SendBuf+ XferOffset[2], Temp+ NumCols* (NumRows- 1), sizeof(double)* NumCols);
  }
  if(-1!= ProcNbrs[3]) {
    CopyCol(SendBuf+ XferOffset[3], Temp);
  }
}
void Xfer2Buf()
{
  if(-1!= ProcNbrs[0]) {
    memcpy(Temp+ ProcExtOffset[0], RecvBuf+ XferOffset[0], sizeof(double)* NumCols);
  }
  if(-1!= ProcNbrs[1]) {
    memcpy(Temp+ ProcExtOffset[1], RecvBuf+ XferOffset[1], sizeof(double)* NumRows);
  }
  if(-1!= ProcNbrs[2]) {
    memcpy(Temp+ ProcExtOffset[2], RecvBuf+ XferOffset[2], sizeof(double)* NumCols);
  }
  if(-1!= ProcNbrs[3]) {
    memcpy(Temp+ ProcExtOffset[3], RecvBuf+ XferOffset[3], sizeof(double)* NumRows);
  }
}
void UpdateNeighbourhood()
{
  Buf2Xfer();
  MPI_Request reqs[8];
  MPI_Status stats[8];
  int nreqs= 0;

  for(int i= 0; i< 4; ++i) {
    if(-1== ProcNbrs[i]) { continue; }
    const int nbr= ProcNbrs[i];
    MPI_Irecv(RecvBuf+ XferOffset[i], NumXferElems, MPI_DOUBLE, nbr, nbr, WComm, reqs+ nreqs); ++nreqs;
    MPI_Isend(SendBuf+ XferOffset[i], NumXferElems, MPI_DOUBLE, nbr, WRank, WComm, reqs+ nreqs); ++nreqs;
  }
  if(nreqs> 0) {
    MPI_Waitall(nreqs, reqs, stats);
  }
  Xfer2Buf();
}

void Init()
{
  CreateNeighbourhood();

  ApplyInitialConditions();

  ApplyBoundaryConditions();

  UpdateNeighbourhood();
}
static int GetIterNumElems(const int it)
{
#if 1
  const int factor= 2;
  if(0== WRank&& 0== it% (factor* WSize)&& 6> it/ (factor* WSize)) {
    printf("New phase (#it= %d) started at: %s\n", it, DateTimePretty());
  }
  if(0== it/ (factor* WSize)) {
    const int x= 0== WRank? NumIntElems* 12/ 10: NumIntElems;
    return x;
  } else if(1== it/ (factor* WSize)) {
    const int x= WRank% 2== it% 2? NumIntElems: NumIntElems/ 2;
    return x;
  } else if(2== it/ (factor* WSize)) {
    const int x= WRank% 4== it% 4? NumIntElems: NumIntElems/ 2;
    return x;
  } else if(3== it/ (factor* WSize)) {
    const int x= WRank% 2== it% 2? NumIntElems/ 5: NumIntElems/ 10;
    return x;
  } else if(4== it/ (factor* WSize)) {
    const int x= NumIntElems- ((int) round(((double) (it- 4* factor* WSize))* 0.01* ((double) NumIntElems)));
    return MAX(NumIntElems/10,x);
  }
#endif

  return NumIntElems;
}
void Iterate()
{
  double res;
  double k[2]; k[0]= 1.0/ ((double) NumCols); k[1]= 1.0/ ((double) NumRows);
  const double w= 1.3;          /* relaxation parameter */
  for(int it= 0; it< NumIters; ++it) {
    res= 0.0;
    static int ix= 0;
    const int nelems= GetIterNumElems(it);
    for(int i= 0; i< nelems; ++i) {
      const double last= Temp[ix];
      Temp[ix]= (1- w)* Temp[ix]+ w* (Temp[Nbrs[ix][0]]+ Temp[Nbrs[ix][1]]+ Temp[Nbrs[ix][2]]+ Temp[Nbrs[ix][3]])* 0.25;
      res+= fabs(last- Temp[ix]);
      ix= (ix+ 1)% NumIntElems;
    }
    UpdateNeighbourhood();
    res/= ((double) nelems);
    MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, WComm);
    res/= ((double) WSize);

    if(0== WRank&& (0== it|| 0== (it+ 1)% ShowInterval|| it+ 1== NumIters)) {
      printf("Iteration-%3d: %.6e\n", it+ 1, res); fflush(stdout);
    }
    if(res< 1.0e-12) {
      if(0== WRank) {
        printf("Final residual: %.6e\n", res);
        fflush(stdout);
      }
      break;
    }
  }
}

int main(int argc, char *argv[])
{
  int ret= MPI_Init(&argc, &argv);
  if(MPI_SUCCESS!= ret) { goto bye; }

  MPI_Comm_rank(WComm, &WRank);
  MPI_Comm_size(WComm, &WSize);
  if(0== WRank) {
    static const char *built = __DATE__ " " __TIME__;
    printf("  Program built on: %s\n", built);
    printf("Program started at: %s\n", DateTime0Pretty());
  }

  ReadControlFile(argc> 1? argv[1]: "sor.in");

  Alloc();

  Init();

  Iterate();

  Free();

 bye:
  if(0== WRank) {
    printf("Finalize() at: %.2lf us\n", Wtime()* 1.0e6);
  }
  ret= MPI_Finalize();
  return ret;
}
