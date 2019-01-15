/*   al2co.c
* ===========================================================================
*
*                            PUBLIC DOMAIN NOTICE
*              		Department of Biochemistry  
*		University of Texas Southwestern Medical Center at Dallas
*
*  This software is freely available to the public for use.We have not placed
*  any restriction on its use or reproduction.
*
*  Although all reasonable efforts have been taken to ensure the accuracy
*  and reliability of the software and data, the University of Texas 
*  Southwestern Medical Center does not and cannot warrant the performance
*  or results that may be obtained by using this software or data. The 
*  University of Texas Southwestern Medical Center disclaims all warranties, 
*  express or implied, including warranties of performance, merchantability 
*  or fitness for any particular purpose.
*
*  Please cite the author in any work or product based on this material.
*
* ===========================================================================
*
* File Name:  al2co.c
*
* Author:  Jimin Pei, Nick Grishin

* Version Creation Date:  12/23/2000 
*
* File Description:
*       A program to calculate positional conservation in a sequence alignment
*
* Last Updated: 06/03/2001 (add -e option)

*  Last Updated: 08/13/2002 (MAXSEQNUM)

 version 1.2
 Last updated: 05/06/03
 
*
* --------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
/* #include <malloc.h> */
#include <stddef.h>


#define SQUARE(a) ((a)*(a))
#define NUM_METHOD 9
#define MAX_WINDOW 20
#define MAX_DELTASITE 20
#define MAXSTR   10001
#define INDI -100
#define MAXSEQNUM 10000

char *digit="0123456789";
void nrerror(char error_text[]);
char *cvector(long nl, long nh);
int *ivector(long nl, long nh);
double *dvector(long nl, long nh);
int **imatrix(long nrl, long nrh, long ncl, long nch);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
double ***d3tensor(long nrl,long nrh,long ncl,long nch,long ndl,long ndh);

int a3let2num(char *let);
int am2num_c(int c);
int am2num(int c);
int am2numBZX(int c);

static void *mymalloc(int size);
char *strsave(char *str);
char *strnsave(char *str, int l);
static char **incbuf(int n, char **was);
static int *incibuf(int n, int *was);

void err_readali(int err_num);
void readali(char *filename);
static void printali(char *argt, int chunk, int n, int len, char **name, char **seq, int *start,int *csv);
int **ali_char2int(char **aseq,int start_num, int start_seq);
int **read_alignment2int(char *filename,int start_num,int start_seq);

void counter(int b);
double effective_number(int **ali, int *marks, int n, int start, int end);
double effective_number_nogaps(int **ali, int *marks, int n, int start, int end);
double effective_number_nogaps_expos(int **ali, int *marks, int n, int start, int end, int pos);

void freq(int **ali,double **f,int *num_gaps,int *effindiarr,double gap_threshold);
double *overall_freq(int **ali, int startp, int endp, int *mark);
double *overall_freq_wgt(int **ali,int startp,int endp,int *mark,double *wgt);
double *h_weight(int **ali, int ip);
void h_freq(int **ali, double **f, double **hfr);
void entro_conv(double **f, int **ali, double *econv);
void ic_freq(int **ali, double **f, double **icf);
void variance_conv(double **f, int **ali, double **oaf, double *vconv);
void pairs_conv(double **f,int **ali,int **matrix1,int indx,double *pconv);



typedef struct _conv_info{
        double **fq, **hfq, **icfq;
        char *alifilename;
        int alignlen;
	int nali;
	int *ngap;
	int gapless50;
	double eff_num_seq;
	double *over_all_frq;
	int  *eff_indi_arr;
	double *avc,*csi;
        double ***conv;
            } conv_info;
char **aname, **aseq;
int nal, alilen, *astart, *alen;
int **alignment;
double **u_oaf,**h_oaf;
char *am="-WFYMLIVACGPTSNQDEHRKBZX*.wfymlivacgptsnqdehrkbzx";
char *am3[]={
"---",
"TRP",
"PHE",
"TYR",
"MET",
"LEU",
"ILE",
"VAL",
"ALA",
"CYS",
"GLY",
"PRO",
"THR",
"SER",
"ASN",
"GLN",
"ASP",
"GLU",
"HIS",
"ARG",
"LYS",
"ASX",
"GLX",
"UNK",
"***"
"...",
};

void nmlconv(double *conv, conv_info cvf, int wn,int mi);
int  **read_aa_imatrix(FILE *fmat);
int  **identity_imat(long n);
void argument(void);
void print_parameters(FILE *outfile,char *argi,char *argo,int nt,char *argt,int argb,char *args,int argm,int argf,int argc, int argw, char *argn,char *arga, char *arge, double argg, char *argp,char *argd);
char ARG_I[2048],ARG_O[2048],ARG_T[2048],ARG_P[2048],ARG_D[2048],ARG_S[2048],ARG_N[50],ARG_A[50],ARG_E[50];

int main(int argc, char *argv[])
{
	FILE *fout, *fpdb,*matrixfile,*fpdbout,*ft;
	conv_info convinfo;
	char fstr[200];
	int i,j,k,l,nt=0;
	double sumofconv/*,heteroatom*/;
	int *ptoc1;
	char *ptoc,tmpstr[200],*convstr;
	int **smatrix;
	double *consv;
	double **consvall;
	int markali[MAXSEQNUM];
	int ARG_F=2,ARG_C=0,ARG_W=1,ARG_M=0,ARG_B=60;
	double ARG_G=0.5;
	double max_index, mini_index;
	int *csv_index;
	int NAL;
	int **ALIGNMENT;
	double threshold;


	/*read input arguments */
        if(argc<=2) { argument(); exit(0);}
	for(i=1;i<argc;i++) {
	    if(strcmp(argv[i],"-i")==0) {strcpy(ARG_I,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-o")==0) {strcpy(ARG_O,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-t")==0) {strcpy(ARG_T,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-p")==0) {strcpy(ARG_P,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-d")==0) {strcpy(ARG_D,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-s")==0) {strcpy(ARG_S,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-n")==0) {strcpy(ARG_N,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-a")==0) {strcpy(ARG_A,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-e")==0) {strcpy(ARG_E,argv[i+1]);i++;continue;}
	    if(strcmp(argv[i],"-f")==0) {sscanf(argv[i+1],"%d",&ARG_F);i++;continue;}
	    if(strcmp(argv[i],"-b")==0) {sscanf(argv[i+1],"%d",&ARG_B);i++;continue;}
	    if(strcmp(argv[i],"-c")==0) {sscanf(argv[i+1],"%d",&ARG_C);i++;continue;}
	    if(strcmp(argv[i],"-w")==0) {sscanf(argv[i+1],"%d",&ARG_W);i++;continue;}
	    if(strcmp(argv[i],"-m")==0) {sscanf(argv[i+1],"%d",&ARG_M);i++;continue;}
	    if(strcmp(argv[i],"-g")==0) {sscanf(argv[i+1],"%lf",&ARG_G);i++;continue;}
				}
	
        if((ARG_F>2)||(ARG_F<0)){fprintf(stderr,"frequency calculation method(-f): \n0, unweighted; 1, Henikoff weight; 2, independent count\n");
                    exit(0);}
        if((ARG_C>2)||(ARG_C<0)){fprintf(stderr,"conservation calculation strategy(-c):\n0,entropy;1,variance;2,sumofpairs\n");
                    exit(0);}
        if((ARG_M>2)||(ARG_M<0)){fprintf(stderr,"matrix transform(-m):\n0, no transform;1,normalization;2,adjustment\n");
                    exit(0);}
	if((ARG_G>1.0)||(ARG_G<=0)){fprintf(stderr,"gap percentage(-g) to suppress calculation must be no more than 1 and more than 0 \n");
		    exit(0);}
	
	/* smatrix:  identity matrix if reading from input failed */
        if((matrixfile=fopen(ARG_S,"r"))==NULL){
                if(strlen(ARG_S)!=0)fprintf(stderr, "Warning: not readable  matrixfile: %s; default: identity matrix\n",ARG_S);
                smatrix = identity_imat(24);   }
        else {smatrix=read_aa_imatrix(matrixfile);
	      /*for(i=1;i<=20;i++){
		for(j=1;j<=20;j++){
			fprintf(stderr, "%d ",smatrix[i][j]);
			
				  }
		fprintf(stderr,"\n");
				}*/
			
	      fclose(matrixfile); }

	/* read alignment */
	ALIGNMENT = read_alignment2int(ARG_I,1,1);
	if( (strcmp(ARG_E, "T") == 0) || (strcmp(ARG_E, "t")==0) ) {
		alignment = read_alignment2int(ARG_I,1,1);
		alignment = alignment + 1;
		NAL = nal;
		nal = nal -1;
	} 
	else {
		alignment=read_alignment2int(ARG_I,1,1);
		NAL = nal;
	}

	if(nal>=MAXSEQNUM) {
	    fprintf(stderr, "Error: Number of sequences exceeds %d\n", MAXSEQNUM);
	    exit(0);
	}
	if(alignment==NULL){
		fprintf(stderr, "alignment file not readable\n");
			   }

	/* memory allocation for the elements in convinfo*/
	convinfo.alignlen=alilen;
	convinfo.nali=nal;
	convinfo.ngap=ivector(0,alilen);
	convinfo.eff_indi_arr=ivector(0,alilen+1);
	convinfo.alifilename=cvector(0,strlen(ARG_I));
	strcpy(convinfo.alifilename,ARG_I);
	convinfo.over_all_frq=dvector(0,20);
	convinfo.fq=dmatrix(0,20,0,alilen);
	convinfo.hfq=dmatrix(0,20,0,alilen);
	convinfo.icfq=dmatrix(0,20,0,alilen);
	convinfo.conv=d3tensor(0,MAX_WINDOW,0,NUM_METHOD,0,alilen);
	convinfo.avc=dvector(0,NUM_METHOD);
	convinfo.csi=dvector(0,NUM_METHOD);
	consv=dvector(0,alilen);
	consvall=dmatrix(0,NUM_METHOD,0,alilen);/* for all 9 methods */
	csv_index=ivector(0,alilen);
	for(i=1;i<=MAX_WINDOW;i++)
		for(j=1;j<=NUM_METHOD;j++)
			for(k=0;k<=alilen;k++)
				convinfo.conv[i][j][k]=INDI;

	/* get the frequencies */
	freq(alignment,convinfo.fq,convinfo.ngap,convinfo.eff_indi_arr,ARG_G);
	h_freq(alignment,convinfo.fq,convinfo.hfq);
	ic_freq(alignment,convinfo.fq,convinfo.icfq);

	convinfo.gapless50=0;
	threshold = 0.5*convinfo.nali;
	for(i=1;i<=alilen;i++){
		if(convinfo.ngap[i]<=threshold)
		convinfo.gapless50++;
			      }
	for(i=1;i<=nal;i++){markali[i]=1;}
	convinfo.over_all_frq=overall_freq(alignment,1,alilen,markali);


	/* get the conservation indeces for method[1..NUM_METHOD] */
	entro_conv(convinfo.fq,alignment,convinfo.conv[1][1]);	
	entro_conv(convinfo.hfq,alignment,convinfo.conv[1][2]);
	entro_conv(convinfo.icfq,alignment,convinfo.conv[1][3]);
	variance_conv(convinfo.fq,alignment,u_oaf,convinfo.conv[1][4]);
	variance_conv(convinfo.hfq,alignment,h_oaf,convinfo.conv[1][5]);
	variance_conv(convinfo.icfq,alignment,u_oaf,convinfo.conv[1][6]);
	pairs_conv(convinfo.fq,alignment,smatrix,ARG_M,convinfo.conv[1][7]);
	pairs_conv(convinfo.hfq,alignment,smatrix,ARG_M,convinfo.conv[1][8]);
	pairs_conv(convinfo.icfq,alignment,smatrix,ARG_M,convinfo.conv[1][9]);

	/* average over window size ARG_W */
	for(i=1;i<=9;i++){
		sumofconv=0;
		k=0;
		for(j=1;j<=alilen;j++){
		   if(convinfo.conv[1][i][j]==INDI)continue;
 		   sumofconv+=convinfo.conv[1][i][j];
		   k++;
		   			}
		convinfo.avc[i]=sumofconv/k;
			 }
	if(ARG_W>1){
        ptoc1=convinfo.eff_indi_arr;
        for(i=1;i<=NUM_METHOD;i++){
		j=ARG_W;
                if(j>*ptoc1){fprintf(stderr,"window size too big!\nARG_W:%d;   ptoc1:%d\n",j,*ptoc1);exit(0);}
                for(k=(j+1)/2;k<=convinfo.eff_indi_arr[0]-j/2;k++){
                        sumofconv=0;
                        for(l=k-(j-1)/2;l<=k+j/2;l++){
                                sumofconv+=convinfo.conv[1][i][*(ptoc1+l)];
                                                     }
                        convinfo.conv[j][i][*(ptoc1+k)]=sumofconv/j;
                                                                     }
		for(k=1;k<(j+1)/2;k++){
			sumofconv=0;
			for(l=1;l<=2*k-1;l++){
				sumofconv+=convinfo.conv[1][i][*(ptoc1+l)];
					     }
			convinfo.conv[j][i][*(ptoc1+k)]=convinfo.avc[i]+(sumofconv/(2*k-1)-convinfo.avc[i])*sqrt(1.0*(2*k-1)/ARG_W);
				      }
		for(k=0;k<=j/2-1;k++){
			sumofconv=0;
			for(l=1;l<=2*k+1;l++){
				sumofconv+=convinfo.conv[1][i][*(ptoc1+convinfo.eff_indi_arr[0]-l+1)];
					     }
			convinfo.conv[j][i][*(ptoc1+convinfo.eff_indi_arr[0]-k)]=convinfo.avc[i]+(sumofconv/(2*k+1)-convinfo.avc[i])*sqrt(1.0*(2*k+1)/ARG_W);
				   }
                                  }
		    }


	/* normalize the conservation indices */
	nmlconv(consv, convinfo, ARG_W, ARG_C*3+ARG_F+1);
	if((strcmp(ARG_N,"F")==0)||(strcmp(ARG_N,"f")==0)){
	    for(i=1;i<=convinfo.alignlen;i++){
		if(convinfo.conv[ARG_W][ARG_C*3+ARG_F+1][i]==INDI){
			consv[i]=0-convinfo.csi[ARG_C*3+ARG_F+1]+convinfo.avc[ARG_C*3+ARG_F+1];}
		else consv[i]=convinfo.conv[ARG_W][ARG_C*3+ARG_F+1][i];
					  }
							}

	/* find the largest and the smallest conservation indices */
	max_index=mini_index=consv[1];
	for(i=2;i<=alilen;i++) {
		if(consv[i]>max_index) max_index=consv[i];
		if(consv[i]<mini_index) mini_index=consv[i];
				}
	for(i=1;i<=alilen;i++) {
		csv_index[i]=(int)(9.99*(consv[i]-mini_index)/(max_index-mini_index));
				}

        /* print file with conservation indices mapped to the alignment */
        if((ft=fopen(ARG_T,"w"))==NULL){;}
        else{
                nt=1;
                fclose(ft);
                if(ARG_B){;}
                else{ARG_B=60;}
                /*printali(ARG_T, ARG_B, nal, alilen, aname, aseq, astart, csv_index);*/
		printali(ARG_T, ARG_B, NAL, alilen, aname, aseq, astart, csv_index);
            }

	/* change b-factor to conservation index */
if((fpdb=fopen(ARG_P,"r"))==NULL){
	if(strlen(ARG_P)!=0){fprintf(stderr,"please give the right pdb file\n");
				}
				 }
else{
        if((fpdbout=fopen(ARG_D,"w"))==NULL){
                if(strlen(ARG_D)!=0){fprintf(stderr,"output pdb file using default name\n");}
                fpdbout=stdout;}

	convstr=cvector(0,200);
	while(fgets(fstr,MAXSTR,fpdb)!=NULL){
		if(strncmp(fstr,"ATOM   ",7)==0)break;
					    }
	if(feof(fpdb)){fprintf(stderr,"end of the file reached: not a good pdb file");exit(0);}

	ptoc=fstr+23;
	j=atoi(ptoc);i=1;
	for(i=1;ALIGNMENT[1][i]==0;i++);
	ptoc=fstr+17;
	if(strncmp(ptoc,am3[ALIGNMENT[1][i]],3)!=0){
		fprintf(stderr,"The first sequence does not match the pdb file\n");
		exit(0);			}
	while(!feof(fpdb)){
		ptoc=fstr+66;
		strcpy(tmpstr, ptoc);
		*(fstr+61)='\0';
		/*if(strncmp(fstr,"HETATM ",7)==0){
			heteroatom=-1.0;
			sprintf(convstr,"%5.2f",heteroatom);
			strcat(fstr,convstr);
			strcat(fstr,tmpstr);
			fprintf(fpdbout,"%s",fstr);}
		else{*/
		if(consv[i]>-10){sprintf(convstr,"%5.2f",consv[i]);}
		else{consv[i]=-9.9;sprintf(convstr,"%5.2f",consv[i]);}
		strcat(fstr,convstr);
		strcat(fstr,tmpstr);
		fprintf(fpdbout,"%s",fstr);
		    /*}*/

		fgets(fstr,MAXSTR,fpdb);
		if(strncmp(fstr,"ATOM   ",7)!=0){
		   if(strncmp(fstr,"HETATM ",7)==0){;}
		   else{fprintf(fpdbout,"END\n");break;}
						}
		ptoc=fstr+23;
		if(j!=atoi(ptoc)) {
			/*fprintf(stderr,"i=%d  ptoc=%d\n", i, atoi(ptoc));*/
			i=i+atoi(ptoc)-j;j=atoi(ptoc);for(;ALIGNMENT[1][i]==0;i++);}
		ptoc=fstr+17;
		if(strncmp(ptoc,am3[ALIGNMENT[1][i]],3)!=0){
			if(strncmp(fstr,"HETATM ",7)==0){;}
			if(i>alilen) {fprintf(fpdbout,"END\n");break;}
			if(strncmp(ptoc,am3[ALIGNMENT[1][i]],3)!=0){
			   if(strncmp(fstr,"HETATM ",7)!=0){
				fprintf(stderr,"The first sequence does not match the pdb file\n");
				fprintf(stderr,"%s %s", am3[ALIGNMENT[1][i]], ptoc);
				exit(0);
							   }
								  }
							}
			   }	
	fclose(fpdb);fclose(fpdbout);
	}

        /* print conservation indices */
    if( (strcmp(ARG_A,"T")==0)||(strcmp(ARG_A,"t")==0) ) {
	/* print results from all the 9 methods */
	for(j=1;j<=NUM_METHOD;j++) {
           nmlconv(consvall[j], convinfo, ARG_W, j);
           if((strcmp(ARG_N,"F")==0)||(strcmp(ARG_N,"f")==0)){
            for(i=1;i<=convinfo.alignlen;i++){
                if(convinfo.conv[ARG_W][j][i]==INDI)
                        consvall[j][i]=0-convinfo.csi[j]+convinfo.avc[j];
                else consvall[j][i]=convinfo.conv[ARG_W][j][i];
            }                              
           }
	}

	if((fout=fopen(ARG_O,"w"))==NULL) fout = stdout;
        for(i=1;i<=convinfo.alignlen;i++){
	    fprintf(fout, "%-5d %c      ",i,am[alignment[1][i]]);
	    for(j=1;j<=NUM_METHOD;j++) {
		fprintf(fout, "%8.3f", consvall[j][i]);
	    }
	    if(convinfo.conv[ARG_W][1][i]==INDI){
		fprintf(fout,"     *\n");
	    } else fprintf(fout, "\n");
        }
        fprintf(fout,"* gap fraction no less than %5.2f; conservation set to M-S\n", ARG_G);
        fprintf(fout,"  M: mean;  S: standard deviation\n");
	print_parameters(fout,ARG_I,ARG_O,nt,ARG_T,ARG_B,ARG_S,ARG_M,ARG_F,ARG_C,ARG_W,ARG_N,ARG_A,ARG_E,ARG_G,ARG_P,ARG_D);
	fclose(fout);

    }
    else {
        if((fout=fopen(ARG_O,"w"))==NULL){
        for(i=1;i<=convinfo.alignlen;i++){
                if(convinfo.conv[ARG_W][ARG_C*3+ARG_F+1][i]==INDI){
                fprintf(stdout, "%-5d %c      %6.3f    *\n", i,am[alignment[1][i]], consv[i]);
                                                                  }
                else{
                fprintf(stdout, "%-5d %c      %6.3f\n", i,am[alignment[1][i]], consv[i]);
                    }
                                         }

        fprintf(stdout,"* gap fraction no less than %5.2f; conservation set to M-S\n", ARG_G);
	fprintf(stdout,"  M: mean;  S: standard deviation\n");
        print_parameters(stdout,ARG_I,ARG_O,nt,ARG_T,ARG_B,ARG_S,ARG_M,ARG_F,ARG_C,ARG_W,ARG_N,ARG_A,ARG_E,ARG_G,ARG_P,ARG_D);
                                         }
        else{
        for(i=1;i<=convinfo.alignlen;i++){
                if(convinfo.conv[ARG_W][ARG_C*3+ARG_F+1][i]==INDI){
                fprintf(fout, "%-5d %c      %6.3f *\n", i,am[alignment[1][i]], consv[i]);
                                                                  }
                else{
                fprintf(fout, "%-5d %c      %6.3f\n", i,am[alignment[1][i]], consv[i]);
                    }
                                         }
        fprintf(fout,"* gap fraction no less than %5.2f; conservation set to M-S\n", ARG_G);
	fprintf(fout,"  M: mean;  S: standard deviation\n");
        print_parameters(fout,ARG_I,ARG_O,nt,ARG_T,ARG_B,ARG_S,ARG_M,ARG_F,ARG_C,ARG_W,ARG_N,ARG_A,ARG_E,ARG_G,ARG_P,ARG_D);
        fclose(fout);
            }
	}
	/*print_parameters(stderr,ARG_I,ARG_O,nt,ARG_T,ARG_B,ARG_S,ARG_M,ARG_F,ARG_C,ARG_W,ARG_N,ARG_G,ARG_P,ARG_D);*/
	exit(0);

}			

void nmlconv(double *conv, conv_info cvf, int wn,int mi)
{
        double mean=0,vc=0;
        int i,cps=0;

        for(i=1;i<=cvf.alignlen;i++){
                if(cvf.conv[wn][mi][i]==INDI) continue;
                cps++;
                mean+=cvf.conv[wn][mi][i];
                              }
        if(cps<=1) {fprintf(stderr,"less than one position\n");exit(0);}
        mean=mean/cps;
	/*fprintf(stderr, "mean value is: %f\n",mean);*/
        cvf.avc[mi]=mean;
        for(i=1;i<=cvf.alignlen;i++){
                if(cvf.conv[wn][mi][i]==INDI) continue;
                vc+=(cvf.conv[wn][mi][i]-mean)*(cvf.conv[wn][mi][i]-mean);
                              }
        vc=sqrt(vc/(cps-1));
	/*fprintf(stderr, "sigma is: %f\n",vc);*/
        cvf.csi[mi]=vc;

        if(vc==0){fprintf(stderr,"variance is zero\n");exit(0);}
        for(i=1;i<=cvf.alignlen;i++){
                if(cvf.conv[wn][mi][i]==INDI) {conv[i]=-1.0;continue;}
                if(vc!=0)conv[i]=(cvf.conv[wn][mi][i]-mean)/vc;
                else conv[i]=0;
                              }
}

int  **read_aa_imatrix(FILE *fmat){

/* read matrix from file *fmat */

int i,ri,rj,c,flag,j,t;
int col[31],row[31];
char stri[11];
int **mat;

mat=imatrix(0,25,0,25);
for(i=0;i<=25;++i)for(j=0;j<=25;++j)mat[i][j]=0;

i=0;
ri=0;
rj=0;

flag=0;


while( (c=getc(fmat)) != EOF){

if(flag==0 && c=='#'){flag=-1;continue;}
else if(flag==-1 && c=='\n'){flag=0;continue;}
else if(flag==-1){continue;}
else if(flag==0 && c==' '){flag=1;continue;}
else if(flag==1 && c=='\n'){flag=0;continue;}
else if(flag==1 && c==' '){continue;}
else if(flag==1){
                ++i;
                if(i>=25){nrerror("matrix has more than 25 columns: FATAL");exit(0);}
                col[i]=am2numBZX(c);
                continue;
                }
else if(flag==0 && c!=' ' && c!='#'){
                ri=0;
                ++rj;
                if(rj>=25){nrerror("matrix has more than 25 rows: FATAL");exit(0);}
                row[rj]=am2numBZX(c);
                flag=2;
                continue;
                }
else if (flag==2 && c==' '){for(i=0;i<=10;++i){stri[i]=' ';}j=0;continue;}
else if (flag==2 && c=='\n'){flag=0;continue;}
else if (flag==2){flag=3;stri[++j]=c;if(j>10){nrerror("string too long:FATAL");exit(0);}continue;}
else if (flag==3 && c==' ' || flag==3 && c=='\n'){
                        j=0;
                        ++ri;
                        t=atoi(stri);
                        mat[row[rj]][col[ri]]=t;
                        if (c=='\n')flag=0;else flag=2;
                        continue;
                        }
else if (flag==3){stri[++j]=c;continue;}

}

return mat;
}

int    **identity_imat(long n){
/* allocates square integer identity matrix of length n+1: m[0.n][0.n] */

int i,j;
int **m;
m=imatrix(0,n,0,n);
for(i=0;i<=n;++i)for(j=0;j<=n;++j){if(i==j)m[i][j]=1;else m[i][j]=0;}
return m;
}

void argument()
{
fprintf(stderr,"      al2co   arguments:\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -i    Input alignment file [File in]\n");
fprintf(stderr,"        Format: ClustalW or simple alignment format\n");
fprintf(stderr,"        The title (first line) should begin with \"CLUSTAL W\", or\n");
fprintf(stderr,"        the title line should be deleted.\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -o    Output file with conservation index for each position in the\n");
fprintf(stderr,"        alignment [File out] Optional\n");
fprintf(stderr,"        Default = STDOUT\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -t    Output file with conservation index mapped to the alignment\n");
fprintf(stderr,"        [File out] Optional\n");
fprintf(stderr,"        Conservation indices are linearly rescaled to be from 0\n");
fprintf(stderr,"        to 9.99. C'=9.99*(C-MIN)/(MAX-MIN), where C and C' are the\n");
fprintf(stderr,"        the indices before and after rescaling respectively, MAX and\n");
fprintf(stderr,"        MIN are the highest index and lowest index before rescaling\n");
fprintf(stderr,"        respectively. The integer part of each rescaled index is\n");
fprintf(stderr,"        written out along with the sequence alignment.\n");
fprintf(stderr,"        Default = no output\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -b    Block size of the output alignment file with conservation\n");
fprintf(stderr,"        [Integer] Optional\n");
fprintf(stderr,"        Default = 60\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -s    Input file with the scoring matrix [File in] Optional\n");
fprintf(stderr,"        Format: NCBI\n");
fprintf(stderr,"        Notice: Scoring matrix is only used for sum-of-pairs measure\n");
fprintf(stderr,"        with option -c  2.\n");
fprintf(stderr,"        Default = identity matrix\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -m    Scoring matrix transformation [Integer] Optional\n");
fprintf(stderr,"        Options:\n");
fprintf(stderr,"        0=no transformation,\n");
fprintf(stderr,"        1=normalization S'(a,b)=S(a,b)/sqrt[S(a,a)*S(b,b)],\n");
fprintf(stderr,"        2=adjustment S\"(a,b)=2*S(a,b)-(S(a,a)+S(b,b))/2\n");
fprintf(stderr,"        Default = 0\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -f    Weighting scheme for amino acid frequency estimation [Integer] Optional\n");
fprintf(stderr,"        Options:\n");
fprintf(stderr,"        0=unweighted,\n");
fprintf(stderr,"        1=weighted by the modified method of Henikoff & Henikoff (2,3),\n");
fprintf(stderr,"        2=independent-count based (1)\n");
fprintf(stderr,"        Default = 2\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -c    Conservation calculation method [Integer] Optional\n");
fprintf(stderr,"        Options:\n");
fprintf(stderr,"        0=entropy-based    C(i)=sum_{a=1}^{20}f_a(i)*ln[f_a(i)], where f_a(i)\n");
fprintf(stderr,"          is the frequency of amino acid a at position i,\n");
fprintf(stderr,"        1=variance-based   C(i)=sqrt[sum_{a=1}^{20}(f_a(i)-f_a)^2], where f_a\n");
fprintf(stderr,"          is the overall frequency of amino acid a,\n");
fprintf(stderr,"        2=sum-of-pairs measure   C(i)=sum_{a=1}^{20}sum_{b=1}^{20}f_a(i)*f_b(i)*S_{ab},\n");
fprintf(stderr,"          where S_{ab} is the element of a scoring matrix for amino acids a and b\n");
fprintf(stderr,"        Default = 0\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -w    Window size used for averaging [Integer] Optional\n");
fprintf(stderr,"        Default = 1\n");
fprintf(stderr,"        Recommended value for motif analysis: 3\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -n    Normalization option [T/F] Optional\n");
fprintf(stderr,"        Subtract the mean from each conservation index and divide by the\n");
fprintf(stderr,"        standard deviation.\n");
fprintf(stderr,"        Default = T\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -a    All methods option [T/F] Optional\n");
fprintf(stderr,"        If set to true, the results of all 9 methods will be output.\n");
fprintf(stderr,"        1. unweighted entropy measure; 2. Henikoff entropy measure;\n");
fprintf(stderr,"        3. independent count entropy measure;\n");
fprintf(stderr,"        4. unweighted variance measure; 5. Henikoff variance measure;\n");
fprintf(stderr,"        6. independent count variance measure;\n");
fprintf(stderr,"        7. unweighted matrix-based sum-of-pairs measure;\n");
fprintf(stderr,"        8. Henikoff matrix-based sum-of-pairs measure;\n");
fprintf(stderr,"        9. independent count matrix-based sum-of-pairs measure;\n");
fprintf(stderr,"        Default = F\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -e    Excluding the first sequence from calculation [T/F] Optional\n");
fprintf(stderr,"        If set to true, the first sequence in the alignment will not\n");
fprintf(stderr,"	be included in the conservation calculation.\n");
fprintf(stderr,"        Default = F\n\n");
fprintf(stderr,"  -g    Gap fraction to suppress conservation calculation [Real] Optional\n");
fprintf(stderr,"        The value should be more than 0 and no more than 1. Conservation\n");
fprintf(stderr,"        indices are calculated only for positions with gap fraction less\n");
fprintf(stderr,"        than the specified value. Otherwise, conservation indices will\n");
fprintf(stderr,"        be set to M-S, where M is the mean conservation value and S is\n");
fprintf(stderr,"        the standard deviation.\n");
fprintf(stderr,"        Default = 0.5\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -p    Input pdb file [File in] Optional\n");
fprintf(stderr,"        The sequence in the pdb file should match exactly the first sequence of\n");
fprintf(stderr,"        the alignment.\n");
fprintf(stderr,"\n");
fprintf(stderr,"  -d    Output pdb file [File Out] Optional\n");
fprintf(stderr,"        The B-factors are replaced by the conservation indices.\n");
fprintf(stderr,"        Default = STDOUT\n");
fprintf(stderr,"\n");


}

void print_parameters(FILE *outfile,char *argi,char *argo,int nt,char *argt,int argb,char *args,int argm,int argf,int argc, int argw, char *argn,char *arga, char *arge, double argg, char *argp,char *argd)
{
	FILE *fp;
	char *mt[]={
		"no transformation",
		"normalization S'(a,b)=S(a,b)/sqrt[S(a,a)*S(b,b)]",
		"adjustment S\"(a,b)=2*S(a,b)-(S(a,a)+S(b,b))/2",
		   };
	char *ws[]={
		"unweighted",
		"weighted by the modified method of Henikoff & Henikoff",
		"independent-count based",
		   };
	char *cm[]={
		"entropy-based",
		"variance-based",
		"sum-of-pairs measure",
		   };

	fprintf(outfile, "\nal2co - The parameters are:\n\n");
	fprintf(outfile, "Input alignment file - %s\n", argi);

	if((fp=fopen(argo,"r"))==NULL){
		fprintf(outfile, "Output conservation - STDOUT\n");
					 }
	else{	fprintf(outfile, "Output conservation file - %s\n", argo); fclose(fp);} 

	if(nt==0){;}
	else if(nt==1) 	{   
		fprintf(outfile, "Output alignment file with index - %s;", argt);
		fprintf(outfile, " Block size - %d\n",argb); 
			}

	if((fp=fopen(args,"r"))==NULL){;}
	else {
		if(argc==2) {
			fprintf(outfile,"Input matrix file - %s\n",args);
			fprintf(outfile,"Matrix transformation -  %s\n",mt[argm]);
			    }
		else {
			fprintf(outfile,"Input matrix file - %s\n",args);
			if( (strcmp(arga,"T")==0) || (strcmp(arga, "t")==0) ) {;}
			else fprintf(outfile,"Warning - Matrix not used: matrix is for sum-of-pairs measure, -c 2\n"); }
		fclose(fp);
	      }

	if( (strcmp(arga,"T")==0) || (strcmp(arga, "t")==0) ) {
	     fprintf(outfile, "All 9 methods are used - true \n");
	     fprintf(outfile, "   1. unweighted entropy measure; 2. Henikoff entropy measure;\n");
	     fprintf(outfile, "   3. independent count entropy measure;\n");
	     fprintf(outfile, "   4. unweighted variance measure; 5. Henikoff variance measure;\n");
	     fprintf(outfile, "   6. independent count variance measure;\n");
	     fprintf(outfile, "   7. unweighted matrix-based sum-of-pairs measure;\n");
	     fprintf(outfile, "   8. Henikoff matrix-based sum-of-pairs measure;\n");
	     fprintf(outfile, "   9. independent count matrix-based sum-of-pairs measure;\n");
   	}
	else {
	  fprintf(outfile, "Weighting scheme - %s\n", ws[argf]);
	  fprintf(outfile, "Conservation calculation method - %s\n",cm[argc]);
	}

	if( (strcmp(arge, "T")==0) || (strcmp(arge, "t")==0 ) ) {
		fprintf(outfile, "The first sequence is not used in calculation\n");
	}

	fprintf(outfile, "Window size - %d\n", argw);
	 
	if((strcmp(argn,"f")==0)||(strcmp(argn,"F")==0)){
		fprintf(outfile, "Conservation not normalized\n");}
	else fprintf(outfile, "Conservation normalized to zero mean and unity variance\n");

	fprintf(outfile, "Gap fraction to suppress calculation - %5.2f\n",argg);
	if((fp=fopen(argp,"r"))==NULL){;}
	else{
		fprintf(outfile,"Input pdb file - %s\n", argp);
		fclose(fp);
		if((fp=fopen(argd,"r"))==NULL){
			fprintf(outfile, "Output pdb file - STDOUT\n");
						}
		else {	fprintf(outfile, "Output pdb file - %s\n", argd);
			fclose(fp);
			}
		}
	
}
#define NR_END 1
#define FREE_ARG char*


void nrerror(char error_text[]){
fprintf(stderr,"%s\n",error_text);
fprintf(stderr,"FATAL - execution terminated\n");
exit(1);
}


char *cvector(long nl, long nh){
char *v;
v=(char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
if (!v) nrerror("allocation failure in ivector()");
return v-nl+NR_END;
}


int *ivector(long nl, long nh){
int *v;
v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
if (!v) nrerror("allocation failure in ivector()");
return v-nl+NR_END;
}

long *lvector(long nl, long nh){
long int *v;
v=(long int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long int)));
if (!v) nrerror("allocation failure in lvector()");
return v-nl+NR_END;
}

double *dvector(long nl, long nh){
double *v;
v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
if (!v) nrerror("allocation failure in dvector()");
return v-nl+NR_END;
}


int **imatrix(long nrl, long nrh, long ncl, long nch){
long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
int **m;
m=(int **)malloc((size_t)((nrow+NR_END)*sizeof(int*)));
if (!m) nrerror("allocation failure 1 in imatrix()");
m += NR_END;
m -= nrl;

m[nrl]=(int *)malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
if (!m[nrl]) nrerror("allocation failure 2 in imatrix()");
m[nrl] += NR_END;
m[nrl] -= ncl;

for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

return m;

}


double **dmatrix(long nrl, long nrh, long ncl, long nch){
long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
double **m;
m=(double **)malloc((size_t)((nrow+NR_END)*sizeof(double*)));
if (!m) nrerror("allocation failure 1 in dmatrix()");
m += NR_END;
m -= nrl;

m[nrl]=(double *)calloc((size_t)((nrow*ncol+NR_END)),sizeof(double));
if (!m[nrl]) nrerror("allocation failure 2 in dmatrix()");
m[nrl] += NR_END;
m[nrl] -= ncl;

for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

return m;

}



double ***d3tensor(long nrl,long nrh,long ncl,long nch,long ndl,long ndh){
long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
double ***t;

t=(double ***) malloc((size_t)((nrow+NR_END)*sizeof(double**)));
if(!t)nrerror("allocation failure 1 in d3tensor()");
t += NR_END;
t -= nrl;

t[nrl]=(double **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double*)));
if(!t[nrl])nrerror("allocation failure 2 in d3tensor()");
t[nrl] += NR_END;
t[nrl] -= ncl;

t[nrl][ncl]=(double *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(double)));
if(!t[nrl][ncl])nrerror("allocation failure 3 in d3tensor()");
t[nrl][ncl] += NR_END;
t[nrl][ncl] -= ndl;

for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
for(i=nrl+1;i<=nrh;i++){
	t[i]=t[i-1]+ncol;
	t[i][ncl]=t[i-1][ncl]+ncol*ndep;
	for(j=ncl+1;j<=nch;j++)t[i][j]=t[i][j-1]+ndep;
	}
return t;
}

int am2num_c(c)
{
switch (c) {
           	 case 'W':
                	c=1; break;
		 case 'w':
			c=26; break;
           	 case 'F': 
                	c=2; break;
                 case 'f':
			c=27; break;
           	 case 'Y': 
                	c=3; break;
		 case 'y':
                        c=28; break;
           	 case 'M': 
                	c=4; break;
		 case 'm':
                        c=29; break;
           	 case 'L': 
                	c=5; break;
		 case 'l':
                        c=30; break;
           	 case 'I': 
          		c=6; break;
		 case 'i':
                        c=31; break;
           	 case 'V': 
           		c=7; break;
		 case 'v':
                        c=32; break;
          	 case 'A': 
			c=8; break;
		 case 'a':
                        c=33; break; 
           	 case 'C': 
                	c=9; break;
		 case 'c':
                        c=34; break;
		 case 'G': 
			c=10; break;
		 case 'g':
                        c=35; break;
           	 case 'P': 
             	 	c=11; break;
		 case 'p':
                        c=36; break;
       		 case 'T': 
			c=12; break;
		 case 't':
                        c=37; break;
	         case 'S': 
			c=13; break;
		 case 's':
                        c=38; break;
           	 case 'N': 
                	c=14; break;
		 case 'n':
                        c=39; break;
           	 case 'Q': 
                	c=15; break;
		 case 'q':
                        c=40; break;
           	 case 'D': 
                	c=16; break;
		 case 'd':
                        c=41; break;
           	 case 'E': 
                	c=17; break;
		 case 'e':
                        c=42; break;
           	 case 'H': 
                	c=18; break;
		 case 'h':
                        c=43; break;
           	 case 'R': 
                	c=19; break;
		 case 'r':
                        c=44; break;
           	 case 'K': 
                	c=20; break;
		 case 'k':
                        c=45; break;
           	 default : 
			c=0; 
		}
return (c);
}


int am2num(c)
{
switch (c) {
           	 case 'W': case 'w':
                	c=1; break;
           	 case 'F': case 'f':
                	c=2; break;
           	 case 'Y': case 'y':
                	c=3; break;
           	 case 'M': case 'm':
                	c=4; break;
           	 case 'L': case 'l':
                	c=5; break;
           	 case 'I': case 'i':
          		c=6; break;
           	 case 'V': case 'v':
           		c=7; break;
          	 case 'A': case 'a': 
			c=8; break;
           	 case 'C': case 'c':
                	c=9; break;
		 case 'G': case 'g':
			c=10; break;
           	 case 'P': case 'p':
             	 	c=11; break;
       		 case 'T': case 't':
			c=12; break;
	         case 'S': case 's':
			c=13; break;
           	 case 'N': case 'n':
                	c=14; break;
           	 case 'Q': case 'q':
                	c=15; break;
           	 case 'D': case 'd':
                	c=16; break;
           	 case 'E': case 'e':
                	c=17; break;
           	 case 'H': case 'h':
                	c=18; break;
           	 case 'R': case 'r':
                	c=19; break;
           	 case 'K': case 'k':
                	c=20; break;
           	 default : 
			c=0; 
		}
return (c);
}

int am2numBZX(c)
{
switch (c) {
                 case 'W': case 'w':
                        c=1; break;
                 case 'F': case 'f':
                        c=2; break;
                 case 'Y': case 'y':
                        c=3; break;
                 case 'M': case 'm':
                        c=4; break;
                 case 'L': case 'l':
                        c=5; break;
                 case 'I': case 'i':
                        c=6; break;
                 case 'V': case 'v':
                        c=7; break;
                 case 'A': case 'a':
                        c=8; break;
                 case 'C': case 'c':
                        c=9; break;
                 case 'G': case 'g':
                        c=10; break;
                 case 'P': case 'p':
                        c=11; break;
                 case 'T': case 't':
                        c=12; break;
                 case 'S': case 's':
                        c=13; break;
                 case 'N': case 'n':
                        c=14; break;
                 case 'Q': case 'q':
                        c=15; break;
                 case 'D': case 'd':
                        c=16; break;
                 case 'E': case 'e':
                        c=17; break;
                 case 'H': case 'h':
                        c=18; break;
                 case 'R': case 'r':
                        c=19; break;
                 case 'K': case 'k':
                        c=20; break;
                 case 'B': case 'b':
                        c=21; break;
                 case 'Z': case 'z':
                        c=22; break;
                 case 'X': case 'x':
                        c=23; break;
                 case '*':
                        c=24; break;
                 default :
                        c=0;
                }
return (c);
}


static char str[MAXSTR+1];

/*char **aname, **aseq;
int nal, alilen, *astart, *alen;
int **alignment; 
*/



static void *mymalloc(int size);
char *strsave(char *str);
char *strnsave(char *str, int l);
static char **incbuf(int n, char **was);
static int *incibuf(int n, int *was);

void readali(char *filename);
int **ali_char2int(char **aseq,int start_num, int start_seq);
int **read_alignment2int(char *filename,int start_num,int start_seq);

void counter(int b);
double effective_number(int **ali, int *marks, int n, int start, int end);
double effective_number_nogaps(int **ali, int *marks, int n, int start, int end);
double effective_number_nogaps_expos(int **ali, int *marks, int n, int start, int end, int pos);



static void *mymalloc(size)
int size;
{
	void *buf;

	if ((buf = malloc(size)) == NULL) {
		fprintf(stderr, "Not enough memory: %d\n", size);
		exit(1);
	}
	return buf;
}

char *strsave(str)
char *str;
{
	char *buf;
	int l;

	l = strlen(str);
	buf = mymalloc(l + 1);
	strcpy(buf, str);
	return buf;
}

char *strnsave(str, l)
char *str;
int l;
{
	char *buf;

	buf = mymalloc(l + 1);
	memcpy(buf, str, l);
	buf[l] = '\0';
	return buf;
}

static char **incbuf(n, was)
int n;
char **was;
{
	char **buf;

	buf = mymalloc((n+1) * sizeof(buf[0]));
	if (n > 0) {
		memcpy(buf, was, n * sizeof(was[0]));
		free(was);
	}
	buf[n] = NULL;
	return buf;
}

static int *incibuf(n, was)
int n, *was;
{
	int *ibuf;

	ibuf = mymalloc((n+1) * sizeof(ibuf[0]));
	if (n > 0) {
		memcpy(ibuf, was, n * sizeof(was[0]));
		free(was);
	}
	ibuf[n] = 0;
	return ibuf;
}
void err_readali(int err_num)
{
	fprintf(stderr,"Error with reading alignment: %d\n",err_num);
}

void readali(filename)
char *filename;
{
	FILE *fp;
	char *s, *ss, *seqbuf;
	int n, l, len, len0;
	int ii,mark=1;

	if ((fp = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "No such file: \"%s\"\n", filename);
		err_readali(1);
		;exit(1);
	}
	alilen = 0;
	nal = 0;
	n = 0;
	if(fgets(str, MAXSTR, fp) != NULL) {
		if(strncmp(str,"CLUSTAL W",9)!=0){rewind(fp);}
					   }
	while (fgets(str, MAXSTR, fp) != NULL) {
		for (ss = str; isspace(*ss); ss++) ;
		if ((mark==0)&&(ii<=ss-str)) {continue;}
		if (*ss == '\0') {
			if (n == 0) {
				continue;
			}
			if (nal == 0) {
				if (n == 0) {
					fprintf(stderr, "No alignments read\n");
					err_readali(2);
					exit(1);
				}
				nal = n;
			} else if (n != nal) {
				fprintf(stderr, "Wrong nal, was: %d, now: %d\n", nal, n);
				err_readali(3); exit(1);
			}
			n = 0;
			continue;
		}
		for (s = ss; *s != '\0' && !isspace(*s); s++) ;
		*s++ = '\0';
		if (nal == 0) {
			astart = incibuf(n, astart);
			alen = incibuf(n, alen);
			aseq = incbuf(n, aseq);
			aname = incbuf(n, aname);
			aname[n] = strsave(ss);
		} else {
			if (n < 0 || n >= nal) {
				fprintf(stderr, "Bad sequence number: %d of %d\n", n, nal);
				err_readali(4);  exit(1);
			}
			if (strcmp(ss, aname[n]) != 0) {
				fprintf(stderr, "Names do not match");
				fprintf(stderr, ", was: %s, now: %s\n", aname[n], ss);
				err_readali(5);  exit(1);
			}
		}
		for (ss = s; isspace(*ss); ss++);
		if(mark==1){
		ii = ss-str;
		mark=0;}
		
		for (s = ss; isdigit(*s); s++) ;
		if (isspace(*s)) {
			if (nal == 0) {
				astart[n] = atoi(ss);
			}
			for (ss = s; isspace(*ss); ss++);
		}
		for (s = ss, l = 0; *s != '\0' && !isspace(*s); s++) {
			if (isalpha(*s)) {
				l++;
			}
		}
		len = s - ss;
		if (n == 0) {
			len0 = len;
			alilen += len;
		} else if (len != len0) {
			fprintf(stderr, "wrong len for %s", aname[n]);
			fprintf(stderr, ", was: %d, now: %d\n", len0, len);
			err_readali(6); exit(1);
		}
		alen[n] += l;
		if (aseq[n] == NULL) {
			aseq[n] = strnsave(ss, len);
		} else {
			seqbuf = mymalloc(alilen+1);
			memcpy(seqbuf, aseq[n], alilen-len);
			free(aseq[n]);
			aseq[n] = seqbuf;
			memcpy(seqbuf+alilen-len, ss, len);
			seqbuf[alilen] = '\0';
		}
		n++;
	}
	if (nal == 0) {
		if (n == 0) {
			fprintf(stderr, "No alignments read\n");
			err_readali(7);  exit(1);
		}
		nal = n;
	} else if (n != 0 && n != nal) {
		fprintf(stderr, "Wrong nal, was: %d, now: %d\n", nal, n);
		err_readali(8);  exit(1);
	}
	fclose(fp);
}

static void printali(char *argt, int chunk, int n, int len, char **name, char **seq, int *start,int *csv)
{
        int i, j, k, jj, mlen, sta, *pos;
	char csv_str[100], arg_t[2048];
        char *sq;
	int *isq;
	FILE *fpp;

	strcpy(arg_t,argt);
	fpp=fopen(arg_t,"w");
	strcpy(csv_str, "CSV:");
        pos = mymalloc(n * sizeof(pos[0]));
        memcpy(pos, start, n * sizeof(pos[0]));
        for (i=0; i < n && start[i] == 0; i++) ;
        sta = (i < n);
        for (i=1, mlen=strlen(name[0]); i < n; i++) {
                if (mlen < (int)strlen(name[i])) {
                        mlen = strlen(name[i]);
                }
        }
        jj = 0;
        do {
                if (jj > 0) {
                        fprintf(fpp, "\n");
                }
                for (i=0; i < n; i++) {
			if(i==0) {
				fprintf(fpp,"\n\n"); 
				for(k=0;k<mlen+4;k++){
					if(k<(int)strlen(csv_str)){
					fprintf(fpp,"%c",csv_str[k]);}
					else fprintf(fpp," ");
						   }
				if (sta) {
					fprintf(fpp,"%4d ", pos[i]);
				}
				isq = csv + jj;
				for (j=0; j+jj < len && j < chunk; j++) {
					if (isdigit(isq[j+1])) {
						pos[i]++;
					}
					fprintf(fpp,"%d", isq[j+1]);
				}
				if (sta) {
					fprintf(fpp, " %4d", pos[i]-1);
				}
				fprintf(fpp, "\n");
				  }

			for(k=0;k<mlen+4;k++){
			if(k<(int)strlen(name[i])){
			fprintf(fpp,"%c",name[i][k]);}
			else fprintf(fpp," ");
					   }			
                        if (sta) {
                                fprintf(fpp, "%4d ", pos[i]);
                        }
                        sq = seq[i] + jj;
                        for (j=0; j+jj < len && j < chunk; j++) {
                                if (isalpha(sq[j])) {
                                        pos[i]++;
                                }
                                fprintf(fpp, "%c", sq[j]);
                        }
                        if (sta) {
                                fprintf(fpp, " %4d", pos[i]-1);
                        }
                        fprintf(fpp, "\n");
                }
                jj += j;
        } while (jj < len);
        free(pos);
	fclose(fpp);
}

int **ali_char2int(char **aseq, int start_num, int start_seq){
/* fills the alignment ali[start_num..start_num+nal-1][start_seq..start_seq+alilen-1]
convetring charater to integer from aseq[0..nal-1][0..alilen-1]
*/

int i,j,end_num,end_seq;
int **ali;
end_num=start_num+nal-1;
end_seq=start_seq+alilen-1;
ali=imatrix(start_num,end_num,start_seq,end_seq);
for(i=start_num;i<=end_num;++i)for(j=start_seq;j<=end_seq;++j)ali[i][j]=am2num(aseq[i-start_num][j-start_seq]);
return ali;
}

int **read_alignment2int(char *filename,int start_num,int start_seq){
int **ali;
readali(filename);
ali=ali_char2int(aseq,start_num,start_seq);
return ali;
}



double effective_number(int **ali, int *marks, int n, int start, int end){

/* from the alignment of n sequences ali[1..n][1..l]
calculates effective number of sequences that are marked by 1 in mark[1..n]
for the segment of positions ali[][start..end]
Neff=ln(1-0.05*N-of-different-letters-per-site)/ln(0.95)
*/

int i,k,a,flag,ngc;
int *amco,lettercount=0,sitecount=0,gsitecount=0;
int *nogaps;
double letpersite=0,neff,randomcount=0.0;
amco=ivector(0,20);
nogaps=ivector(0,n);
for(i=0;i<=n;++i)nogaps[i]=0;
for(k=start;k<=end;++k){
	for(a=0;a<=20;++a)amco[a]=0;
	for(i=1;i<=n;++i)if(marks[i]==1)amco[ali[i][k]]++;
	flag=0;for(a=1;a<=20;++a)if(amco[a]>0){flag=1;lettercount++;}
	if(flag==1)sitecount++;
		       }
letpersite=1.0*lettercount/sitecount;
for(k=start;k<=end;++k){
	ngc=0;for(i=1;i<=n;++i)if(marks[i]==1)if(ali[i][k]!=0)ngc++;
	if(ngc!=0)gsitecount++;
	nogaps[ngc]++;
		       }
for(i=0;i<=n;++i)fprintf(stderr,"%3d %4d\n",i,nogaps[i]);

if(gsitecount!=sitecount){fprintf(stderr,"Counts didn't match in \"effective_number\": FATAL\n");exit(0);}

for(i=1;i<=n;++i)randomcount+=nogaps[i]*20.0*(1.0-exp(-0.05129329438755*i));
randomcount=randomcount/gsitecount;
letpersite=letpersite*20.0*(1.0-exp(-0.05129329438755*n))/randomcount;



neff=-log(1.0-0.05*letpersite)/0.05129329438755;
return neff;
}



double effective_number_nogaps(int **ali, int *marks, int n, int start, int end){

/* from the alignment of n sequences ali[1..n][1..l]
calculates effective number of sequences that are marked by 1 in mark[1..n]
for the segment of positions ali[][start..end]
Neff=ln(1-0.05*N-of-different-letters-per-site)/ln(0.95)
*/

int i,k,a,flag;
int *amco,lettercount=0,sitecount=0;
double letpersite=0,neff;
amco=ivector(0,20);
for(k=start;k<=end;++k){
	flag=0;for(i=1;i<=n;++i)if(marks[i]==1 && ali[i][k]==0)flag=1;
	if(flag==1)continue;
	for(a=0;a<=20;++a)amco[a]=0;
	for(i=1;i<=n;++i)if(marks[i]==1)amco[ali[i][k]]++;
	flag=0;for(a=1;a<=20;++a)if(amco[a]>0){flag=1;lettercount++;}
	if(flag==1)sitecount++;
		       }
if(sitecount==0)letpersite=0;
else letpersite=1.0*lettercount/sitecount;


neff=-log(1.0-0.05*letpersite)/0.05129329438755;
return neff;
}

int *letters;

void freq(int **ali,double **f,int *num_gaps,int *effindiarr,double gap_threshold)
{
	int i,j,k,effnumind;
	int count[21];
	
	letters = ivector(0, alilen+1);
	letters[0]=1;

	/* find the number of frequences at each position */
	effnumind=0;
	for(j=1;j<=alilen;j++){
		for(i=0;i<=20;i++) count[i]=0;
		for(i=1;i<=nal;i++) {
			if(ali[i][j]<=20) {count[ali[i][j]]++;}
			else {if(ali[i][j]>25&&ali[i][j]<=45)
				{count[ali[i][j]-25]++;}
			      else {
				fprintf(stderr,"not good number for AA\n");
				exit(0);
				   }
			     }
				    }
		num_gaps[j] = count[0];
	   	f[0][j] = count[0]*1.0/nal;
		if(f[0][j]>1) {
			fprintf(stderr,"gap number>total number\n");
			exit(0);
			      }
		if(f[0][j]>=gap_threshold) {   /* ignore the case where gaps occur >= gap_threshold(percentage of gaps)  */
			f[0][j]=INDI;
			continue;
				}
		effnumind++;
		effindiarr[effnumind]=j;
		count[0]=nal-count[0];
		letters[j] = count[0];
		if(count[0]<=0){
			fprintf(stderr, "count[0] less than 0: %d\n",count[0]);
			exit(0);
			       }
		for(k=1;k<=20;k++){
			f[k][j]=count[k]*1.0/count[0];
				  }
	}
	effindiarr[effnumind+1]=INDI;/*set the last element negative*/
	effindiarr[0]=effnumind;
}			

/* calculating the standard error of frequencies */
double **sigmaf(double **f)
{
	double **sigma;
	int i,j;

	sigma = dmatrix(0,20,0,alilen);
	if(letters[0]!=1) {
		fprintf(stderr, "letters not defined previously\n");
		exit(0);
			  }
	for(j=1;j<=alilen;j++) {
		if(letters[j]==0) {
			fprintf(stderr,"The column contains no letters:%d\n",j);
			for(i=1;i<=20;i++){sigma[i][j]=0;}
			continue;
				  }
		for(i=1;i<=20;i++) {
			if(f[i][j]==0) {sigma[i][j]=0;continue;}
			sigma[i][j]=sqrt(f[i][j]*(1-f[i][j])/letters[j]);
				   }
				}
	return sigma;		
}

int totalw;
double *oaf_ip, *h_oaf_ip; /*unweighted,henikoff frequency at position ip */
double *overall_freq(int **ali, int startp, int endp, int *mark);
double *overall_freq_wgt(int **ali,int startp,int endp,int *mark,double *wgt);
double *h_weight(int **ali, int ip)
{
	double *hwt;
	int count[21];
        int amtypes;
	int i,j;
	int maxstart, maxs, miniend,minie;
	int *mark;
	int gapcount,totalcount,wsign;

	hwt = dvector(0, nal);
	mark = ivector(0, nal);
	oaf_ip = dvector(0,20);
	h_oaf_ip = dvector(0,20);
	for(i=1;i<=20;i++) h_oaf_ip[i] = 0;

	/* mark the sequences with gaps */
	for(i=1;i<=nal;i++) {
	        /* NEW: fix a typo 05/06/03 */
		if((ali[i][ip]>0&&ali[i][ip]<=20)||(ali[i][ip]>25&&ali[i][ip]<=45)) mark[i]=1;
		else if (ali[i][ip]==0)mark[i]=0;
			    }
	
	/* find the maxstart and miniend positions */
	maxstart = 1;
	for(i=1;i<=nal;i++){
		if(mark[i]==0) continue;
		maxs = 1;
 	    	for(j=1;j<=alilen;j++) {
			if(ali[i][j]==0) maxs++;
			if(ali[i][j]>0) break;
					}
		if(maxstart<maxs) maxstart = maxs;
			   }
	miniend = alilen;
	for(i=1;i<=nal;i++){
		if(!mark[i]) continue;
		minie = alilen;
		for(j=alilen;j>0;j--) {
			if(ali[i][j]==0) minie--;
			if(ali[i][j]>0) break;
				      }
		if(miniend>minie) miniend = minie;
 			   }
	/* NEW: 05/06/03 */
	if(maxstart > miniend) maxstart = miniend;

/* special case where only one site is not gap */
        j=0;
        for(i=1;i<=nal;i++) {if(mark[i]>0) j++;}
        if(j==1) {
                for(i=1;i<=nal;i++) hwt[i]=mark[i];
                h_oaf_ip = overall_freq_wgt(ali,maxstart,miniend,mark,hwt);
                oaf_ip = overall_freq(ali,1,alilen,mark);
                return hwt;
                 }

	for(i=1;i<=nal;i++) hwt[i]=0;
	for(j=maxstart;j<=miniend;j++){
		
		amtypes = 0;
		for(i=0;i<=20;i++) count[i]=0;
		for(i=1;i<=nal;i++){
			if(mark[i]==0) continue;
			if(ali[i][j]>=0&&ali[i][j]<=20) count[ali[i][j]]++;
			else if(ali[i][j]>25&&ali[i][j]<=45) count[ali[i][j]-25]++;
				   }
		for(i=0;i<=20;i++) {
			if(count[i]>0) amtypes++;
				   }
		if(amtypes==1) continue; /* identical positions are excluded */
		gapcount=totalcount=0;
		for(i=1;i<=nal;i++){
			if(mark[i]==0) continue;
			if(ali[i][j]==0) gapcount++;
			totalcount++;
				   }
		if(gapcount>totalcount*0.5) continue;/*gap>50% excluded*/

		for(i=1;i<=nal;i++){
			if(mark[i]==0) continue;
			if(ali[i][j]>=0&&ali[i][j]<=20) {
				if(count[ali[i][j]]>0)  {
				    hwt[i]+=1.0/(count[ali[i][j]]*amtypes);
							}
						       }
			if(ali[i][j]>25&&ali[i][j]<=45) {
				if(count[ali[i][j]-25]>0)  {
				    hwt[i]+=1.0/(count[ali[i][j]-25]*amtypes);
							   }
							}
				   }
				}
	/* test if all henikoff weights are zero for all sequences */
	wsign=0;
	for(i=1;i<=nal;i++) {
		if(hwt[i]>0) { wsign=1;break;}
			    }
	gapcount=0;
	if(wsign==0) {
		for(i=1;i<=nal;i++){
			if(ali[i][ip]==0) gapcount++;
				   }
		if(gapcount==nal){fprintf(stderr, "This position contains only gaps\n");exit(0);}
		for(i=1;i<=nal;i++){
			if(ali[i][ip]!=0) hwt[i]=1.0/(nal-gapcount);
				   }
		     }

	h_oaf_ip = overall_freq_wgt(ali,maxstart,miniend,mark,hwt);
	oaf_ip = overall_freq(ali,1,alilen,mark);
	return hwt;
}
	
double **u_oaf,**h_oaf; /* unweighted and henikoff overall frequency */
void h_freq(int **ali, double **f, double **hfr)
{
	int i,j;	
	double *hwt;
	double sumofweight;

	u_oaf = dmatrix(0,20,0,alilen);
	h_oaf = dmatrix(0,20,0,alilen);
	
	for(j=1;j<=alilen;j++) {
		if(f[0][j]==INDI) {hfr[0][j]=INDI;continue;}
		hwt = h_weight(ali, j); /* assign position specific weight */
		for(i=1;i<=20;i++) {
			u_oaf[i][j]=oaf_ip[i];
			h_oaf[i][j]=h_oaf_ip[i];
				   }

 		sumofweight = 0;
		for(i=1;i<=nal;i++) {
			if(ali[i][j]>0) sumofweight+=hwt[i];
				    }
		for(i=1;i<=nal;i++) {
			if(ali[i][j]>0&&ali[i][j]<=20) {
				hfr[ali[i][j]][j]+=hwt[i]/sumofweight;
							}
			else if(ali[i][j]>25&&ali[i][j]<=45) {
				hfr[ali[i][j]-25][j]+=hwt[i]/sumofweight;
							     }
	   			    }
				}
}


void entro_conv(double **f, int **ali, double *econv)
{
	int i,j;

	for (j=1;j<=alilen;j++){
		econv[j]=0;
		if(f[0][j]==INDI) {econv[j]=INDI;continue;}
		for(i=1;i<=20;i++) {
			if(f[i][j]==0) continue;
			econv[j]+=f[i][j]*log(f[i][j]);
				   }
				}
}
		
double *overall_freq(int **ali, int startp, int endp, int *mark)
{
	double *oaf;
	int gapcount,totalcount;
	int total=0;
	int i,j;

	oaf = dvector(0,20);
	for(i=1;i<=20;i++) oaf[i]=0;
	if(startp>endp) {
		fprintf(stderr, "start position larger than ending position\n");
		exit(1);
			}
	for(j=startp;j<=endp;j++) {
		/* excluding those that have >50% gaps*/
		gapcount=totalcount=0;
		for(i=1;i<=nal;i++){
			if(mark[i]==0) continue;
			totalcount++;
			if(ali[i][j]==0) gapcount++;
				   }
		if(gapcount>totalcount*0.5) continue;

		for(i=1;i<=nal;i++) {
			if(mark[i]==0) continue;
			if(ali[i][j]>0&&ali[i][j]<=20) {
				oaf[ali[i][j]]+=1;
				total++;
							}
			if(ali[i][j]>25&&ali[i][j]<=45) {
				oaf[ali[i][j]-25]+=1;
				total++;
							}
				  }
				}
	for(i=1;i<=20;i++) {
		oaf[i]=oaf[i]/total;
			   }		
	return oaf;
}

double *overall_freq_wgt(int **ali,int startp,int endp,int *mark,double *wgt)
{
        double *oaf;
        double total=0;
	int totalcount,gapcount;
        int i,j;

        oaf = dvector(0,20);
        for(i=1;i<=20;i++) oaf[i]=0;
        if(startp>endp) {
                fprintf(stderr, "start position larger than ending position\n");                exit(1);
                        }
        for(j=startp;j<=endp;j++) {
		/* excluding those that have >50% gaps*/
		gapcount=totalcount=0;
		for(i=1;i<=nal;i++){
			if(mark[i]==0) continue;
			totalcount++;
			if(ali[i][j]==0) gapcount++;
				   }
		if(gapcount>totalcount*0.5) continue;
                for(i=1;i<=nal;i++) {
                        if(mark[i]==0) continue;
                        if(ali[i][j]>0&&ali[i][j]<=20) {
                                oaf[ali[i][j]]+=wgt[i];
                                total+=wgt[i];
                                                        }
                        if(ali[i][j]>25&&ali[i][j]<=45) {
                                oaf[ali[i][j]-25]+=wgt[i];
                                total+=wgt[i];
                                                        }
                                  }
	if(total==0) fprintf(stderr,"total=0\n");
                                }
        for(i=1;i<=20;i++) {
                oaf[i]=oaf[i]/total;
                           }
        return oaf;
}

void ic_freq(int **ali, double **f, double **icf)
{
        int i,j,k;
        int ele;
        double  *effnu;
        int *mark;

        mark = ivector(0,nal+10);
        effnu = dvector(0,20);
        for(i=0;i<=20;i++)
                for(j=1;j<=alilen;j++)
                        icf[i][j]=0;
        for(i=0;i<=nal;i++) mark[i]=0;

        for(j=1;j<=alilen;j++){
                if(f[0][j]==INDI) {icf[0][j]=INDI;continue;}
                for(k=0;k<=20;++k)effnu[k]=0;
                for(k=1;k<=20;++k){
                        for(i=1;i<=nal;++i){
                                mark[i]=0;
                                ele=ali[i][j];
                                if(ele==k)mark[i]=1;
                                ele=ali[i][j]-25;
                                if(ele==k) mark[i]=1;
                                         }
                        effnu[k]=effective_number_nogaps(ali,mark,nal,1,alilen);
                        effnu[0]+=effnu[k];
                                  }
                if(effnu[0]==0){fprintf(stderr,"all counts are zeros at the column %d: FATAL\n",j);exit(0);}
                for(k=1;k<=20;k++) effnu[k]=effnu[k]/effnu[0];
                for(k=1;k<=20;k++) icf[k][j] = effnu[k];
                               }
}

void variance_conv(double **f, int **ali, double **oaf, double *vconv)
{
	int i,j;

	for(i=1;i<=alilen;i++) vconv[i]=0;
	for(j=1;j<=alilen;j++) {
		if(f[0][j]==INDI) {vconv[j]=INDI;continue;}
		for(i=1;i<=20;i++) {
			vconv[j]+=(f[i][j]-oaf[i][j])*(f[i][j]-oaf[i][j]);
				   }
		vconv[j]=sqrt(vconv[j]);
			       }
}

void pairs_conv(double **f,int **ali,int **matrix1,int indx,double *pconv)
{
	int i,j,k;
	double **matrix2, **matrix3;

	matrix2 = dmatrix(0,25,0,25);
	matrix3 = dmatrix(0,25,0,25);
	for(i=0;i<=25;i++){
		for(j=0;j<=25;j++) {
			matrix2[i][j]=0;
			matrix3[i][j]=0;
				   }
			  }

	/* get the matrices */
	for(i=1;i<=24;i++){
		if(matrix1[i][i]==0) {
			fprintf(stderr, "diagonal elements zero: %d\n",i);
			exit(0);
				    }
		for(j=1;j<=24;j++)  {
			matrix2[i][j]=matrix1[i][j]*1.0/sqrt(matrix1[i][i]*matrix1[j][j]);
			matrix3[i][j]=matrix1[i][j]*2-0.5*(matrix1[i][i]+matrix1[j][j]);
				    }
				      }
	
	for(j=1;j<=alilen;j++) {
		if(f[0][j]==INDI){pconv[j]=INDI; continue;}
		pconv[j]=0;
		for(i=1;i<=20;i++) {
			for(k=1;k<=20;k++) {
			   if(indx==0){
				pconv[j]+=(f[k][j]*f[i][j])*matrix1[i][k];
				continue;}
			   if(indx==1){
				pconv[j]+=(f[k][j]*f[i][j])*matrix2[i][k];
				continue;}
			   if(indx==2){
				pconv[j]+=(f[k][j]*f[i][j])*matrix3[i][k];
			        continue;} 
			   fprintf(stderr,"not good index number of matrix\n");
			   exit(0);
					   }
				   }
				}
}


		
