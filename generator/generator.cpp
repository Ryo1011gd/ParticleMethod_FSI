#include <cstdio>
#include <cstring>
#include "log.h"
#include "typedefs.h"

using namespace std;

static double ParticleDistance = -1.0;
static vec_t UpperDomain;
static vec_t LowerDomain;
//static double Ratio;

static char boidname[256] = "sample.boid";
static char gridname[256] = "sample.grid";

//static int NofCub=0;
//static int NofCub2=0;
//static int NofCyb=0;
//static int NofCyb2=0;


struct Cuboid_t{
    double space;
    iType_t type;
    iType_t rigidtype;
    vec_t lower;
    vec_t upper;
    vec_t velocity;
    double enthalpy;
};


struct Cuboid2_t{
    double space;
    iType_t type;
    vec_t  lower;
    vec_t  upper;
    vec_t  velocity;
    double enthalpy;
};
struct Cyboid_t{
    double space;
    iType_t type;
    iType_t rigidtype;
    vec_t  lower;
    vec_t  upper;
    vec_t  velocity;
    double enthalpy;
    double ratio;
};

struct Cyboid2_t{
    double space;
    iType_t type;
    vec_t  lower;
    vec_t  upper;
    vec_t  velocity;
    double enthalpy;
    double ratio;
};
struct Recboid_t{
    double space;
    iType_t type;
    vec_t  lower;
    vec_t  upper;
    vec_t  velocity;
    double enthalpy;
    double angle;
};
struct Recboid2_t{
    double space;
    iType_t type;
    vec_t  lower;
    vec_t  upper;
    vec_t  velocity;
    double enthalpy;
    double angle;
};

typedef index_t<Cuboid_t>  iCub_t;
typedef index_t<Cuboid2_t> iCub2_t;
typedef index_t<Cyboid_t>  iCyb_t;
typedef index_t<Cyboid2_t> iCyb2_t;
typedef index_t<Recboid_t> iRec_t;
typedef index_t<Recboid2_t> iRec2_t;

static vector_t<Cuboid_t,iCub_t> Cuboid;
static vector_t<Cuboid2_t,iCub2_t> Cuboid2;
static vector_t<Cyboid_t,iCyb_t> Cyboid;
static vector_t<Cyboid2_t,iCyb2_t> Cyboid2;
static vector_t<Recboid_t,iRec_t> Recboid;
static vector_t<Recboid2_t,iRec2_t> Recboid2;

//static dPclData_t Spacing;
static distinct_t Type;
static distinct_t RigidType;
static vPclData_t InitialPosition;
static vPclData_t Position;
static vPclData_t Velocity;
static dPclData_t Enthalpy;
//static dPclData_t Ratio;




static void readfile(const char *fname);
static int  readCuboid(FILE *fp,const char* endcommand);
static int  readCuboid2(FILE *fp,const char* endcommand);
static int  readCyboid(FILE *fp,const char* endcommand);
static int  readCyboid2(FILE *fp,const char* endcommand);
static int  readRecboid(FILE *fp,const char* endcommand);
static int  readRecboid2(FILE *fp,const char* endcommand);
static void genparticle();
static void writefile(const char *fname);

int main(int argc,char *argv[])
{
    if(argc>1){
        sprintf(boidname,"%s.boid",argv[1]);
        sprintf(gridname,"%s.grid",argv[1]);
    }
    
    readfile(boidname);
    genparticle();
    writefile(gridname);
}

void readfile(const char* fname)
{
    char buf[1024];
    char token[256];
    int pcldistflag=0, lowerflag=0, upperflag=0;
    FILE *fp = fopen(fname,"r");
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)==NULL)continue;
        if(buf[0]=='#')continue;
        if(sscanf(buf,"%s",token)!=1){fprintf(stderr,"token:%s\n",token);continue;}
    	//fprintf(stderr,"%s\n",token);
        if(strcmp(token,"ParticleDistance")==0){
            if(sscanf(buf," %*s %lf",&ParticleDistance)!=1){fprintf(stderr,"ParticleDistance count not 1\n");goto err;}
            pcldistflag=1;
        }
        if(strcmp(token,"LowerDomain")==0){
            if(sscanf(buf," %*s %lf %lf %lf",&LowerDomain[0], &LowerDomain[1], &LowerDomain[2])!=3){fprintf(stderr,"LowerDomain count not 3\n");goto err;}
            lowerflag=1;
        }
        if(strcmp(token,"UpperDomain")==0){
            if(sscanf(buf," %*s %lf %lf %lf",&UpperDomain[0], &UpperDomain[1], &UpperDomain[2])!=3){fprintf(stderr,"UpperDomain count not 3\n");goto err;}
            upperflag=1;
        }
         if(strcmp(token,"StartCuboid")==0){
            if(readCuboid(fp,"EndCuboid")!=0)goto err;
       // NofCub++;
        }
        if(strcmp(token,"StartCuboid2")==0){
            if(readCuboid2(fp,"EndCuboid2")!=0)goto err;
      //  NofCub2++;
        }
         if(strcmp(token,"StartCyboid")==0){
            if(readCyboid(fp,"EndCyboid")!=0)goto err;
     //   NofCyb++;
        }
         if(strcmp(token,"StartRecboid")==0){
            if(readRecboid(fp,"EndRecboid")!=0)goto err;
     //   NofCyb++;
        }
         if(strcmp(token,"StartRecboid2")==0){
            if(readRecboid2(fp,"EndRecboid2")!=0)goto err;
     //   NofCyb++;
        }
        else if(strcmp(token,"StartCyboid2")==0){
            if(readCyboid2(fp,"EndCyboid2")!=0)goto err;
      //  NofCyb2++;
        };

    }
    if(pcldistflag==0)fprintf(stderr,"no ParticleDistance");
    if(lowerflag==0)fprintf(stderr,"no LowerDomain");
    if(upperflag==0)fprintf(stderr,"no UpperDomain");
    return;
    err:
    fprintf(stderr,"error: \n\tfile:%s\n\tline:%s,token:%s\n",fname,buf,token);
    return;
}

int readCuboid(FILE *fp,const char *endcommand)
{
   // char buf[1024];
    char token[256];
    Cuboid_t cub;
    int spaceflag=0;
    int typeflag=0,rigidtypeflag=0,lowerflag=0, upperflag=0, veloflag=0, enthalpyflag=0;
    cub.type = iType_t(0);
    cub.rigidtype = iType_t(0);
    cub.lower = vec_t(0.0,0.0,0.0);
    cub.upper = vec_t(0.0,0.0,0.0);
    cub.velocity = vec_t(0.0,0.0,0.0);
    while(1){
        fprintf(stderr, "line %d\n", __LINE__);
        if(fscanf(fp,"%s",token)!=1)continue;
        if(strcmp(token,endcommand)==0)break;
        else if(strcmp(token,"Spacing")==0){
            if(fscanf(fp, "%lf", &cub.space)!=1)goto err;
            spaceflag=1;
        }
        else if(strcmp(token,"Type")==0){
            if(fscanf(fp,"%d",&cub.type.setvalue())!=1)goto err;
            typeflag=1;
        }
        else if(strcmp(token,"RigidType")==0){
            if(fscanf(fp,"%d",&cub.rigidtype.setvalue())!=1)goto err;
            rigidtypeflag=1;
        }

        else if(strcmp(token,"Lower")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub.lower[iDim])!=1)goto err;
            }
            lowerflag=1;
        }
        else if(strcmp(token,"Upper")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub.upper[iDim])!=1)goto err;
            }
            upperflag=1;
        }
        else if(strcmp(token,"Velocity")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub.velocity[iDim])!=1)goto err;
            }
            veloflag=1;
        }
        else if(strcmp(token,"Enthalpy")==0){
                if(fscanf(fp,"%lf",&cub.enthalpy)!=1)goto err;
            enthalpyflag=1;
        }
        else{
            fprintf(stderr,"no such indication\n");
            goto err;
        }
    }

    if(spaceflag==0)fprintf(stderr,"no indecatio to Spacing");
    if(typeflag==0)fprintf(stderr,"no indecatio to Type");
    if(rigidtypeflag==0)fprintf(stderr,"no indecatio to RigidType");
    if(lowerflag==0)fprintf(stderr,"no indecatio to Lower");
    if(upperflag==0)fprintf(stderr,"no indecatio to Upper");
    if(veloflag==0)fprintf(stderr,"no indecatio to Velocity");
    if(enthalpyflag==0)fprintf(stderr,"no indecatio to Enthalpy");


   
    if(!(spaceflag && typeflag && rigidtypeflag   && lowerflag && upperflag && veloflag && enthalpyflag  ))return 1;
        fprintf(stderr, "line %d\n", __LINE__);
    Cuboid += cub;
    return 0;
    
 err:
    fprintf(stderr,"error: token:%s",token);
    return 1;
}
int readCuboid2(FILE *fp,const char *endcommand)
{
//    char buf[1024];
    char token[256];
    Cuboid2_t cub2;
    int spaceflag=0;
    int typeflag=0, lowerflag=0, upperflag=0, veloflag=0, enthalpyflag=0;
    cub2.type = iType_t(0);
    cub2.lower = vec_t(0.0,0.0,0.0);
    cub2.upper = vec_t(0.0,0.0,0.0);
    cub2.velocity = vec_t(0.0,0.0,0.0);
    while(1){
        fprintf(stderr, "line %d\n", __LINE__);
        if(fscanf(fp,"%s",token)!=1)continue;
        if(strcmp(token,endcommand)==0)break;
        else if(strcmp(token,"Spacing")==0){
            if(fscanf(fp, "%lf", &cub2.space)!=1)goto err;
            spaceflag=1;
        }
        else if(strcmp(token,"Type")==0){
            if(fscanf(fp,"%d",&cub2.type.setvalue())!=1)goto err;
            typeflag=1;
        }
        else if(strcmp(token,"Lower")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub2.lower[iDim])!=1)goto err;
            }
            lowerflag=1;
        }
        else if(strcmp(token,"Upper")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub2.upper[iDim])!=1)goto err;
            }
            upperflag=1;
        }
        else if(strcmp(token,"Velocity")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub2.velocity[iDim])!=1)goto err;
            }
            veloflag=1;
        }
        else if(strcmp(token,"Enthalpy")==0){
                if(fscanf(fp,"%lf",&cub2.enthalpy)!=1)goto err;
            enthalpyflag=1;
        }



        else{
            fprintf(stderr,"no such indication\n");
            goto err;
        }
    }

    if(spaceflag==0)fprintf(stderr,"no indecatio to Spacing");
    if(typeflag==0)fprintf(stderr,"no indecatio to Type");
    if(lowerflag==0)fprintf(stderr,"no indecatio to Lower");
    if(upperflag==0)fprintf(stderr,"no indecatio to Upper");
    if(veloflag==0)fprintf(stderr,"no indecatio to Velocity");
    if(enthalpyflag==0)fprintf(stderr,"no indecatio to Enthalpy");


   
    if(!(spaceflag && typeflag && lowerflag && upperflag && veloflag && enthalpyflag  ))return 1;
        fprintf(stderr, "line %d\n", __LINE__);
    Cuboid2 += cub2;
    return 0;
    
 err:
    fprintf(stderr,"error: token:%s",token);
    return 1;
}


int readCyboid(FILE *fp,const char *endcommand)
{
//    char buf[1024];
    char token[256];
    Cyboid_t cyb;
    int spaceflag=0;
    int typeflag=0,rigidtypeflag=0, lowerflag=0, upperflag=0, veloflag=0, enthalpyflag=0;
    int ratioflag=0;
    cyb.type = iType_t(0);
    cyb.rigidtype = iType_t(0);
    cyb.lower = vec_t(0.0,0.0,0.0);
    cyb.upper = vec_t(0.0,0.0,0.0);
    cyb.velocity = vec_t(0.0,0.0,0.0);
    while(1){
        fprintf(stderr, "line %d\n", __LINE__);
        if(fscanf(fp,"%s",token)!=1)continue;
        if(strcmp(token,endcommand)==0)break;
        else if(strcmp(token,"Spacing")==0){
            if(fscanf(fp, "%lf", &cyb.space)!=1)goto err;
            spaceflag=1;
        }
        else if(strcmp(token,"Type")==0){
            if(fscanf(fp,"%d",&cyb.type.setvalue())!=1)goto err;
            typeflag=1;
        }
        else if(strcmp(token,"RigidType")==0){
            if(fscanf(fp,"%d",&cyb.rigidtype.setvalue())!=1)goto err;
            rigidtypeflag=1;
        }
        else if(strcmp(token,"Lower")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cyb.lower[iDim])!=1)goto err;
            }
            lowerflag=1;
        }
        else if(strcmp(token,"Upper")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cyb.upper[iDim])!=1)goto err;
            }
            upperflag=1;
        }
        else if(strcmp(token,"Velocity")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cyb.velocity[iDim])!=1)goto err;
            }
            veloflag=1;
        }
        else if(strcmp(token,"Enthalpy")==0){
                if(fscanf(fp,"%lf",&cyb.enthalpy)!=1)goto err;
            enthalpyflag=1;
        }
        else if(strcmp(token,"Ratio")==0){
                if(fscanf(fp,"%lf",&cyb.ratio)!=1)goto err;
            ratioflag=1;
        }


        else{
            fprintf(stderr,"no such indication\n");
            goto err;
        }
    }

    if(spaceflag==0)fprintf(stderr,"no indecatio to Spacing");
    if(typeflag==0)fprintf(stderr,"no indecatio to Type");
    if(rigidtypeflag==0)fprintf(stderr,"no indecatio to Type");
    if(lowerflag==0)fprintf(stderr,"no indecatio to Lower");
    if(upperflag==0)fprintf(stderr,"no indecatio to Upper");
    if(veloflag==0)fprintf(stderr,"no indecatio to Velocity");
    if(enthalpyflag==0)fprintf(stderr,"no indecatio to Enthalpy");
    if(ratioflag==0)fprintf(stderr,"no indecatio to Ratio");

   
    if(!(spaceflag && typeflag && rigidtypeflag && lowerflag && upperflag && veloflag && enthalpyflag && ratioflag  ))return 1;
        fprintf(stderr, "line %d\n", __LINE__);
    Cyboid += cyb;
    return 0;
    
 err:
    fprintf(stderr,"error: token:%s",token);
    return 1;
}
      

int readCyboid2(FILE *fp,const char *endcommand)
{
//    char buf[1024];
    char token[256];
    Cyboid2_t cyb2;
    int spaceflag=0;
    int typeflag=0, lowerflag=0, upperflag=0, veloflag=0, enthalpyflag=0;
    int ratioflag=0;
    cyb2.type = iType_t(0);
    cyb2.lower = vec_t(0.0,0.0,0.0);
    cyb2.upper = vec_t(0.0,0.0,0.0);
    cyb2.velocity = vec_t(0.0,0.0,0.0);
    while(1){
        fprintf(stderr, "line %d\n", __LINE__);
        if(fscanf(fp,"%s",token)!=1)continue;
        if(strcmp(token,endcommand)==0)break;
        else if(strcmp(token,"Spacing")==0){
            if(fscanf(fp, "%lf", &cyb2.space)!=1)goto err;
            spaceflag=1;
        }
        else if(strcmp(token,"Type")==0){
            if(fscanf(fp,"%d",&cyb2.type.setvalue())!=1)goto err;
            typeflag=1;
        }
        else if(strcmp(token,"Lower")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cyb2.lower[iDim])!=1)goto err;
            }
            lowerflag=1;
        }
        else if(strcmp(token,"Upper")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cyb2.upper[iDim])!=1)goto err;
            }
            upperflag=1;
        }
        else if(strcmp(token,"Velocity")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cyb2.velocity[iDim])!=1)goto err;
            }
            veloflag=1;
        }
        else if(strcmp(token,"Enthalpy")==0){
                if(fscanf(fp,"%lf",&cyb2.enthalpy)!=1)goto err;
            enthalpyflag=1;
        }
        else if(strcmp(token,"Ratio")==0){
                if(fscanf(fp,"%lf",&cyb2.ratio)!=1)goto err;
            ratioflag=1;
        }



        else{
            fprintf(stderr,"no such indication\n");
            goto err;
        }
    }

    if(spaceflag==0)fprintf(stderr,"no indecatio to Spacing");
    if(typeflag==0)fprintf(stderr,"no indecatio to Type");
    if(lowerflag==0)fprintf(stderr,"no indecatio to Lower");
    if(upperflag==0)fprintf(stderr,"no indecatio to Upper");
    if(veloflag==0)fprintf(stderr,"no indecatio to Velocity");
    if(enthalpyflag==0)fprintf(stderr,"no indecatio to Enthalpy");
    if(ratioflag==0)fprintf(stderr,"no indecatio to Ratio");

   
    if(!(spaceflag && typeflag && lowerflag && upperflag && veloflag && enthalpyflag && ratioflag  ))return 1;
        fprintf(stderr, "line %d\n", __LINE__);
    Cyboid2 += cyb2;
    return 0;
    
 err:
    fprintf(stderr,"error: token:%s",token);
    return 1;
}

int readRecboid(FILE *fp,const char *endcommand)
{
//    char buf[1024];
    char token[256];
    Recboid_t rec;
    int spaceflag=0;
    int typeflag=0, lowerflag=0, upperflag=0, veloflag=0, enthalpyflag=0;
    int angleflag=0;
    rec.type = iType_t(0);
    rec.lower = vec_t(0.0,0.0,0.0);
    rec.upper = vec_t(0.0,0.0,0.0);
    rec.velocity = vec_t(0.0,0.0,0.0);
    while(1){
        fprintf(stderr, "line %d\n", __LINE__);
        if(fscanf(fp,"%s",token)!=1)continue;
        if(strcmp(token,endcommand)==0)break;
        else if(strcmp(token,"Spacing")==0){
            if(fscanf(fp, "%lf", &rec.space)!=1)goto err;
            spaceflag=1;
        }
        else if(strcmp(token,"Type")==0){
            if(fscanf(fp,"%d",&rec.type.setvalue())!=1)goto err;
            typeflag=1;
        }
        else if(strcmp(token,"Lower")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&rec.lower[iDim])!=1)goto err;
            }
            lowerflag=1;
        }
        else if(strcmp(token,"Upper")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&rec.upper[iDim])!=1)goto err;
            }
            upperflag=1;
        }
        else if(strcmp(token,"Velocity")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&rec.velocity[iDim])!=1)goto err;
            }
            veloflag=1;
        }
        else if(strcmp(token,"Enthalpy")==0){
                if(fscanf(fp,"%lf",&rec.enthalpy)!=1)goto err;
            enthalpyflag=1;
        }
        else if(strcmp(token,"Angle")==0){
                if(fscanf(fp,"%lf",&rec.angle)!=1)goto err;
            angleflag=1;
        }



        else{
            fprintf(stderr,"no such indication\n");
            goto err;
        }
    }

    if(spaceflag==0)fprintf(stderr,"no indecatio to Spacing");
    if(typeflag==0)fprintf(stderr,"no indecatio to Type");
    if(lowerflag==0)fprintf(stderr,"no indecatio to Lower");
    if(upperflag==0)fprintf(stderr,"no indecatio to Upper");
    if(veloflag==0)fprintf(stderr,"no indecatio to Velocity");
    if(enthalpyflag==0)fprintf(stderr,"no indecatio to Enthalpy");
    if(angleflag==0)fprintf(stderr,"no indecatio to Ratio");

   
    if(!(spaceflag && typeflag && lowerflag && upperflag && veloflag && enthalpyflag && angleflag  ))return 1;
        fprintf(stderr, "line %d\n", __LINE__);
    Recboid += rec;
    return 0;
    
 err:
    fprintf(stderr,"error: token:%s",token);
    return 1;
}

int readRecboid2(FILE *fp,const char *endcommand)
{
//    char buf[1024];
    char token[256];
    Recboid2_t rec2;
    int spaceflag=0;
    int typeflag=0, lowerflag=0, upperflag=0, veloflag=0, enthalpyflag=0;
    int angleflag=0;
    rec2.type = iType_t(0);
    rec2.lower = vec_t(0.0,0.0,0.0);
    rec2.upper = vec_t(0.0,0.0,0.0);
    rec2.velocity = vec_t(0.0,0.0,0.0);
    while(1){
        fprintf(stderr, "line %d\n", __LINE__);
        if(fscanf(fp,"%s",token)!=1)continue;
        if(strcmp(token,endcommand)==0)break;
        else if(strcmp(token,"Spacing")==0){
            if(fscanf(fp, "%lf", &rec2.space)!=1)goto err;
            spaceflag=1;
        }
        else if(strcmp(token,"Type")==0){
            if(fscanf(fp,"%d",&rec2.type.setvalue())!=1)goto err;
            typeflag=1;
        }
        else if(strcmp(token,"Lower")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&rec2.lower[iDim])!=1)goto err;
            }
            lowerflag=1;
        }
        else if(strcmp(token,"Upper")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&rec2.upper[iDim])!=1)goto err;
            }
            upperflag=1;
        }
        else if(strcmp(token,"Velocity")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&rec2.velocity[iDim])!=1)goto err;
            }
            veloflag=1;
        }
        else if(strcmp(token,"Enthalpy")==0){
                if(fscanf(fp,"%lf",&rec2.enthalpy)!=1)goto err;
            enthalpyflag=1;
        }
        else if(strcmp(token,"Angle")==0){
                if(fscanf(fp,"%lf",&rec2.angle)!=1)goto err;
            angleflag=1;
        }



        else{
            fprintf(stderr,"no such indication\n");
            goto err;
        }
    }

    if(spaceflag==0)fprintf(stderr,"no indecatio to Spacing");
    if(typeflag==0)fprintf(stderr,"no indecatio to Type");
    if(lowerflag==0)fprintf(stderr,"no indecatio to Lower");
    if(upperflag==0)fprintf(stderr,"no indecatio to Upper");
    if(veloflag==0)fprintf(stderr,"no indecatio to Velocity");
    if(enthalpyflag==0)fprintf(stderr,"no indecatio to Enthalpy");
    if(angleflag==0)fprintf(stderr,"no indecatio to Angle");

   
    if(!(spaceflag && typeflag && lowerflag && upperflag && veloflag && enthalpyflag && angleflag  ))return 1;
        fprintf(stderr, "line %d\n", __LINE__);
    Recboid2 += rec2;
    return 0;
    
 err:
    fprintf(stderr,"error: token:%s",token);
    return 1;
}
      
void genparticle()
{
    const iCub_t cubcount = Cuboid.count();
    for(iCub_t iCub(0);iCub<cubcount;++iCub){
        const Cuboid_t& cub = Cuboid[iCub];
        const vec_t width = cub.upper-cub.lower;
        const vec3<int> count = vec3<int>(round(width[0]/cub.space),round(width[1]/cub.space),round(width[2]/cub.space));
        const vec_t spacing = vec_t(width[0]/count[0],width[1]/count[1],width[2]/count[2]);

        for(double px=cub.lower[0]+0.5*spacing[0]; px<cub.upper[0]-0.49*spacing[0]; px+=spacing[0]){
            for(double py=cub.lower[1]+0.5*spacing[1]; py<cub.upper[1]-0.49*spacing[1]; py+=spacing[1]){
                 for(double pz=cub.lower[2]+0.5*spacing[2]; pz<cub.upper[2]-0.49*spacing[2]; pz+=spacing[2]){

                    fprintf(stderr, "p %e %e %e\n", px, py, pz);
                    Type += cub.type;
                    RigidType += cub.rigidtype;
                    InitialPosition += vec_t(px,py,pz);
                    Position += vec_t(px,py,pz);
                    Velocity += cub.velocity;
                    Enthalpy += cub.enthalpy;


                     
                }
            }
        }
    }

    const iCub2_t cubcount2 = Cuboid2.count();
    for(iCub2_t iCub2(0);iCub2<cubcount2;++iCub2){
        const Cuboid2_t& cub2 = Cuboid2[iCub2];
        const vec_t width = cub2.upper-cub2.lower;
        const vec3<int> count = vec3<int>(round(width[0]/cub2.space),round(width[1]/cub2.space),round(width[2]/cub2.space));
        const vec_t spacing = vec_t(width[0]/count[0],width[1]/count[1],width[2]/count[2]);

        for(double px=cub2.lower[0]+0.01*spacing[0]; px<cub2.upper[0]-0.0*spacing[0]; px+=spacing[0]){
            for(double py=cub2.lower[1]+0.01*spacing[1]; py<cub2.upper[1]-0.0*spacing[1]; py+=spacing[1]){
                 for(double pz=cub2.lower[2]+0.5*spacing[2]; pz<cub2.upper[2]-0.49*spacing[2]; pz+=spacing[2]){

                     fprintf(stderr, "p %e %e %e\n", px, py, pz);
                    Type += cub2.type;
                    Position += vec_t(px,py,pz);
                    Velocity += cub2.velocity;
                    Enthalpy += cub2.enthalpy;


                     
                }
            }
        }
    }

    const iCyb_t cybcount = Cyboid.count();
    for(iCyb_t iCyb(0);iCyb<cybcount;++iCyb){
        const Cyboid_t& cyb = Cyboid[iCyb];
        const vec_t width = cyb.upper-cyb.lower;
        const vec_t center = 0.5*(cyb.upper+cyb.lower);
        const vec3<int> count = vec3<int>(round(width[0]/cyb.space),round(width[1]/cyb.space),round(width[2]/cyb.space));
        const vec_t spacing = vec_t(width[0]/count[0],width[1]/count[1],width[2]/count[2]);

        for(double px=cyb.lower[0]+0.5*spacing[0]; px<cyb.upper[0]-0.49*spacing[0]; px+=spacing[0]){
            for(double py=cyb.lower[1]+0.5*spacing[1]; py<cyb.upper[1]-0.49*spacing[1]; py+=spacing[1]){
                 for(double pz=cyb.lower[2]+0.5*spacing[2]; pz<cyb.upper[2]-0.49*spacing[2]; pz+=spacing[2]){
                     double x = px - center[0];
                                double y = py - center[1];
                                double z = pz - center[2];
                                double r_squared = x*x + y*y + z*z;

                                double inner_radius_squared = 0.25 * width[0] * width[0] * cyb.ratio * cyb.ratio;
                                double outer_radius_squared = 0.25 * width[0] * width[0];

                                if (r_squared > inner_radius_squared && r_squared <= outer_radius_squared) {

                     fprintf(stderr, "p %e %e %e\n", px, py, pz);
                    Type += cyb.type;
                      RigidType += cyb.rigidtype;
                    Position += vec_t(px,py,pz);
                    Velocity += cyb.velocity;
                    Enthalpy += cyb.enthalpy;
                     }                   
                }
            }
        }
    }

    const iCyb2_t cybcount2 = Cyboid2.count();
    for(iCyb2_t iCyb2(0);iCyb2<cybcount2;++iCyb2){
        const Cyboid2_t& cyb2 = Cyboid2[iCyb2];
        const vec_t width = cyb2.upper-cyb2.lower;
        const vec_t center = 0.5*(cyb2.upper+cyb2.lower);
        const vec3<int> count = vec3<int>(round(width[0]/cyb2.space),round(width[1]/cyb2.space),round(width[2]/cyb2.space));
        const vec_t spacing = vec_t(width[0]/count[0],width[1]/count[1],width[2]/count[2]);

        for(double px=cyb2.lower[0]+0.01*spacing[0]; px<cyb2.upper[0]-0.0*spacing[0]; px+=spacing[0]){
            for(double py=cyb2.lower[1]+0.01*spacing[1]; py<cyb2.upper[1]-0.0*spacing[1]; py+=spacing[1]){
                 for(double pz=cyb2.lower[2]+0.5*spacing[2]; pz<cyb2.upper[2]-0.49*spacing[2]; pz+=spacing[2]){
                       double x = (px-center[0]);
                       double y = (py-center[1]);
                  if(x*x+y*y <= 0.5*0.5*0.5*0.5*width[0]*width[0]*width[1]*width[1] && x*x+y*y > 0.5*0.5*0.5*0.5*width[0]*width[0]*width[1]*width[1]*cyb2.ratio*cyb2.ratio*cyb2.ratio*cyb2.ratio){

                     fprintf(stderr, "p %e %e %e\n", px, py, pz);
                    Type += cyb2.type;
                    Position += vec_t(px,py,pz);
                    Velocity += cyb2.velocity;
                    Enthalpy += cyb2.enthalpy;

                   }
                     
                }
            }
        }
    }

    const iRec_t reccount = Recboid.count();
    for(iRec_t iRec(0);iRec<reccount;++iRec){
        const Recboid_t& rec = Recboid[iRec];
        const vec_t width = rec.upper-rec.lower;
       // const vec_t center = 0.5*(cyb2.upper+cyb2.lower);
        const vec3<int> count = vec3<int>(round(width[0]/rec.space),round(width[1]/rec.space),round(width[2]/rec.space));
        const vec_t spacing = vec_t(width[0]/count[0],width[1]/count[1],width[2]/count[2]);

        for(double px=rec.lower[0]+0.01*spacing[0]; px<rec.upper[0]-0.0*spacing[0]; px+=spacing[0]){
            for(double py=rec.lower[1]+0.01*spacing[1]; py<rec.upper[1]-0.0*spacing[0]; py+=spacing[1]){
                 for(double pz=rec.lower[2]+0.5*spacing[2]; pz<rec.upper[2]-0.49*spacing[2]; pz+=spacing[2]){
                      // double tan(rec.angle) = py/px;
                  // double x = px*px;
                  // double y = py*py;
                  // double distance = sqrt(x+y);
                   //  double x = rec.upper[0]-rec.lower[0];
                       
                  if(tan(rec.angle*3.1415/180) > py/px){

                     fprintf(stderr, "p %e %e %e\n", px, py, pz);
                    Type += rec.type;
                    Position += vec_t(px,py,pz);
                    Velocity += rec.velocity;
                    Enthalpy += rec.enthalpy;

                   }
                     
                }
            }
        }
    }

  const iRec2_t rec2count = Recboid2.count();
    for(iRec2_t iRec2(0);iRec2<rec2count;++iRec2){
        const Recboid2_t& rec2 = Recboid2[iRec2];
        const vec_t width = rec2.upper-rec2.lower;
       // const vec_t center = 0.5*(cyb2.upper+cyb2.lower);
        const vec3<int> count = vec3<int>(round(width[0]/rec2.space),round(width[1]/rec2.space),round(width[2]/rec2.space));
        const vec_t spacing = vec_t(width[0]/count[0],width[1]/count[1],width[2]/count[2]);

        for(double px=rec2.lower[0]+0.01*spacing[0]; px<rec2.upper[0]-0.0*spacing[0]; px+=spacing[0]){
            for(double py=rec2.lower[1]+0.01*spacing[1]; py<rec2.upper[1]-0.0*spacing[0]; py+=spacing[1]){
                 for(double pz=rec2.lower[2]+0.5*spacing[2]; pz<rec2.upper[2]-0.49*spacing[2]; pz+=spacing[2]){
                  double x = px*cos(rec2.angle*3.1415/180)-py*sin(rec2.angle*3.1415/180);
                  double y = px*sin(rec2.angle*3.1415/180)+py*cos(rec2.angle*3.1415/180);
                       
                

                     fprintf(stderr, "p %e %e %e\n", x, y, pz);
                    Type += rec2.type;
                    Position += vec_t(x,y,pz);
                    Velocity += rec2.velocity;
                    Enthalpy += rec2.enthalpy;

                   }
                     
                }
            }
        }
    






    fprintf(stderr, "%d particles were generated\n", Type.count().getvalue());

}



void writefile(const char *fname)
{
    FILE *fp = fopen(fname,"w");
    const iPcl_t& pclCount = Type.count();
    fprintf(fp,"%lf\n",0.0);
    fprintf(fp,"%d %e  %e %e %e  %e %e %e\n",
            pclCount.getvalue(),
            ParticleDistance,
            LowerDomain[0], UpperDomain[0],
            LowerDomain[1], UpperDomain[1],
            LowerDomain[2], UpperDomain[2]
            );
    for(iPcl_t iPcl(0);iPcl<pclCount;++iPcl){
        fprintf(fp,"%d   %e %e %e %e %e %e  %e %e %e \n",
                Type[iPcl].getvalue(),
                Position[iPcl][0],Position[iPcl][1],Position[iPcl][2],
                Position[iPcl][0],Position[iPcl][1],Position[iPcl][2],
                Velocity[iPcl][0],Velocity[iPcl][1],Velocity[iPcl][2]);
        
        
        
    }
    fclose(fp);
}


                
