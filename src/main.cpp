//================================================================================================//
//------------------------------------------------------------------------------------------------//
//              FSI-Mopving Particle Hydrodynamics (Full-Explicit)                                //
//------------------------------------------------------------------------------------------------//
//    Copyright       : Ryo Yokoyama                                                              //
//    OpenACC GPU     : 2025                                                                      //
//    HPCSDK 25.1 CUDA version 12.4                                                            	  //
//    MPI-OpenACC hybrid Parallel Computation							                          //
//    For HPCSDK https://developer.nvidia.com/nvidia-hpc-sdk-241-download                         //
//    For MPI   https://www.open-mpi.org/software/ompi/v4.1/					                  //
//    Final Check     : 2025 28 May                                                               //
//    												                                              //
//                                                                                                //
//================================================================================================//
//        PLEASE DO NOT DISTRIBUTE THIS CODE TO OUTSIDE OF OKAMOTO LABO                           //
//                                                                                                //
//================================================================================================//
//                      HOW TO COMPILE IN LINUX SYSTEM                                            //
//                   1. /FSI/generator/make/                                                      //
//                   2. /FSI/source/make                                                          //
//                   3. /results/and ./generate.sh  and ./execute.sh                              //
//================================================================================================//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include "errorfunc.h"
#include "log.h"
#include <mutex>
#include <openacc.h>


#ifdef _CUDA
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#endif

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
#define TWO_DIMENSIONAL

//****PLEASE CHOOSE MODULE ***//

#define Bar_Module
//#define DAM_Module
//#define Turek_Hron
//#define Rolling1
//#define Rolling//#define Rolling
// #define Hydroelastic

#define DIM 3
#define CUDA
#define FLUID
#define STRUCTURE


// Property definition
#define TYPE_COUNT   6
#define FLUID_BEGIN  0
#define FLUID_END    2
#define STRUCTURE_BEGIN 2
#define STRUCTURE_END   4
#define WALL_BEGIN   4
#define WALL_END     6

#define  DEFAULT_LOG  "sample.log"
#define  DEFAULT_DATA "sample.data"
#define  DEFAULT_GRID "sample.grid"
#define  DEFAULT_PROF "sample%03d.prof"
#define  DEFAULT_VTK  "sample%03d.vtk"

// Calculation and Output
static double ParticleSpacing=0.0;
static double ParticleVolume=0.0;
static double OutputInterval=0.0;
static double OutputNext=0.0;
static double VtkOutputInterval=0.0;
static double VtkOutputNext=0.0;
static double EndTime=0.0;
static double Time=0.0;
static double Dt=1.0e100;
static double Elastic_Dt=1.0e100;
static double DomainMin[DIM];
static double DomainMax[DIM];
static double DomainWidth[DIM];
#pragma acc declare create(ParticleSpacing,ParticleVolume,Elastic_Dt,Dt,DomainMin,DomainMax,DomainWidth)

#define Mod(x,w) ((x)-(w)*floor((x)/(w)))   // mod 

#define MAX_NEIGHBOR_COUNT 512
// Particle
static int ParticleCount;
static int *Property;                     // particle type
static double (*Mass);                    // mass
static double (*Position)[DIM];
static double (*InitialPosition)[DIM];
static double (*Velocity)[DIM];           // momentum
static double (*Force)[DIM];              // total explicit force acting on the particle
static int *NeighborCount;                   // [ParticleCount]
static int (*Neighbor)[MAX_NEIGHBOR_COUNT];  // [ParticleCount]
static int *InitialNeighborCount;                   // [ParticleCount]
static int (*InitialNeighbor)[MAX_NEIGHBOR_COUNT];  // [ParticleCount]
static int *InitialStructureNeighborCount;                   // [ParticleCount]
static int (*InitialStructureNeighbor)[MAX_NEIGHBOR_COUNT];  // [ParticleCount]
static double (*NeighborCalculatedPosition)[DIM];
#define MARGIN (0.1*ParticleSpacing)
#pragma acc declare create(ParticleCount,Property,Mass,Position,InitialPosition,Velocity,Force,NeighborCount,Neighbor,InitialNeighborCount,InitialNeighbor,NeighborCalculatedPosition)
#pragma acc declare create(InitialNeighbor,InitialNeighborCount)


// BackGroundCells
#ifdef TWO_DIMENSIONAL
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1])
#else
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1]*CellCount[2] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1]*CellCount[2] + ((iCZ)%CellCount[2]+CellCount[2])%CellCount[2])
#endif

static int PowerParticleCount;
static int ParticleCountPower;                   
static double CellWidth = 0.0;
static int CellCount[DIM];
static int CellCounts = 0;
static int *CellParticleBegin;  // beginning of particles in the cell
static int *CellParticleEnd;    // number of particles in the cell
static int *CellIndex;  // [ParticleCountPower>>1]
static int *CellParticle;       // array of particle id in the cells) [ParticleCountPower>>1]
#pragma acc declare create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount,CellCounts,CellParticleBegin,CellParticleEnd,CellIndex,CellParticle)

// Type
static double Density[TYPE_COUNT];
static double BulkModulus[TYPE_COUNT];
static double BulkViscosity[TYPE_COUNT];
static double ShearViscosity[TYPE_COUNT];
static double SurfaceTension[TYPE_COUNT];
static double CofA[TYPE_COUNT];   // coefficient for attractive pressure
static double CofK;               // coefficinet (ratio) for diffuse interface thickness normalized by ParticleSpacing
static double InteractionRatio[TYPE_COUNT][TYPE_COUNT];
#pragma acc declare create(Density,BulkModulus,BulkViscosity,ShearViscosity,SurfaceTension,CofA,CofK,InteractionRatio)


// Fluid
static int FluidParticleBegin;
static int FluidParticleEnd;
static double *DensityA;        // number density per unit volume for attractive pressure
static double (*GravityCenter)[DIM];
static double *PressureA;       // attractive pressure (surface tension)
static double *VolStrainP;        // number density per unit volume for base pressure
static double *DivergenceP;     // volumetric strainrate for pressure B
static double *PressureP;       // base pressure
static double *VirialPressureAtParticle; // VirialPressureInSingleParticleRegion
static double (*VirialStressAtParticle)[DIM][DIM];
static double *Mu;              // viscosity coefficient for shear
static double *Lambda;          // viscosity coefficient for bulk
static double *Kappa;           // bulk modulus
#pragma acc declare create(DensityA,GravityCenter,PressureA,VolStrainP,DivergenceP,PressureP,VirialPressureAtParticle,VirialStressAtParticle,Mu,Lambda,Kappa)

static double Gravity[DIM] = {0.0,0.0,0.0};
#pragma acc declare create(Gravity)

// Wall
static int WallParticleBegin;
static int WallParticleEnd;
static double WallCenter[WALL_END][DIM];
static double WallVelocity[WALL_END][DIM];
static double WallOmega[WALL_END][DIM];
static double WallRotation[WALL_END][DIM][DIM];
#pragma acc declare create(WallCenter,WallVelocity,WallOmega,WallRotation)

//Structure
static double YoungModulus[TYPE_COUNT];
static double PoissonRatio[TYPE_COUNT];
static int StructureParticleBegin;
static int StructureParticleEnd;
static double (*AngularVelocity)[DIM];
static double *LambdaLames;
static double *MuLames;
static double (*RotationMatrix)[DIM][DIM];
static double (*quaternion)[4];
static double (*Normalizer)[DIM][DIM];
static double (*DeformGradient)[DIM][DIM];
static double (*Strain)[DIM][DIM];
static double (*Stress)[DIM][DIM];
static double (*Acceleration)[DIM];
static double (*Conversion)[DIM][DIM];
static double (*Young);
static int *Original;
#pragma acc declare create(YoungModulus,PoissonRatio,AngularVelocity,LambdaLames,MuLames,RotationMatrix,quaternion,Normalizer,DeformGradient,Strain,Stress,Acceleration,Conversion,Original,Young)


// proceedures
static void readDataFile(char *filename);
static void readGridFile(char *filename);
static void writeProfFile(char *filename);
static void writeVtkFile(char *filename);
static void initializeWeight( void );
static void initializeDomain( void );
static void initializeFluid( void );
static void initializeWall( void );
static void initializeStructure( void );
static void calculateConvection();
static void calculateWall();
static void calculatePeriodicBoundary();
static void resetForce();
static int neighborCalculation();
static void calculateNeighbor();
static void calculateInitialNeighbor();
static void calculatePhysicalCoefficients();
static void calculateDensityA();
static void calculatePressureA();
static void calculateGravityCenter();
static void calculateDiffuseInterface();
static void calculateDensityP();
static void calculateDivergenceP();
static void calculatePressureP();
static void calculateViscosityV();
static void calculateGravity();
static void calculateAcceleration();
static void calculateVirialPressureAtParticle();
static void calculateVirialStressAtParticle();

//Procedure of the elastic body calculation in Explicit
static void calculateLamesconstant();
static void calculateElasticDeformationVector();
static void calculateStress();
static void resetAccel();
static void setInitialVelocityProfile();
static void calculateNormalizer();
static void calculateStressForce();
static void selectFreeGPU();
static void calculateInterfaceForce();
static void updateElasticPosition();



// dual kernel functions
static double RadiusRatioA;
static double RadiusRatioG;
static double RadiusRatioP;
static double RadiusRatioV;

static double MaxRadius = 0.0;
static double RadiusA = 0.0;
static double RadiusG = 0.0;
static double RadiusP = 0.0;
static double RadiusV = 0.0;
static double Swa = 1.0;
static double Swg = 1.0;
static double Swp = 1.0;
static double Swv = 1.0;
static double N0a = 1.0;
static double N0p = 1.0;
static double R2g = 1.0;

#pragma acc declare create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


#pragma acc routine seq
static double weight(const double rij[DIM], double radius)
{
    // Compute the magnitude of rij
    double r_squared = 0.0;
    
    #ifdef TWO_DIMENSIONAL
    #pragma acc loop seq
    for (int iD = 0; iD < 2; ++iD)
    {
        r_squared += rij[iD] * rij[iD];
    }
    #else
        #pragma acc loop seq
    for (int iD = 0; iD < 3; ++iD)
    {
        r_squared += rij[iD] * rij[iD];
    }
    #endif

    double r = sqrt(r_squared);
    double q = r / radius;

#ifdef TWO_DIMENSIONAL
    return (1.0 / Swp) * (1.0 / (radius * radius)) * ((1.0 - q) * (1.0 - q));
#else
    return (1.0 / Swp) * (1.0 / (radius * radius * radius)) * ((1.0 - q) * (1.0 - q));
#endif
}


#pragma acc routine seq
static double wa(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#else
    return 1.0/Swa * 1.0/(h*h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#endif
}

#pragma acc routine seq
static double dwadr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#else
    return 1.0/Swa * 1.0/(h*h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#endif
}

#pragma acc routine seq
static double wg(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwgdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wp(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swp * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwpdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wv(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swv * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwvdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


	clock_t cFrom, cTill, cStart, cEnd;
	clock_t cNeigh=0, cExplicit=0, cVirial=0, cOther=0;

const double YMIN = 0.0;
const double YMAX = 0.41;
const double UMAX = 1.0;
const double H = YMAX - YMIN;


const double L = 0.20;               // Beam length [m]
const double kL = 1.875;             // First vibration mode
const double k = kL / L;
const double K = 3.25e6;             // Bulk modulus [Pa]
const double vL = 0.02;              // Tip velocity coefficient

// f(x) in radians with x in degrees → convert to radians
double compute_fx(double x) {
    double kx = k * x;
    double term1 = (cos(kL) + cosh(kL)) * (cosh(kx) - cos(kx));
    double term2 = (sin(kL) - sinh(kL)) * (sinh(kx) - sin(kx));
    return term1 + term2;
} 


void setInitialVelocityProfile() {
#ifdef Bar_Module
#pragma acc kernels present(Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM])
#pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        // Local parameters
        double rho = Density[Property[iP]];         // Volume density
        double c0 = sqrt(K / rho);                 // Speed of sound
        double VL = vL * c0;                       // vL * c0 term

        // Position
        double x = InitialPosition[iP][0];                // x-axis assumed as beam axis

        // Normalized mode shape f(x) / f(L)
        double fx = compute_fx(x);
        double fL = compute_fx(L);

        // Set velocity
        Velocity[iP][0] = 0.0;
        Velocity[iP][1] = 0.01*c0 * fx / fL;  // Initial vertical velocity
        Velocity[iP][2] = 0.0;
    }
#endif

#ifdef Turek_Hron
#pragma acc parallel loop present(Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Velocity[0:ParticleCount][0:DIM])
for (int i = FluidParticleBegin; i < FluidParticleEnd; ++i) {
    double x = Position[i][0];
    double y = Position[i][1];

    if (x <= 0.01) {  // inlet region
        double uy = y - YMIN; // y in [0, H]
        double u = (1.5*4.0 * UMAX / (H * H)) * uy * (H - uy);
        Velocity[i][0] = u;
        Velocity[i][1] = 0.0;
        Velocity[i][2] = 0.0;
    }
       if (x > 1.5 && Time <0.7) {  // inlet region
        double uy = y - YMIN; // y in [0, H]
        double u = (4.0 * UMAX / (H * H)) * uy * (H - uy);
        Velocity[i][0] = u;
        Velocity[i][1] = 0.0;
        Velocity[i][2] = 0.0;
    }
}

#endif


}



/*
// Define constants for plate dimensions and vibration modes
const double a = 0.4; // Plate length (m)
const double b = 0.4; // Plate width (m)
const int m = 2; // Vibration mode in x-direction
const int n = 1; // Vibration mode in y-direction

void setInitialVelocityProfile() {
    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = std::numeric_limits<double>::max();
    double yMax = std::numeric_limits<double>::lowest();

    // Calculate the min and max of x and y coordinates
    #pragma omp parallel for reduction(min:xMin, yMin) reduction(max:xMax, yMax)
    for (int i = 0; i < ParticleCount; ++i) {
        xMin = std::min(xMin, Position[i][0]);
        xMax = std::max(xMax, Position[i][0]);
        yMin = std::min(yMin, Position[i][1]);
        yMax = std::max(yMax, Position[i][1]);
    }

    // Normalize x and y and set initial velocity
    #pragma omp parallel for
    for (int i = 0; i < ParticleCount; ++i) {
        double x = (Position[i][0] - xMin) ; // Normalize x to [0, 1]
        double y = (Position[i][1] - yMin); // Normalize y to [0, 1]

        // Calculate initial velocity in z-direction based on Eq. (22)
        Velocity[i][2] = sin(m * M_PI * x / a) * sin(n * M_PI * y / b);
        // Set other velocity components to zero
        Velocity[i][0] = 0.0; // x-velocity
        Velocity[i][1] = 0.0; // y-velocity
    }
}

*/





int main(int argc, char *argv[])
{
  
	
    char logfilename[1024]  = DEFAULT_LOG;
    char datafilename[1024] = DEFAULT_DATA;
    char gridfilename[1024] = DEFAULT_GRID;
    char proffilename[1024] = DEFAULT_PROF;
    char vtkfilename[1024] = DEFAULT_VTK;
    int numberofthread = 1;
    
    {
        if(argc>1)strcpy(datafilename,argv[1]);
        if(argc>2)strcpy(gridfilename,argv[2]);
        if(argc>3)strcpy(proffilename,argv[3]);
        if(argc>4)strcpy(vtkfilename,argv[4]);
        if(argc>5)strcpy( logfilename,argv[5]);
    	if(argc>6)numberofthread=atoi(argv[6]);
    }
   // selectFreeGPU();

    log_open(logfilename);
    {
        time_t t=time(NULL);
        log_printf("start reading files at %s\n",ctime(&t));
    }
	{
		#ifdef _OPENMP
		omp_set_num_threads( numberofthread );
		#pragma omp parallel
		{
			if(omp_get_thread_num()==0){
				log_printf("omp_get_num_threads()=%d\n", omp_get_num_threads() );
			}
		}
		#endif
    }
    
    readDataFile(datafilename);
    readGridFile(gridfilename);
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    initializeWeight();
    initializeFluid();
    initializeWall();
    initializeDomain();

//	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
//	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
//	#pragma acc enter data create(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
//	#pragma acc enter data create(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
//	#pragma acc enter data create(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
//	#pragma acc enter data create(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
//	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	
	// data transfer from host to device
	#pragma acc update device(ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCount,Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc update device(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc update device(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc update device(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc update device(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	
	#pragma acc update device(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT])
	#pragma acc update device(AngularVelocity[0:ParticleCount][0:DIM],LambdaLames[0:ParticleCount],MuLames[0:ParticleCount],Young[0:ParticleCount])
	#pragma acc update device(Acceleration[0:ParticleCount][0:DIM])

	{
	
    calculateInitialNeighbor();
	calculateNeighbor();
	calculateDensityA();
	calculateGravityCenter();
	calculateDensityP();
    calculateLamesconstant();
    calculateNormalizer();
	//setInitialVelocityProfile();
	writeVtkFile("output.vtk");
		
		{
			time_t t=time(NULL);
			log_printf("start main roop at %s\n",ctime(&t));
		}
		int iStep=(int)(Time/Dt);
		cStart = clock();
		cFrom = cStart;
		while(Time < EndTime + 1.0e-5*Dt){
			
			if( Time + 1.0e-5*Dt >= OutputNext ){
				char filename[256];
				sprintf(filename,proffilename,iStep);
				writeProfFile(filename);
				log_printf("@ Prof Output Time : %e\n", Time );
				OutputNext += OutputInterval;
			}
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;
			
			#ifdef Turek_Hron
			setInitialVelocityProfile();
            #endif
			
			// wall calculation
			calculateWall();
			
			// periodic boundary calculation
			calculatePeriodicBoundary();
			
			// reset Force to calculate conservative interaction
			resetForce();
			resetAccel();
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			// calculate Neighbor
			//if(neighborCalculation()==1){
				calculateNeighbor();
			//}
			cTill = clock(); cNeigh += (cTill-cFrom); cFrom = cTill;
			
			// calculate density
			calculateDensityA();
			calculateGravityCenter();
			calculateDensityP();
			calculateDivergenceP();
			
			// calculate physical coefficient (viscosity, bulk modulus, bulk viscosity..)
			calculatePhysicalCoefficients();
			
			#ifdef FLUID
			// calculate pressure 
			calculatePressureP();
			
			// calculate P(s,rho) s:fixed
			calculatePressureA();
			
			// calculate diffuse interface force
			calculateDiffuseInterface();
			
			// calculate shear viscosity
			calculateViscosityV();
			#endif

            calculateGravity();
			
			calculateInterfaceForce();
		        
            
            // calculate intermidiate Velocity
            calculateAcceleration();    		
			
			//resetForce();
			 
			 // particle movement
        	calculateConvection();

			
			#ifdef STRUCTURE
			
			//resetForce();
   			 int substeps = (int)(Dt / Elastic_Dt + 0.5); // 安全のため四捨五入

  			  for (int substep = 0; substep < substeps; ++substep) {
     			   calculateElasticDeformationVector();  // 変形ベクトルの計算

      			  calculateStress();                    // 応力テンソルの計算

      			  calculateStressForce();              // 応力による力の計算
        
      			  updateElasticPosition();
  				 }

			#endif

        

			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			
			if( Time + 1.0e-5*Dt >= VtkOutputNext ){
				calculateVirialStressAtParticle();
				cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;

				char filename [256];
				sprintf(filename,vtkfilename,iStep);
				writeVtkFile(filename);
				log_printf("@ Vtk Output Time : %e\n", Time );
				VtkOutputNext += VtkOutputInterval;
				cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

			}
			
			Time += Dt;
			iStep++;
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
		}
	}
	cEnd = cTill;
	
	{
		time_t t=time(NULL);
		log_printf("end main roop at %s\n",ctime(&t));
		log_printf("neighbor search:         %lf [CPU sec]\n", (double)cNeigh/CLOCKS_PER_SEC);
		log_printf("explicit calculation:    %lf [CPU sec]\n", (double)cExplicit/CLOCKS_PER_SEC);
		log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
		log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
		log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cVirial+cOther)/CLOCKS_PER_SEC);
		log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
	}
	
	
	#pragma acc exit data delete(ParticleCount,ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc exit data delete(Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc exit data delete(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc exit data delete(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc exit data delete(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc exit data delete(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc exit data delete(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc exit data delete(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM])
	#pragma acc exit data delete(CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	#pragma acc exit data delete(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc exit data delete(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc exit data delete(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc exit data delete(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc exit data delete(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT],RotationMatrix[0:ParticleCount][0:DIM][0:DIM],quaternion[0:4])
	#pragma acc exit data delete(AngularVelocity[0:ParticleCount][0:DIM],LambdaLames[0:ParticleCount],MuLames[0:ParticleCount],Young[0:ParticleCount])
	#pragma acc exit data delete(Normalizer[0:ParticleCount][0:DIM][0:DIM],DeformGradient[0:ParticleCount][0:DIM][0:DIM],Strain[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc exit data delete(Stress[0:ParticleCount][0:DIM][0:DIM],Acceleration[0:ParticleCount][0:DIM],Conversion[0:ParticleCount][0:DIM][0:DIM],Original[0:ParticleCount])
	#pragma acc exit data delete(InitialNeighborCount[0:ParticleCount],InitialNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],InitialStructureNeighborCount[0:ParticleCount],InitialStructureNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])

	return 0;
	
}

static void readDataFile(char *filename)
{
    FILE * fp;
    char buf[1024];
    const int reading_global=0;
    int mode=reading_global;
    

 
    fp=fopen(filename,"r");
    mode=reading_global;
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)!=NULL){
            if(buf[0]=='#'){}
       else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
        else if(sscanf(buf," ElasticDt %lf",&Elastic_Dt)==1){mode=reading_global;}
       else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
       else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
       else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
        	// else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
       else if(sscanf(buf," Density %lf %lf %lf %lf %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3],&Density[4],&Density[5])==6){mode=reading_global;}
       else if(sscanf(buf," BulkModulus %lf %lf %lf %lf %lf %lf",&BulkModulus[0],&BulkModulus[1],&BulkModulus[2],&BulkModulus[3],&BulkModulus[4],&BulkModulus[5])==6){mode=reading_global;}
       else if(sscanf(buf," BulkViscosity %lf %lf %lf %lf %lf %lf",&BulkViscosity[0],&BulkViscosity[1],&BulkViscosity[2],&BulkViscosity[3],&BulkViscosity[4],&BulkViscosity[5])==6){mode=reading_global;}
       else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3],&ShearViscosity[4],&ShearViscosity[5])==6){mode=reading_global;}
       else if(sscanf(buf," SurfaceTension %lf %lf %lf %lf",&SurfaceTension[0],&SurfaceTension[1],&SurfaceTension[4],&SurfaceTension[5])==4){mode=reading_global;}
       else if(sscanf(buf," YoungModulus %lf %lf %lf %lf",&YoungModulus[2],&YoungModulus[3],&YoungModulus[4],&YoungModulus[5])==4){mode=reading_global;}
       else if(sscanf(buf," PoissonRatio %lf %lf %lf %lf ",&PoissonRatio[2],&PoissonRatio[3],&PoissonRatio[4],&PoissonRatio[5])==4){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type0) %lf %lf %lf %lf %lf %lf",&InteractionRatio[0][0],&InteractionRatio[0][1],&InteractionRatio[0][2],&InteractionRatio[0][3],&InteractionRatio[0][4],&InteractionRatio[0][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type1) %lf %lf %lf %lf %lf %lf",&InteractionRatio[1][0],&InteractionRatio[1][1],&InteractionRatio[1][2],&InteractionRatio[1][3],&InteractionRatio[1][4],&InteractionRatio[1][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type2) %lf %lf %lf %lf %lf %lf",&InteractionRatio[2][0],&InteractionRatio[2][1],&InteractionRatio[2][2],&InteractionRatio[2][3],&InteractionRatio[2][4],&InteractionRatio[2][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type3) %lf %lf %lf %lf %lf %lf",&InteractionRatio[3][0],&InteractionRatio[3][1],&InteractionRatio[3][2],&InteractionRatio[3][3],&InteractionRatio[3][4],&InteractionRatio[3][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type4) %lf %lf %lf %lf %lf %lf",&InteractionRatio[4][0],&InteractionRatio[4][1],&InteractionRatio[4][2],&InteractionRatio[4][3],&InteractionRatio[4][4],&InteractionRatio[4][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type5) %lf %lf %lf %lf %lf %lf",&InteractionRatio[5][0],&InteractionRatio[5][1],&InteractionRatio[5][2],&InteractionRatio[5][3],&InteractionRatio[5][4],&InteractionRatio[5][5])==6){mode=reading_global;}
       else if(sscanf(buf," Gravity %lf %lf %lf", &Gravity[0], &Gravity[1], &Gravity[2])==3){mode=reading_global;}
       else if(sscanf(buf," Wall6  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[4][0],  &WallCenter[4][1],  &WallCenter[4][2],  &WallVelocity[4][0],  &WallVelocity[4][1],  &WallVelocity[4][2],  &WallOmega[4][0],  &WallOmega[4][1],  &WallOmega[4][2])==9){mode=reading_global;}
       else if(sscanf(buf," Wall7  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[5][0],  &WallCenter[5][1],  &WallCenter[5][2],  &WallVelocity[5][0],  &WallVelocity[5][1],  &WallVelocity[5][2],  &WallOmega[5][0],  &WallOmega[5][1],  &WallOmega[5][2])==9){mode=reading_global;}
       else{
                log_printf("Invalid line in data file \"%s\"\n", buf);
            }
        }
    }
    fclose(fp);
	
	#pragma acc enter data create(ParticleCount,ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc enter data create(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc enter data create(Gravity[0:DIM])
//	#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM])
	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM])
	#pragma acc enter data create(WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc enter data create(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT])
	
}

static void readGridFile(char *filename)
{
    FILE *fp=fopen(filename,"r");
	char buf[1024];   
	
	
	try{
		
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%lf",&Time);
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%d  %lf  %lf %lf %lf  %lf %lf %lf",
			&ParticleCount,
			&ParticleSpacing,
			&DomainMin[0], &DomainMax[0],
			&DomainMin[1], &DomainMax[1],
			&DomainMin[2], &DomainMax[2]);
		#ifdef TWO_DIMENSIONAL
		ParticleVolume = ParticleSpacing*ParticleSpacing;
		#else
		ParticleVolume = ParticleSpacing*ParticleSpacing*ParticleSpacing;
		#endif
		
		Property = (int *)malloc(ParticleCount*sizeof(int));
        	Position = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        	InitialPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Velocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		DensityA = (double *)malloc(ParticleCount*sizeof(double));
		GravityCenter = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		PressureA = (double *)malloc(ParticleCount*sizeof(double));
		VolStrainP = (double *)malloc(ParticleCount*sizeof(double));
		DivergenceP = (double *)malloc(ParticleCount*sizeof(double));
		PressureP = (double *)malloc(ParticleCount*sizeof(double));
		VirialPressureAtParticle = (double *)malloc(ParticleCount*sizeof(double));
		VirialStressAtParticle = (double (*) [DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		Mass = (double (*))malloc(ParticleCount*sizeof(double));
		Force = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Mu = (double (*))malloc(ParticleCount*sizeof(double));
		Lambda = (double (*))malloc(ParticleCount*sizeof(double));
		Kappa = (double (*))malloc(ParticleCount*sizeof(double));
		Young = (double (*))malloc(ParticleCount*sizeof(double));
		
		#pragma acc enter data create(Property[0:ParticleCount])               attach(Property)
		#pragma acc enter data create(Position[0:ParticleCount][0:DIM])        attach(Position)
		#pragma acc enter data create(InitialPosition[0:ParticleCount][0:DIM]) attach(InitialPosition)
		#pragma acc enter data create(Velocity[0:ParticleCount][0:DIM])        attach(Velocity)
		#pragma acc enter data create(DensityA[0:ParticleCount])               attach(DensityA)
		#pragma acc enter data create(GravityCenter[0:ParticleCount][0:DIM])   attach(GravityCenter)
		#pragma acc enter data create(PressureA[0:ParticleCount])              attach(PressureA)
		#pragma acc enter data create(VolStrainP[0:ParticleCount])             attach(VolStrainP)
		#pragma acc enter data create(DivergenceP[0:ParticleCount])            attach(DivergenceP)
		#pragma acc enter data create(PressureP[0:ParticleCount])              attach(PressureP)
		#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount])               attach(VirialPressureAtParticle)
		#pragma acc enter data create(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])   attach(VirialStressAtParticle)
		#pragma acc enter data create(Mass[0:ParticleCount])                attach(Mass)
		#pragma acc enter data create(Force[0:ParticleCount][0:DIM])        attach(Force)
		#pragma acc enter data create(Mu[0:ParticleCount])                  attach(Mu)
		#pragma acc enter data create(Lambda[0:ParticleCount])              attach(Lambda)
		#pragma acc enter data create(Kappa[0:ParticleCount])               attach(Kappa)
		#pragma acc enter data create(Young[0:ParticleCount])               attach(Young)
        
    
      		LambdaLames = (double (*))malloc(ParticleCount*sizeof(double));
       		MuLames = (double (*))malloc(ParticleCount*sizeof(double));
       		quaternion = (double (*)[4])malloc(ParticleCount*sizeof(double [4]));
       		RotationMatrix = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        	Normalizer = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        	DeformGradient = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        	Strain = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        	Stress = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        	AngularVelocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        	Acceleration = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        	Conversion = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        	Original = (int *)malloc(ParticleCount*sizeof(int));
        	
        	#pragma acc enter data create(LambdaLames[0:ParticleCount]) attach(LambdaLames)
		#pragma acc enter data create(MuLames[0:ParticleCount]) attach(MuLames)
		#pragma acc enter data create(quaternion[0:ParticleCount][0:4]) attach(quaternion)
		#pragma acc enter data create(RotationMatrix[0:ParticleCount][0:DIM][0:DIM]) attach(RotationMatrix)
		#pragma acc enter data create(Normalizer[0:ParticleCount][0:DIM][0:DIM]) attach(Normalizer)
		#pragma acc enter data create(DeformGradient[0:ParticleCount][0:DIM][0:DIM]) attach(DeformGradient)
		#pragma acc enter data create(Strain[0:ParticleCount][0:DIM][0:DIM]) attach(Strain)
		#pragma acc enter data create(Stress[0:ParticleCount][0:DIM][0:DIM]) attach(Stress)
		#pragma acc enter data create(Acceleration[0:ParticleCount][0:DIM]) attach(Acceleration)
		#pragma acc enter data create(AngularVelocity[0:ParticleCount][0:DIM]) attach(AngularVelocity)
		#pragma acc enter data create(Conversion[0:ParticleCount][0:DIM][0:DIM]) attach(Conversion)
		#pragma acc enter data create(Original[0:ParticleCount]) attach(Original)
		
        	
		NeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
		Neighbor       = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
        	InitialNeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
        	InitialNeighbor       = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
        	InitialStructureNeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
        	InitialStructureNeighbor       = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
		NeighborCalculatedPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		
		#pragma acc enter data create(NeighborCount[0:ParticleCount]) attach(NeighborCount)
		#pragma acc enter data create(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(Neighbor)
		#pragma acc enter data create(InitialNeighborCount[0:ParticleCount]) attach(InitialNeighborCount)
		#pragma acc enter data create(InitialNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(InitialNeighbor)
		#pragma acc enter data create(InitialStructureNeighborCount[0:ParticleCount]) attach(InitialStructureNeighborCount)
		#pragma acc enter data create(InitialStructureNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(InitialStructureNeighbor)
		#pragma acc enter data create(NeighborCalculatedPosition[0:ParticleCount][0:DIM]) attach(NeighborCalculatedPosition)
		
	//	double (*q)[DIM] = Position;
		double (*v)[DIM] = Velocity;
		
		for(int iP=0;iP<ParticleCount;++iP){
			if(fgets(buf,sizeof(buf),fp)==NULL)break;
			sscanf(buf,"%d  %lf %lf %lf %lf %lf %lf  %lf %lf %lf",
				&Property[iP],
				&Position[iP][0],&Position[iP][1],&Position[iP][2],
                &InitialPosition[iP][0],&InitialPosition[iP][1],&InitialPosition[iP][2],
				&v[iP][0],&v[iP][1],&v[iP][2]
			);
		}
	}catch(...){};
	
	fclose(fp);
	
    FluidParticleBegin = -1;
    FluidParticleEnd = -1;
    StructureParticleBegin = -1;
    StructureParticleEnd = -1;
    WallParticleBegin = -1;
    WallParticleEnd = -1;

    for (int iP = 0; iP < ParticleCount; ++iP) {
        int prop = Property[iP];

        if (FLUID_BEGIN <= prop && prop < FLUID_END) {
            if (FluidParticleBegin == -1) FluidParticleBegin = iP;
            FluidParticleEnd = iP + 1;
        } else if (STRUCTURE_BEGIN <= prop && prop < STRUCTURE_END) {
            if (StructureParticleBegin == -1) StructureParticleBegin = iP;
            StructureParticleEnd = iP + 1;
        } else if (WALL_BEGIN <= prop && prop < WALL_END) {
            if (WallParticleBegin == -1) WallParticleBegin = iP;
            WallParticleEnd = iP + 1;
        }
    }

    if (FluidParticleBegin != -1)
        printf("Fluid Particles: %d\n", FluidParticleEnd - FluidParticleBegin);
    else
        printf("Fluid Particles: 0\n");

    if (StructureParticleBegin != -1)
        printf("Structure Particles: %d\n", StructureParticleEnd - StructureParticleBegin);
    else
        printf("Structure Particles: 0\n");

    if (WallParticleBegin != -1)
        printf("Wall Particles: %d\n", WallParticleEnd - WallParticleBegin);
    else
        printf("Wall Particles: 0\n");

	#pragma acc update device(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(Property[0:ParticleCount][0:DIM])
	#pragma acc update device(Position[0:ParticleCount][0:DIM])
	#pragma acc update device(InitialPosition[0:ParticleCount][0:DIM])
	#pragma acc update device(Velocity[0:ParticleCount][0:DIM])
//	#pragma acc update device(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)


	
}

static void writeProfFile(char *filename)
{
    FILE *fp=fopen(filename,"w");

    fprintf(fp,"%e\n",Time);
    fprintf(fp,"%d %e %e %e %e %e %e %e\n",
            ParticleCount,
            ParticleSpacing,
            DomainMin[0], DomainMax[0],
            DomainMin[1], DomainMax[1],
            DomainMin[2], DomainMax[2]);

 //   const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    for(int iP=0;iP<ParticleCount;++iP){
            fprintf(fp,"%d %e %e %e %e %e %e  %e %e %e\n",
                    Property[iP],
                    Position[iP][0], Position[iP][1], Position[iP][2],
                    InitialPosition[iP][0],InitialPosition[iP][1],InitialPosition[iP][2],
                    v[iP][0], v[iP][1], v[iP][2]
            );
    }
    fflush(fp);
    fclose(fp);
}

static void writeVtkFile(char *filename)
{
	// update parameters to be output
	#pragma acc update host(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount],Mass[0:ParticleCount],Young[0:ParticleCount])
	#pragma acc update host(NeighborCount[0:ParticleCount],InitialStructureNeighborCount[0:ParticleCount],Force[0:ParticleCount][0:DIM],Normalizer[0:ParticleCount][0:DIM][0:DIM],DeformGradient[0:ParticleCount][0:DIM][0:DIM],CellParticle[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc update host(Stress[0:ParticleCount][0:DIM][0:DIM],Strain[0:ParticleCount][0:DIM][0:DIM],Acceleration[0:ParticleCount][0:DIM],LambdaLames[0:ParticleCount],MuLames[0:ParticleCount])

    const double (*v)[DIM] = Velocity;

    FILE *fp=fopen(filename, "w");

    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Unstructured Grid Example\n");
    fprintf(fp, "ASCII\n");

    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)Position[iP][0], (float)Position[iP][1], (float)Position[iP][2]);
    }
    fprintf(fp, "CELLS %d %d\n", ParticleCount, 2*ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 %d ",iP);
    }
    fprintf(fp, "\n");
    fprintf(fp, "CELL_TYPES %d\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 ");
    }
    fprintf(fp, "\n");

    fprintf(fp, "\n");

    fprintf(fp, "POINT_DATA %d\n", ParticleCount);
    fprintf(fp, "SCALARS label float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", Property[iP]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "\n");
    fprintf(fp, "VECTORS displacement float\n");
    for(int iP=0;iP<ParticleCount;++iP){
        const double displacement[DIM]={Position[iP][0]-InitialPosition[iP][0],Position[iP][1]-InitialPosition[iP][1],Position[iP][2]-InitialPosition[iP][2]};
        fprintf(fp, "%e %e %e\n", (float)displacement[0], (float)displacement[1], (float)displacement[2]);
    }
    for (int iD=0;iD<DIM;iD++){
       for(int jD=0;jD<DIM;jD++){
    fprintf(fp, "\n"); fprintf(fp," SCALARS stress%d%d float \n", iD, jD);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Stress[iP][iD][jD]);
    }
    }
    }
        for (int iD=0;iD<DIM;iD++){
       for(int jD=0;jD<DIM;jD++){
    fprintf(fp, "\n"); fprintf(fp," SCALARS strain%d%d float \n", iD, jD);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Strain[iP][iD][jD]);
    }
    }
    }


/*
    fprintf(fp, "\n");
    for(int iD=0;iD<DIM;++iD){
        for(int jD=0;jD<DIM;++jD){
            fprintf(fp, "\n");    fprintf(fp, "SCALARS deform[%d][%d] float 1\n",iD,jD);
            fprintf(fp, "LOOKUP_TABLE default\n");
            for(int iP=0;iP<ParticleCount;++iP){
                fprintf(fp, "%e\n", (float)DeformGradient[iP][iD][jD]);
            }
        }
    }
        */
    fprintf(fp, "VECTORS velocity float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
    }
    fprintf(fp, "\n");

        fprintf(fp, "VECTORS accel float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)Acceleration[iP][0], (float)Acceleration[iP][1], (float)Acceleration[iP][2]);
    }
    fprintf(fp, "\n");

//    fprintf(fp, "VECTORS GravityCenter float\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e %e %e\n", (float)GravityCenter[iP][0], (float)GravityCenter[iP][1], (float)GravityCenter[iP][2]);
//    }
//    fprintf(fp, "\n");
/*
    fprintf(fp, "VECTORS force float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
    }
    fprintf(fp, "\n");
*/

    
    fprintf(fp, "SCALARS Initialneighbor float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", InitialStructureNeighborCount[iP]);
    }
    fprintf(fp, "SCALARS neighbor float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", NeighborCount[iP]);
    }
//    fprintf(fp, "SCALARS DensityA float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)DensityA[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS PressureA float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)PressureA[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS VolStrainP float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)VolStrainP[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS DivergenceP float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)DivergenceP[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS PressureP float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)PressureP[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS VirialPressureAtParticle float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)VirialPressureAtParticle[iP]); // trivial operation is done for 
 //   }
//	fprintf(fp, "\n");
//	for(int iD=0;iD<DIM-1;++iD){
//		for(int jD=0;jD<DIM-1;++jD){
//			fprintf(fp, "SCALARS VirialStressAtParticle[%d][%d] float 1\n",iD,jD);
//			fprintf(fp, "LOOKUP_TABLE default\n");
//			for(int iP=0;iP<ParticleCount;++iP){
//				fprintf(fp, "%e\n", (float)VirialStressAtParticle[iP][iD][jD]); // trivial operation is done for 
//			}
//			fprintf(fp, "\n");    
//		}
//	}
//  
//    fprintf(fp, "SCALARS Mu float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)Mu[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS Lambda float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)Lambda[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS Kappa float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)Kappa[iP]);
//    }
//    fprintf(fp, "\n");
 //   fprintf(fp, "SCALARS neighbor float 1\n");
 //   fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%d\n", NeighborCount[iP]);
//    }

    fprintf(fp, "VECTORS velocity float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
    }
    fprintf(fp, "\n");

//    fprintf(fp, "VECTORS GravityCenter float\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e %e %e\n", (float)GravityCenter[iP][0], (float)GravityCenter[iP][1], (float)GravityCenter[iP][2]);
//    }
//    fprintf(fp, "\n");
    fprintf(fp, "VECTORS force float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
    }
    fprintf(fp, "\n");

	
    fflush(fp);
    fclose(fp);
}

static void initializeWeight()
{
	RadiusRatioG = RadiusRatioA;
	
	RadiusA = RadiusRatioA*ParticleSpacing;
	RadiusG = RadiusRatioG*ParticleSpacing;
	RadiusP = RadiusRatioP*ParticleSpacing;
	RadiusV = RadiusRatioV*ParticleSpacing;
	
	
#ifdef TWO_DIMENSIONAL
		Swa = 1.0/2.0 * 2.0/15.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swg = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swp = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swv = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		R2g = 1.0/2.0 * 1.0/30.0* M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing /Swg;
#else	//code for three dimensional
		Swa = 1.0/3.0 * 1.0/5.0*M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swg = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swp = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swv = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		R2g = 1.0/3.0 * 4.0/105.0*M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing/ParticleSpacing /Swg;
#endif
	
	
	    {// N0a
        const double radius_ratio = RadiusA/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusA*RadiusA){
                        const double rij = sqrt(rij2);
                        const double wij = wa(rij,RadiusA);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusA*RadiusA){
                            const double rij = sqrt(rij2);
                            const double wij = wa(rij,RadiusA);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0a = sum;
        log_printf("N0a = %e, count=%d\n", N0a, count);
    }	

    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusP*RadiusP){
                        const double rij = sqrt(rij2);
                        const double wij = wp(rij,RadiusP);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusP*RadiusP){
                            const double rij = sqrt(rij2);
                            const double wij = wp(rij,RadiusP);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }
	
	#pragma acc update device(RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	

}


static void initializeFluid()
{
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Kappa[iP]=BulkModulus[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Lambda[iP]=BulkViscosity[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Young[iP]=YoungModulus[Property[iP]];
	}
	#ifdef TWO_DIMENSIONAL
	CofK = 0.350778153;
	double integN=0.024679383;
	double integX=0.226126699;
	#else 
	CofK = 0.326976006;
	double integN=0.021425779;
	double integX=0.233977488;
	#endif
	
	for(int iT=0;iT<TYPE_COUNT;++iT){
		CofA[iT]=SurfaceTension[iT] / ((RadiusG/ParticleSpacing)*(integN+CofK*CofK*integX));
	}
    
    const double unit[DIM][DIM] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    for(int iP=0;iP<ParticleCount;++iP){
        for(int iD=0;iD<DIM;++iD){
            for(int jD=0;jD<DIM;++jD){
                Original[iP] = iP;
                Conversion[iP][iD][jD] = unit[iD][jD];
            }
        }
    }
    for(int iP=0;iP<ParticleCount;++iP){
        quaternion[iP][0] = 1.0;
        quaternion[iP][1] = 0.0;
        quaternion[iP][2] = 0.0;
        quaternion[iP][3] = 0.0;
    }
	
	#pragma acc update device(Mass[0:ParticleCount])
	#pragma acc update device(Kappa[0:ParticleCount])
	#pragma acc update device(Lambda[0:ParticleCount])
	#pragma acc update device(Mu[0:ParticleCount])
	#pragma acc update device(CofK,CofA[0:TYPE_COUNT])
	#pragma acc update device(Original[0:ParticleCount])
	#pragma acc update device(Conversion[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc update device(quaternion[0:ParticleCount][0:DIM])
}



static void initializeWall()
{
	
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		
		double theta;
		double normal[DIM]={0.0,0.0,0.0};
		double q[DIM+1];
		double t[DIM];
		double (&R)[DIM][DIM]=WallRotation[iProp];
		
		theta = abs(WallOmega[iProp][0]*WallOmega[iProp][0]+WallOmega[iProp][1]*WallOmega[iProp][1]+WallOmega[iProp][2]*WallOmega[iProp][2]);
		if(theta!=0.0){
			for(int iD=0;iD<DIM;++iD){
				normal[iD]=WallOmega[iProp][iD]/theta;
			}
		}
		q[0]=normal[0]*sin(theta*Dt/2.0);
		q[1]=normal[1]*sin(theta*Dt/2.0);
		q[2]=normal[2]*sin(theta*Dt/2.0);
		q[3]=cos(theta*Dt/2.0);
		t[0]=WallVelocity[iProp][0]*Dt;
		t[1]=WallVelocity[iProp][1]*Dt;
		t[2]=WallVelocity[iProp][2]*Dt;
		
		R[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[0][1] = 2.0*(q[0]*q[1]-q[2]*q[3]);
		R[0][2] = 2.0*(q[0]*q[2]+q[1]*q[3]);
		
		R[1][0] = 2.0*(q[0]*q[1]+q[2]*q[3]);
		R[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[1][2] = 2.0*(q[1]*q[2]-q[0]*q[3]);
		
		R[2][0] = 2.0*(q[0]*q[2]-q[1]*q[3]);
		R[2][1] = 2.0*(q[1]*q[2]+q[0]*q[3]);
		R[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
		
	}
	#pragma acc update device(WallRotation[0:WALL_END][0:DIM][0:DIM])
}

static void initializeDomain( void )
{
	CellWidth = ParticleSpacing;
	
	double cellCount[DIM];
	
	cellCount[0] = round((DomainMax[0] - DomainMin[0])/CellWidth);
	cellCount[1] = round((DomainMax[1] - DomainMin[1])/CellWidth);
	#ifdef TWO_DIMENSIONAL
	cellCount[2] = 1;
	#else
	cellCount[2] = round((DomainMax[2] - DomainMin[2])/CellWidth);
	#endif
	
	CellCount[0] = (int)cellCount[0];
	CellCount[1] = (int)cellCount[1];
	CellCount[2] = (int)cellCount[2];
	CellCounts   = cellCount[0]*cellCount[1]*cellCount[2];
	
	if(cellCount[0]!=(double)CellCount[0] || cellCount[1]!=(double)CellCount[1] ||cellCount[2]!=(double)CellCount[2]){
		fprintf(stderr,"DomainWidth/CellWidth is not integer\n");
		DomainMax[0] = DomainMin[0] + CellWidth*(double)CellCount[0];
		DomainMax[1] = DomainMin[1] + CellWidth*(double)CellCount[1];
		DomainMax[2] = DomainMin[2] + CellWidth*(double)CellCount[2];
		fprintf(stderr,"Changing the Domain Max to (%e,%e,%e)\n", DomainMax[0], DomainMax[1], DomainMax[2]);
	}
	DomainWidth[0] = DomainMax[0] - DomainMin[0];
	DomainWidth[1] = DomainMax[1] - DomainMin[1];
	DomainWidth[2] = DomainMax[2] - DomainMin[2];
	
	CellParticleBegin = (int *)malloc( CellCounts * sizeof(int) );
	CellParticleEnd   = (int *)malloc( CellCounts * sizeof(int) );
	#pragma acc enter data create(CellParticleBegin[0:CellCounts]) attach(CellParticleBegin)
	#pragma acc enter data create(CellParticleEnd  [0:CellCounts]) attach(CellParticleEnd)
	
	
	// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
	ParticleCountPower=0;  
	while((ParticleCount>>ParticleCountPower)!=0){
		++ParticleCountPower;
	}
	PowerParticleCount = (1<<ParticleCountPower);
	fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
	CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	#pragma acc enter data create(CellIndex   [0:PowerParticleCount]) attach(CellIndex)
	#pragma acc enter data create(CellParticle[0:PowerParticleCount]) attach(CellParticle)
	
	MaxRadius = ((RadiusA>MaxRadius) ? RadiusA : MaxRadius);
	MaxRadius = ((RadiusG>MaxRadius) ? RadiusG : MaxRadius);
	MaxRadius = ((RadiusP>MaxRadius) ? RadiusP : MaxRadius);
	MaxRadius = ((RadiusV>MaxRadius) ? RadiusV : MaxRadius);
	
	#pragma acc update device(CellWidth,CellCount[0:DIM],CellCounts)
	#pragma acc update device(DomainMax[0:DIM],DomainMin[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCountPower,PowerParticleCount)
	#pragma acc update device(MaxRadius)
}


static int neighborCalculation( void ){
	double maxShift2=0.0;
	#pragma acc parallel loop reduction (max:maxShift2)
	#pragma omp parallel for reduction (max:maxShift2)
	for(int iP=0;iP<ParticleCount;++iP){
		 double disp[DIM];
         #pragma acc loop seq
         for(int iD=0;iD<DIM;++iD){
            disp[iD] = Mod(Position[iP][iD] - NeighborCalculatedPosition[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
         }
		const double shift2 = disp[0]*disp[0]+disp[1]*disp[1]+disp[2]*disp[2];
		if(shift2>maxShift2){
			maxShift2=shift2;
		}
	}
	
	if(maxShift2>0.5*MARGIN*0.5*MARGIN){
		return 1;
	}
	else{
		return 0;
	}
}


static void calculateInitialNeighbor( void )
{
    
    // calculate CellIndex[iP]
    #pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount],InitialPosition[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0; iP<(1<<ParticleCountPower); ++iP){
        if(iP<ParticleCount){
            const int iCX=((int)floor((InitialPosition[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
            const int iCY=((int)floor((InitialPosition[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
            const int iCZ=((int)floor((InitialPosition[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
            CellIndex[iP]=CellId(iCX,iCY,iCZ);
            CellParticle[iP]=iP;
        }
        else{
            CellIndex[ iP ]    = CellCount[0]*CellCount[1]*CellCount[2];
            CellParticle[ iP ] = ParticleCount;
        }
    }
    
    {
        // sort with CellIndex
        // https://edom18.hateblo.jp/entry/2020/09/21/150416
        for(int iMain=0;iMain<ParticleCountPower;++iMain){
            for(int iSub=0;iSub<=iMain;++iSub){
                
                int dist = (1<< (iMain-iSub));
                
                #pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
                #pragma acc loop independent
                #pragma omp parallel for
                for(int iP=0;iP<(1<<ParticleCountPower);++iP){
                    bool up = ((iP >> iMain) & 2) == 0;
                    
                    if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
                        int tmpCellIndex    = CellIndex[ iP ];
                        int tmpCellParticle = CellParticle[ iP ];
                        CellIndex[ iP ]     = CellIndex[ iP | dist ];
                        CellParticle[ iP ]  = CellParticle[ iP | dist ];
                        CellIndex[ iP | dist ]    = tmpCellIndex;
                        CellParticle[ iP | dist ] = tmpCellParticle;
                    }
                }
            }
        }
    }
    
    // search for CellParticleBegin[iC]
    #pragma acc kernels present(CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
    {
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iC=0;iC<CellCounts;++iC){
            CellParticleBegin[iC]=0;
            CellParticleEnd[iC]=0;
        }
        
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0; iP<ParticleCount; ++iP){
            if( CellIndex[iP]<CellIndex[iP+1] ){
                CellParticleEnd[ CellIndex[iP] ]   =iP+1;
                CellParticleBegin[ CellIndex[iP+1] ]=iP+1;
            }
        }
    }
    
    // calculate neighbor
    #pragma acc kernels present(InitialStructureNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],InitialStructureNeighborCount[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
        InitialStructureNeighborCount[iP]=0;
        for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
            InitialStructureNeighbor[iP][iN]=-1;
        }
    }

#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts],InitialPosition[0:ParticleCount][0:DIM],Property[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
    const int range = (int)(ceil((MaxRadius+MARGIN)/CellWidth));
    const int iCX=((int)floor((InitialPosition[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
    const int iCY=((int)floor((InitialPosition[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
    const int iCZ=
#ifdef TWO_DIMENSIONAL
    0;
#else
    ((int)floor((InitialPosition[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
#endif

    #pragma acc loop seq
    for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
        #pragma acc loop seq
        for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
            #ifdef TWO_DIMENSIONAL
           const int jCZ=0;
            {
                const int jC = CellId(jCX, jCY, jCZ);
                #pragma acc loop seq
                for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                    const int jP = CellParticle[jCP];
                    double qij[3]; // Always use 3 elements
                    for(int iD=0;iD<2;++iD){
                        qij[iD] = Mod(InitialPosition[jP][iD] - InitialPosition[iP][iD] + 0.5*DomainWidth[iD], DomainWidth[iD]) - 0.5*DomainWidth[iD];
                    }
                    qij[2] = 0.0; // fake Z layer

                    const double qij2 = qij[0]*qij[0] + qij[1]*qij[1] + qij[2]*qij[2];
                    if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN) && Property[jP] >= STRUCTURE_BEGIN && Property[jP] < STRUCTURE_END){
                        if(InitialStructureNeighborCount[iP] >= MAX_NEIGHBOR_COUNT){
                            InitialStructureNeighborCount[iP]++;
                        }
                        else if(iP != jP){
                            InitialStructureNeighbor[iP][InitialStructureNeighborCount[iP]] = jP;
                            InitialStructureNeighborCount[iP]++;
                        }
                    }
                }
            }
            #else
            for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
                const int jC = CellId(jCX, jCY, jCZ);
                #pragma acc loop seq
                for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                    const int jP = CellParticle[jCP];
                    double qij[3];
                    for(int iD=0;iD<3;++iD){
                        qij[iD] = Mod(InitialPosition[jP][iD] - InitialPosition[iP][iD] + 0.5*DomainWidth[iD], DomainWidth[iD]) - 0.5*DomainWidth[iD];
                    }
                    const double qij2 = qij[0]*qij[0] + qij[1]*qij[1] + qij[2]*qij[2];
                    if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN) && Property[jP] >= STRUCTURE_BEGIN && Property[jP] < STRUCTURE_END){
                        if(InitialStructureNeighborCount[iP] >= MAX_NEIGHBOR_COUNT){
                            InitialStructureNeighborCount[iP]++;
                        }
                        else if(iP != jP){
                            InitialStructureNeighbor[iP][InitialStructureNeighborCount[iP]] = jP;
                            InitialStructureNeighborCount[iP]++;
                        }
                    }
                }
            }
            #endif
        }
    }
}


    
    #pragma acc kernels present(NeighborCalculatedPosition[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            NeighborCalculatedPosition[iP][iD]=InitialPosition[iP][iD];
        }
    }
    
}


 
static void calculateNeighbor( void )
{
	
	// calculate CellIndex[iP]
	#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0; iP<(1<<ParticleCountPower); ++iP){
		if(iP<ParticleCount){
			const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
			const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
			const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
			CellIndex[iP]=CellId(iCX,iCY,iCZ);
			CellParticle[iP]=iP;
		}
		else{
			CellIndex[ iP ]    = CellCount[0]*CellCount[1]*CellCount[2];
			CellParticle[ iP ] = ParticleCount;
		}
	}
	
	{
		// sort with CellIndex
		// https://edom18.hateblo.jp/entry/2020/09/21/150416
		for(int iMain=0;iMain<ParticleCountPower;++iMain){
			for(int iSub=0;iSub<=iMain;++iSub){
				
				int dist = (1<< (iMain-iSub));
				
				#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
				#pragma acc loop independent
				#pragma omp parallel for
				for(int iP=0;iP<(1<<ParticleCountPower);++iP){
					bool up = ((iP >> iMain) & 2) == 0;
					
					if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
						int tmpCellIndex    = CellIndex[ iP ];
						int tmpCellParticle = CellParticle[ iP ];
						CellIndex[ iP ]     = CellIndex[ iP | dist ];
						CellParticle[ iP ]  = CellParticle[ iP | dist ];
						CellIndex[ iP | dist ]    = tmpCellIndex;
						CellParticle[ iP | dist ] = tmpCellParticle;
					}
				}
			}
		}
	}
	
	// search for CellParticleBegin[iC]
	#pragma acc kernels present(CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	{
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iC=0;iC<CellCounts;++iC){
			CellParticleBegin[iC]=0;
			CellParticleEnd[iC]=0;
		}
		
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0; iP<ParticleCount; ++iP){
			if( CellIndex[iP]<CellIndex[iP+1] ){
				CellParticleEnd[ CellIndex[iP] ]   =iP+1;
				CellParticleBegin[ CellIndex[iP+1] ]=iP+1;
			}
		}
	}
    
    // calculate neighbor
	#pragma acc kernels present(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCount[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        NeighborCount[iP]=0;
    	for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
    		Neighbor[iP][iN]=-1;
    	}
    }
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const int range = (int)(ceil((MaxRadius+MARGIN)/CellWidth));
    	const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
    	const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
    	const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];

#ifdef TWO_DIMENSIONAL
    	#pragma acc loop seq
        for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
        	#pragma acc loop seq
            for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
                const int jCZ=0;
                const int jC=CellId(jCX,jCY,jCZ);
            	#pragma acc loop seq
                for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                    const int jP=CellParticle[jCP];
                    double qij[DIM];
                	#pragma acc loop seq
                    for(int iD=0;iD<DIM;++iD){
                        qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                    }
                    const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                    if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
                        if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        	NeighborCount[iP]++;
                        }
                        else if(iP!=jP){
                            Neighbor[iP][NeighborCount[iP]] = jP;
                            NeighborCount[iP]++;
                        }
                    }
                }
            }
        }

    	    	
#else // TWO_DIMENSIONAL
    	#pragma acc loop seq
        for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
        	#pragma acc loop seq
        	for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
        		#pragma acc loop seq
                for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
                    const int jC=CellId(jCX,jCY,jCZ);
                	#pragma acc loop seq
                    for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                        const int jP=CellParticle[jCP];
                        double qij[DIM];
                    	#pragma acc loop seq
                        for(int iD=0;iD<DIM;++iD){
                            qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                        }
                        const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                        if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
                            if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        		NeighborCount[iP]++;
                        	}
                        	else if(iP!=jP){
                            	Neighbor[iP][NeighborCount[iP]] = jP;
                            	NeighborCount[iP]++;
                        	}
                        }
                    }
                }
            }
        }
#endif // TWO_DIMENSIONAL
    }
	
	#pragma acc kernels present(NeighborCalculatedPosition[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			NeighborCalculatedPosition[iP][iD]=Position[iP][iD];
		}
	}
	
}




/*

// Function to select a free GPU and ensure simulation runs exclusively on it
static void selectFreeGPU() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n");
        return;
    }

    int selectedDevice = -1;
    size_t maxFreeMem = 0;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Unable to get properties for device %d.\n", device);
            continue;
        }

        size_t freeMem = 0, totalMem = 0;
        cudaSetDevice(device);
        err = cudaMemGetInfo(&freeMem, &totalMem);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Unable to get memory info for device %d.\n", device);
            continue;
        }

        printf("Device %d: %s, Free Memory: %.2f MB, Total Memory: %.2f MB\n",
               device, prop.name, freeMem / (1024.0 * 1024.0), totalMem / (1024.0 * 1024.0));

        // Update selected device if this one has more free memory
        if (freeMem > maxFreeMem) {
            maxFreeMem = freeMem;
            selectedDevice = device;
        }
    }

    if (selectedDevice != -1) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, selectedDevice);
        printf("Selecting GPU %d (%s) with %.2f MB free memory.\n", 
               selectedDevice, prop.name, maxFreeMem / (1024.0 * 1024.0));
        acc_set_device_num(selectedDevice, acc_device_nvidia); // Set for OpenACC
        cudaSetDevice(selectedDevice);                         // Set for CUDA

        // Ensure the GPU is exclusively used
        err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Unable to set exclusive GPU mode for device %d.\n", selectedDevice);
        }
    } else {
        printf("No suitable GPU found with sufficient memory. Using default GPU.\n");
    }
}

*/



static void calculateConvection()
{
#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){

        Acceleration[iP][0] += Force[iP][0]/Mass[iP];  //For AI data//
        Acceleration[iP][1] += Force[iP][1]/Mass[iP]; //For AI data//
        Acceleration[iP][2] += Force[iP][2]/Mass[iP]; //For AI data// 

        Position[iP][0] += Velocity[iP][0]*Dt;
        Position[iP][1] += Velocity[iP][1]*Dt;
        Position[iP][2] += Velocity[iP][2]*Dt;
    }
}


static void updateElasticPosition()
{

#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
      
      #ifdef Bar_Module 
        if (InitialPosition[iP][0]<0.001){
            
            Position[iP][0] = InitialPosition[iP][0];
            Position[iP][1] = InitialPosition[iP][1];
            Position[iP][2] = InitialPosition[iP][2];
            
             Velocity[iP][0]=0.0;
             Velocity[iP][1]=0.0;
             Velocity[iP][2]=0.0;
             Force[iP][0]=0.0;
             Force[iP][1]=0.0;
             Force[iP][2]=0.0;
            
        }
        else{
        
            Velocity[iP][0] += Acceleration[iP][0]*Elastic_Dt;
            Velocity[iP][1] += Acceleration[iP][1]*Elastic_Dt;
            Velocity[iP][2] += Acceleration[iP][2]*Elastic_Dt;
            
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
        }
        #endif
        #ifdef Turek_Hron
        if (InitialPosition[iP][0]<0.205){
            
            Position[iP][0] = InitialPosition[iP][0];
            Position[iP][1] = InitialPosition[iP][1];
            Position[iP][2] = InitialPosition[iP][2];
            
             Velocity[iP][0]=0.0;
             Velocity[iP][1]=0.0;
             Velocity[iP][2]=0.0;
            
        }
        else{
        
            Velocity[iP][0] += Acceleration[iP][0]*Elastic_Dt;
            Velocity[iP][1] += Acceleration[iP][1]*Elastic_Dt;
            Velocity[iP][2] += Acceleration[iP][2]*Elastic_Dt;
            
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
        }
        #endif
        #ifdef DAM_Module 
        if (InitialPosition[iP][1]<0.002){
            
            Position[iP][0] = InitialPosition[iP][0];
            Position[iP][1] = InitialPosition[iP][1];
            Position[iP][2] = InitialPosition[iP][2];
            
             Velocity[iP][0]=0.0;
             Velocity[iP][1]=0.0;
             Velocity[iP][2]=0.0;
             Force[iP][0]=0.0;
             Force[iP][1]=0.0;
             Force[iP][2]=0.0;
            
        }
        else{
        
            Velocity[iP][0] += Acceleration[iP][0]*Elastic_Dt;
            Velocity[iP][1] += Acceleration[iP][1]*Elastic_Dt;
            Velocity[iP][2] += Acceleration[iP][2]*Elastic_Dt;
            
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
        }
        #endif
       #ifdef Rolling1 
        if (InitialPosition[iP][1]<0.003){
            
            Position[iP][0] = InitialPosition[iP][0];
            Position[iP][1] = InitialPosition[iP][1];
            Position[iP][2] = InitialPosition[iP][2];
            
             Velocity[iP][0]=0.0;
             Velocity[iP][1]=0.0;
             Velocity[iP][2]=0.0;
             Force[iP][0]=0.0;
             Force[iP][1]=0.0;
             Force[iP][2]=0.0;
            
        }
        else{
        
            Velocity[iP][0] += Acceleration[iP][0]*Elastic_Dt;
            Velocity[iP][1] += Acceleration[iP][1]*Elastic_Dt;
            Velocity[iP][2] += Acceleration[iP][2]*Elastic_Dt;
            
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
        }
        #endif
        #ifdef Hydroelastic
        if (InitialPosition[iP][0]<0.01 || InitialPosition[iP][0]>1.99){
            
            Position[iP][0] = InitialPosition[iP][0];
            Position[iP][1] = InitialPosition[iP][1];
            Position[iP][2] = InitialPosition[iP][2];
            
             Velocity[iP][0]=0.0;
             Velocity[iP][1]=0.0;
             Velocity[iP][2]=0.0;
             Force[iP][0]=0.0;
             Force[iP][1]=0.0;
             Force[iP][2]=0.0;
            
        }
        else{
        
            Velocity[iP][0] += Acceleration[iP][0]*Elastic_Dt;
            Velocity[iP][1] += Acceleration[iP][1]*Elastic_Dt;
            Velocity[iP][2] += Acceleration[iP][2]*Elastic_Dt;
            
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
        }
        #endif
       #ifdef Rolling2
        if (InitialPosition[iP][1]>0.3420){
            
            Position[iP][0] = InitialPosition[iP][0];
            Position[iP][1] = InitialPosition[iP][1];
            Position[iP][2] = InitialPosition[iP][2];
            
             Velocity[iP][0]=0.0;
             Velocity[iP][1]=0.0;
             Velocity[iP][2]=0.0;
             Force[iP][0]=0.0;
             Force[iP][1]=0.0;
             Force[iP][2]=0.0;
            
        }
        else{
        
            Velocity[iP][0] += Acceleration[iP][0]*Elastic_Dt;
            Velocity[iP][1] += Acceleration[iP][1]*Elastic_Dt;
            Velocity[iP][2] += Acceleration[iP][2]*Elastic_Dt;
            
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
        }
        #else
        
            Velocity[iP][0] += Acceleration[iP][0]*Elastic_Dt;
            Velocity[iP][1] += Acceleration[iP][1]*Elastic_Dt;
            Velocity[iP][2] += Acceleration[iP][2]*Elastic_Dt;
            
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
       #endif
        
    }
}


static void resetForce()
{
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]=0.0;
        }
    }
}


static void calculatePhysicalCoefficients()
{	
  #pragma acc kernels present (Property[0:ParticleCount],Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mass[iP]=Density[Property[iP]]*ParticleVolume;
    }
    
    #pragma acc kernels present (Kappa[0:ParticleCount],Property[0:ParticleCount],VolStrainP[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Kappa[iP]=BulkModulus[Property[iP]];
        if(VolStrainP[iP]<0.0){Kappa[iP]=0.0;}
    }
   
    
    #pragma acc kernels present(Lambda[0:ParticleCount],VolStrainP[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    Lambda[iP]=BulkViscosity[Property[iP]];
      //  if(VolStrainP[iP]<0.0){Lambda[iP]=0.0;}
    }
    
    #pragma acc kernels present (Property[0:ParticleCount],Mu[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mu[iP]=ShearViscosity[Property[iP]];
    }
     #pragma acc kernels present (Property[0:ParticleCount],Young[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Young[iP]=YoungModulus[Property[iP]];
    }
}



static void calculateDensityA()
{
    
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	{	
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
            if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
			double sum = 0.0;
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double radius = RadiusA;
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(radius*radius - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wa(rij,radius);
					sum += weight;
				}
			}
			DensityA[iP]=sum;
		}
	}
}


static void calculateGravityCenter()
{
 

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM])
	{
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
            if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
			double sum[DIM]={0.0,0.0,0.0};
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(RadiusG*RadiusG - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wg(rij,RadiusG);
					#pragma acc loop seq
					for(int iD=0;iD<DIM;++iD){
						sum[iD] += xij[iD]*weight/R2g*RadiusG;
					}
				}
			}
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				GravityCenter[iP][iD] = sum[iD];
			}
		}
	}
}

static void calculatePressureA()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],PressureA[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureA[iP] = CofA[Property[iP]]*(DensityA[iP]-N0a)/ParticleSpacing;
		if(N0a<=DensityA[iP]){
			PressureA[iP] = 0.0;
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],PressureA[0:ParticleCount],DensityA[0:ParticleCount],PressureA[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
        	double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusA;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = ratio_ij * dwadr(rij,radius);
            	const double dwji = ratio_ji * dwadr(rij,radius);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
            	#pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureA[iP]*dwij+PressureA[jP]*dwji)*eij[iD]* ParticleVolume;
                }
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}

static void calculateDiffuseInterface()
{
	
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
        if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
		const double ai = CofA[Property[iP]]*(CofK)*(CofK);
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			const double aj = CofA[Property[iP]]*(CofK)*(CofK);
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
			double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 > 0){
				const double rij = sqrt(rij2);
				const double wij = ratio_ij * wg(rij,RadiusG);
				const double wji = ratio_ji * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (aj*GravityCenter[jP][iD]*wji-ai*GravityCenter[iP][iD]*wij)/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				const double dwij = ratio_ij * dwgdr(rij,RadiusG);
				const double dwji = ratio_ji * dwgdr(rij,RadiusG);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (aj*GravityCenter[jP][iD]*dwji-ai*GravityCenter[iP][iD]*dwij)*xij[iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (gr)*eij[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateDensityP()
{
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VolStrainP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
	//if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = wp(rij,radius);
				sum += weight;
			}
		}
		VolStrainP[iP] = (sum - N0p);
	}
}

static void calculateDivergenceP()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Velocity[0:ParticleCount][0:DIM],DivergenceP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
	//if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					sum -= uij[iD]*eij[iD]*dw;
				}
			}
		}
		DivergenceP[iP]=sum;
	}
}

static void calculatePressureP()
{
	
	#pragma acc kernels present (PressureP[0:ParticleCount],Lambda[0:ParticleCount],DivergenceP[0:ParticleCount],VolStrainP[0:ParticleCount],Kappa[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureP[iP] = -Lambda[iP]*DivergenceP[iP];
		if(VolStrainP[iP]>0.0){
			PressureP[iP]+=Kappa[iP]*VolStrainP[iP];
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],PressureP[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
	if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] += (PressureP[iP]+PressureP[jP])*gradw[iD]*ParticleVolume;
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateInterfaceForce()
{
    #pragma acc kernels present (PressureP[0:ParticleCount], Lambda[0:ParticleCount], DivergenceP[0:ParticleCount], VolStrainP[0:ParticleCount], Kappa[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP = 0; iP < ParticleCount; ++iP) {
        PressureP[iP] = -Lambda[iP] * DivergenceP[iP];
        if (VolStrainP[iP] > 0.0) {
            PressureP[iP] += Kappa[iP] * VolStrainP[iP];
        }
    }

    #pragma acc kernels present(Property[0:ParticleCount], Position[0:ParticleCount][0:DIM], Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], NeighborCount[0:ParticleCount], PressureP[0:ParticleCount], Force[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP = StructureParticleBegin; iP <StructureParticleEnd; ++iP) {
        double force[DIM] = {0.0, 0.0, 0.0};
        #pragma acc loop seq
        for(int iN = 0; iN < NeighborCount[iP]; ++iN) {
            const int jP = Neighbor[iP][iN];
       if(STRUCTURE_BEGIN<=Property[jP] && Property[jP]<STRUCTURE_END ) continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD = 0; iD < DIM; ++iD) {
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];
            }
	    const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            const double radius = RadiusP;
            if (rij2 < radius * radius) {
                const double rij = sqrt(rij2); // Avoid division by zero
                const double dw = dwpdr(rij, radius);
                const double wij = weight(xij,RadiusP);
                double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};

                #pragma acc loop seq
                for(int iD = 0; iD < DIM; ++iD) {
                    force[iD] += (PressureP[iP] + PressureP[jP])*gradw[iD]*ParticleVolume;
                }
            }
        }

        #pragma acc loop seq
        for(int iD = 0; iD < DIM; ++iD) {
            Force[iP][iD] += force[iD];
        }
    }
}




static void calculateViscosityV(){

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Velocity[0:ParticleCount][0:DIM],Mu[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
    	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
        	if(RadiusV*RadiusV - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,RadiusV);
            	const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
        		double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
            	double fij[DIM] = {0.0,0.0,0.0};
        		#pragma acc loop seq
            	for(int iD=0;iD<DIM;++iD){
            		#ifdef TWO_DIMENSIONAL
            		force[iD] += 8.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#else
            		force[iD] += 10.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#endif
            	}
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}



static void calculateLamesconstant()
{
#pragma acc parallel loop present(Property[0:ParticleCount], \
                                  YoungModulus[0:TYPE_COUNT], \
                                  PoissonRatio[0:TYPE_COUNT], \
                                  LambdaLames[0:ParticleCount], \
                                  MuLames[0:ParticleCount])
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; iP++) {
        const double E = YoungModulus[Property[iP]];
        const double v = PoissonRatio[Property[iP]];

        LambdaLames[iP] = (E * v) / ((1.0 + v) * (1.0 - 2.0 * v));
        MuLames[iP]     = E / (2.0 * (1.0 + v));
    }
}



static void calculateNormalizer() {
#ifdef TWO_DIMENSION
    const int dim = 2;
#else
    const int dim = 3;
#endif

    // --- 初期化 ---
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; iP++) {
        #pragma acc loop seq
        for (int iD = 0; iD < dim; ++iD)
            #pragma acc loop seq
            for (int jD = 0; jD < dim; ++jD)
                Normalizer[iP][iD][jD] = 0.0;
    }

    // --- 加算 ---
    #pragma acc kernels present(InitialStructureNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], InitialPosition[0:ParticleCount][0:DIM], InitialStructureNeighborCount[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; iP++) {
    #pragma acc loop seq
        for (int iN = 0; iN < InitialStructureNeighborCount[iP]; ++iN) {
            const int jP = InitialStructureNeighbor[iP][iN];
            if (iP == jP) continue;

            double xij0[3] = {0.0};
           #pragma acc loop seq
            for (int iD = 0; iD < dim; ++iD) {
                xij0[iD] = Mod(InitialPosition[jP][iD] - InitialPosition[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];
            }
            if (dim == 2) xij0[2] = 0.0; // 安全のためゼロ埋め

            const double wij = weight(xij0, RadiusP);
           #pragma acc loop seq
            for (int iD = 0; iD < dim; ++iD){
            #pragma acc loop seq
                for (int jD = 0; jD < dim; ++jD){
                    Normalizer[iP][iD][jD] += wij * xij0[iD] * xij0[jD];
                    }
                }
        }

#ifdef TWO_DIMENSIONAL
// --- 2D: 行列式と逆行列 ---
const double a = Normalizer[iP][0][0];
const double b = Normalizer[iP][0][1];
const double c = Normalizer[iP][1][0];
const double d = Normalizer[iP][1][1];
const double detA = a * d - b * c;

if (detA != 0.0) {
    double inv[2][2];
    inv[0][0] =  d / detA;
    inv[0][1] = -b / detA;
    inv[1][0] = -c / detA;
    inv[1][1] =  a / detA;
      #pragma acc loop seq
    for (int iD = 0; iD < 2; ++iD){
         #pragma acc loop seq
        for (int jD = 0; jD < 2; ++jD){
            Normalizer[iP][iD][jD] = inv[iD][jD];
            }
          }
} 
else {
    // fallback: use identity or zero
    #pragma acc loop seq
    for (int iD = 0; iD < 2; ++iD){
         #pragma acc loop seq
        for (int jD = 0; jD < 2; ++jD){
            Normalizer[iP][iD][jD] = (iD == jD) ? 1.0 : 0.0;
            }
       }
}
#else
        // --- 3D: 余因子法で逆行列 ---
        const double detA =
            Normalizer[iP][0][0] * (Normalizer[iP][1][1] * Normalizer[iP][2][2] - Normalizer[iP][1][2] * Normalizer[iP][2][1]) -
            Normalizer[iP][0][1] * (Normalizer[iP][1][0] * Normalizer[iP][2][2] - Normalizer[iP][1][2] * Normalizer[iP][2][0]) +
            Normalizer[iP][0][2] * (Normalizer[iP][1][0] * Normalizer[iP][2][1] - Normalizer[iP][1][1] * Normalizer[iP][2][0]);

        if (detA != 0.0) {
            double adj[3][3];
            adj[0][0] =  Normalizer[iP][1][1]*Normalizer[iP][2][2] - Normalizer[iP][1][2]*Normalizer[iP][2][1];
            adj[0][1] = -Normalizer[iP][1][0]*Normalizer[iP][2][2] + Normalizer[iP][1][2]*Normalizer[iP][2][0];
            adj[0][2] =  Normalizer[iP][1][0]*Normalizer[iP][2][1] - Normalizer[iP][1][1]*Normalizer[iP][2][0];

            adj[1][0] = -Normalizer[iP][0][1]*Normalizer[iP][2][2] + Normalizer[iP][0][2]*Normalizer[iP][2][1];
            adj[1][1] =  Normalizer[iP][0][0]*Normalizer[iP][2][2] - Normalizer[iP][0][2]*Normalizer[iP][2][0];
            adj[1][2] = -Normalizer[iP][0][0]*Normalizer[iP][2][1] + Normalizer[iP][0][1]*Normalizer[iP][2][0];

            adj[2][0] =  Normalizer[iP][0][1]*Normalizer[iP][1][2] - Normalizer[iP][0][2]*Normalizer[iP][1][1];
            adj[2][1] = -Normalizer[iP][0][0]*Normalizer[iP][1][2] + Normalizer[iP][0][2]*Normalizer[iP][1][0];
            adj[2][2] =  Normalizer[iP][0][0]*Normalizer[iP][1][1] - Normalizer[iP][0][1]*Normalizer[iP][1][0];

            #pragma acc loop seq
            for (int iD = 0; iD < 3; ++iD){
            #pragma acc loop seq
                for (int jD = 0; jD < 3; ++jD){
                    Normalizer[iP][iD][jD] = adj[iD][jD] / detA;
                    }
                }
        }
#endif
    }
}

static void resetAcceleration()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Acceleration[iP][iD]=0.0;
        }
        #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            AngularVelocity[iP][iD]=0.0;
        }
    }
}


static void calculateElasticDeformationVector() {
#ifdef TWO_DIMENSIONAL
    const int dim = 2;
#else
    const int dim = 3;
#endif

    #pragma acc data present(Position[0:ParticleCount][0:3], \
                             InitialPosition[0:ParticleCount][0:3], \
                             InitialStructureNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
                             Original[0:ParticleCount], \
                             Conversion[0:ParticleCount][0:3][0:3], \
                             DeformGradient[0:ParticleCount][0:3][0:3], \
                             InitialStructureNeighborCount[0:ParticleCount], \
                             DomainWidth[0:3], \
                             Normalizer[0:ParticleCount][0:3][0:3])
    {
        // Step 1: Initialize DeformGradient
        #pragma acc parallel loop
        for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
            #pragma acc loop seq
            for (int i = 0; i < dim; ++i)
                #pragma acc loop seq
                for (int j = 0; j < dim; ++j)
                    DeformGradient[iP][i][j] = 0.0;
        }

        // Step 2: Compute moment matrix
        #pragma acc parallel loop
        for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
            #pragma acc loop seq
            for (int iN = 0; iN < InitialStructureNeighborCount[iP]; ++iN) {
                const int jP = InitialStructureNeighbor[iP][iN];
                if (iP == jP) continue;
                const int jOP = Original[jP];

                double ui[dim] = {0.0}, uj[dim] = {0.0}, xij0[dim] = {0.0}, xij[dim] = {0.0};
                #pragma acc loop seq
                for (int iD = 0; iD < dim; ++iD) {
                    ui[iD] = Mod(Position[iP][iD] - InitialPosition[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];
                    uj[iD] = Conversion[jP][iD][iD] *
                             (Mod(Position[jOP][iD] - InitialPosition[jOP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD]);
                    xij0[iD] = Mod(InitialPosition[jP][iD] - InitialPosition[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];
                    xij[iD] = xij0[iD] + (uj[iD] - ui[iD]);
                }

                double wij = weight(xij0, RadiusP);

                #pragma acc loop seq
                for (int i = 0; i < dim; ++i) {
                    #pragma acc loop seq
                    for (int j = 0; j < dim; ++j) {
                        #pragma acc atomic
                        DeformGradient[iP][i][j] += wij * xij[i] * xij0[j];
                    }
                }
            }
        }

        // Step 3: Multiply with Normalizer
        #pragma acc parallel loop
        for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
            double Ftemp[dim][dim] = {{0.0}};
            #pragma acc loop seq
            for (int i = 0; i < dim; ++i)
                #pragma acc loop seq
                for (int j = 0; j < dim; ++j) {
                    double sum = 0.0;
                    #pragma acc loop seq
                    for (int k = 0; k < dim; ++k)
                        sum += DeformGradient[iP][i][k] * Normalizer[iP][k][j];
                    Ftemp[i][j] = sum;
                }

            #pragma acc loop seq
            for (int i = 0; i < dim; ++i)
                #pragma acc loop seq
                for (int j = 0; j < dim; ++j)
                    DeformGradient[iP][i][j] = Ftemp[i][j];
        }
    }
}

static void calculateStress() {
#ifdef TWO_DIMENSIONAL
    const int dim = 2;
#else
    const int dim = 3;
#endif

    #pragma acc parallel loop present(DeformGradient[0:ParticleCount][0:3][0:3], \
                                      Strain[0:ParticleCount][0:3][0:3], \
                                      Stress[0:ParticleCount][0:3][0:3], \
                                      MuLames[0:ParticleCount], \
                                      LambdaLames[0:ParticleCount])
    for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP) {
        double strain[3][3] = {{0.0}};
        double traceStrain = 0.0;

        // --- Compute Green-Lagrange strain tensor ---
        #pragma acc loop seq
        for (int i = 0; i < dim; ++i) {
            #pragma acc loop seq
            for (int j = 0; j < dim; ++j) {
                double sum = 0.0;
                #pragma acc loop seq
                for (int k = 0; k < dim; ++k) {
                    sum += DeformGradient[iP][k][i] * DeformGradient[iP][k][j];
                }
                strain[i][j] = 0.5 * (sum - (i == j ? 1.0 : 0.0));
                if (i == j) traceStrain += strain[i][j];
            }
        }

        // --- Store strain tensor ---
        #pragma acc loop seq
        for (int i = 0; i < dim; ++i) {
            #pragma acc loop seq
            for (int j = 0; j < dim; ++j) {
                Strain[iP][i][j] = strain[i][j];
            }
        }

        // --- Compute 2nd Piola-Kirchhoff stress tensor ---
        const double mu = MuLames[iP];
        const double lambda = LambdaLames[iP];

        #pragma acc loop seq
        for (int i = 0; i < dim; ++i) {
            #pragma acc loop seq
            for (int j = 0; j < dim; ++j) {
                Stress[iP][i][j] = 2.0 * mu * strain[i][j];
                if (i == j) Stress[iP][i][j] += lambda * traceStrain;
            }
        }
    }
}


static void calculateStressForce() {
#ifdef TWO_DIMENSIONAL
    const int dim = 2;
#else
    const int dim = 3;
#endif

    #pragma acc data \
        present(Property[0:ParticleCount], \
                InitialStructureNeighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
                Velocity[0:ParticleCount][0:dim], \
                InitialPosition[0:ParticleCount][0:dim], \
                Original[0:ParticleCount], \
                DeformGradient[0:ParticleCount][0:dim][0:dim], \
                Stress[0:ParticleCount][0:dim][0:dim], \
                Normalizer[0:ParticleCount][0:dim][0:dim], \
                InitialStructureNeighborCount[0:ParticleCount], \
                Density[0:TYPE_COUNT], \
                Acceleration[0:ParticleCount][0:3], \
                Mass[0:ParticleCount], \
                DomainWidth[0:3])
    {
        #pragma acc parallel loop
        for (int iP = StructureParticleBegin; iP < StructureParticleEnd; ++iP)
        {
            double P[dim][dim] = {{0.0}};
            #pragma acc loop seq
            for (int i = 0; i < dim; ++i) {
                #pragma acc loop seq
                for (int j = 0; j < dim; ++j) {
                    double sum = 0.0;
                    #pragma acc loop seq
                    for (int k = 0; k < dim; ++k) {
                        #pragma acc loop seq
                        for (int l = 0; l < dim; ++l) {
                            sum += DeformGradient[iP][i][k] * Stress[iP][k][l] * Normalizer[iP][l][j];
                        }
                    }
                    P[i][j] = sum;
                }
            }

            #pragma acc loop seq
            for (int iN = 0; iN < InitialStructureNeighborCount[iP]; ++iN)
            {
                const int jP  = InitialStructureNeighbor[iP][iN];
                if (jP == iP) continue;
                const int jOP = Original[jP];

                double xij0[dim] = {0.0};
                #pragma acc loop seq
                for (int iD = 0; iD < dim; ++iD)
                    xij0[iD] = Mod(InitialPosition[jP][iD] - InitialPosition[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];

                const double wij = weight(xij0, RadiusP);

                double f[dim] = {0.0};
                #pragma acc loop seq
                for (int i = 0; i < dim; ++i) {
                    #pragma acc loop seq
                    for (int j = 0; j < dim; ++j)
                        f[i] += P[i][j] * xij0[j];
                    f[i] *= wij;
                }

                const double invRho_i = 1.0 / Density[Property[iP]];
                const double invRho_j = 1.0 / Density[Property[jOP]];

                #pragma acc loop seq
                for (int iD = 0; iD < dim; ++iD) {
                    #pragma acc atomic
                    Velocity[iP][iD] += invRho_i * f[iD]*Elastic_Dt;//*dim/DIM;
                    #pragma acc atomic
                    Velocity[jOP][iD] -= invRho_j * f[iD]*Elastic_Dt;//*dim/DIM;
                }
            }
        }
    }
}

static void resetAccel() {
   #pragma acc kernels present(Acceleration[0:ParticleCount][0:DIM])
#pragma acc loop independent
#pragma omp parallel for
    for (int iP = 0; iP < ParticleCount; ++iP) {
    #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Acceleration[iP][iD] = 0.0;
        }
    }
      #pragma acc kernels present(AngularVelocity[0:ParticleCount][0:DIM])
#pragma acc loop independent
#pragma omp parallel for
    for (int iP = 0; iP < ParticleCount; ++iP) {
    #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            AngularVelocity[iP][iD] = 0.0;
        }
    }
}





static void calculateGravity(){
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Force[iP][0] += Mass[iP]*Gravity[0];
        Force[iP][1] += Mass[iP]*Gravity[1];
        Force[iP][2] += Mass[iP]*Gravity[2];
    }
    
#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
    Force[iP][0] += Mass[iP]*Gravity[0];
    Force[iP][1] += Mass[iP]*Gravity[1];
    Force[iP][2] += Mass[iP]*Gravity[2];
}
}

static void calculateAcceleration()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
        Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
        Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
    }
#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
    Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
    Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
    Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
}
}

#ifdef Rolling
#define MAX_ANGLE (2.0 * M_PI / 180.0)  // 4度 = 0.0698132 rad
#define ROLLING_PERIOD 1.646          // 周期 [s]
#endif

static void calculateWall()
{
    // --- Reset force ---
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for (int iP = WallParticleBegin; iP < WallParticleEnd; ++iP) {
        for (int iD = 0; iD < DIM; ++iD)
            Force[iP][iD] = 0.0;
    }

#ifdef Rolling
    const double omega_t = 2.0 * M_PI / ROLLING_PERIOD;
    const double theta = MAX_ANGLE * sin(omega_t * Time);     // 角度θ(t)
    const double dtheta_dt = MAX_ANGLE * omega_t * cos(omega_t * Time); // 角速度

    // 差分回転角 Δθ を求める（1ステップ分）
    const double theta_prev = MAX_ANGLE * sin(omega_t * (Time - Dt));
    const double delta_theta = theta - theta_prev;

    const double cosD = cos(delta_theta);
    const double sinD = sin(delta_theta);
    const double Rz[3][3] = {
        { cosD, -sinD, 0.0 },
        { sinD,  cosD, 0.0 },
        {  0.0,   0.0, 1.0 }
    };

    // --- 回転運動を与える ---
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for (int iP = WallParticleBegin; iP < WallParticleEnd; ++iP) {
        const int iProp = Property[iP];
        double r[3] = {
            Position[iP][0] - WallCenter[iProp][0],
            Position[iP][1] - WallCenter[iProp][1],
            Position[iP][2] - WallCenter[iProp][2]
        };

        // 差分回転
        double r_rot[3];
        r_rot[0] = Rz[0][0]*r[0] + Rz[0][1]*r[1];
        r_rot[1] = Rz[1][0]*r[0] + Rz[1][1]*r[1];
        r_rot[2] = r[2]; // Z軸周りの回転のみ

        // 角速度ベクトル（Z軸周り）
        double w[3] = {0.0, 0.0, dtheta_dt};

        // 更新
        Velocity[iP][0] = w[1]*r_rot[2] - w[2]*r_rot[1];
        Velocity[iP][1] = w[2]*r_rot[0] - w[0]*r_rot[2];
        Velocity[iP][2] = w[0]*r_rot[1] - w[1]*r_rot[0];

        Position[iP][0] = r_rot[0] + WallCenter[iProp][0];
        Position[iP][1] = r_rot[1] + WallCenter[iProp][1];
        Position[iP][2] = r_rot[2] + WallCenter[iProp][2];
    }
    
        #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for (int iProp = WALL_BEGIN; iProp < WALL_END; ++iProp) {
        WallCenter[iProp][0] += WallVelocity[iProp][0] * Dt;
        WallCenter[iProp][1] += WallVelocity[iProp][1] * Dt;
        WallCenter[iProp][2] += WallVelocity[iProp][2] * Dt;
    }

#else
    // 元の剛体運動（非回転 or 定回転）
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for (int iP = WallParticleBegin; iP < WallParticleEnd; ++iP) {
    if(Time<0.2){
        const int iProp = Property[iP];
        double r[3] = {
            Position[iP][0] - WallCenter[iProp][0],
            Position[iP][1] - WallCenter[iProp][1],
            Position[iP][2] - WallCenter[iProp][2]
        };

        const double (&R)[DIM][DIM] = WallRotation[iProp];
        const double (&w)[DIM] = WallOmega[iProp];

        double r_rot[3];
        r_rot[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2];
        r_rot[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2];
        r_rot[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2];

        Velocity[iP][0] = w[1]*r_rot[2] - w[2]*r_rot[1] + WallVelocity[iProp][0];
        Velocity[iP][1] = w[2]*r_rot[0] - w[0]*r_rot[2] + WallVelocity[iProp][1];
        Velocity[iP][2] = w[0]*r_rot[1] - w[1]*r_rot[0] + WallVelocity[iProp][2];

        Position[iP][0] = r_rot[0] + WallCenter[iProp][0] + WallVelocity[iProp][0] * Dt;
        Position[iP][1] = r_rot[1] + WallCenter[iProp][1] + WallVelocity[iProp][1] * Dt;
        Position[iP][2] = r_rot[2] + WallCenter[iProp][2] + WallVelocity[iProp][2] * Dt;
    }
   }

    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for (int iProp = WALL_BEGIN; iProp < WALL_END; ++iProp) {
        WallCenter[iProp][0] += WallVelocity[iProp][0] * Dt;
        WallCenter[iProp][1] += WallVelocity[iProp][1] * Dt;
        WallCenter[iProp][2] += WallVelocity[iProp][2] * Dt;
    }
#endif
}




static void calculateVirialStressAtParticle()
{
	//const double (*x)[DIM] = Position;
	const double (*v)[DIM] = Velocity;
	

	#pragma acc kernels present (VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD]=0.0;
			}
		}
	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			// pressureP
			if(RadiusP*RadiusP - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = dwpdr(rij,RadiusP);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureP[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// pressureA
			if(RadiusA*RadiusA - rij2 > 0){
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dwij = ratio * dwadr(rij,RadiusA);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureA[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}

	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Mu[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// viscosity term
			if(RadiusV*RadiusV - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = -dwvdr(rij,RadiusV);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				const double vij[DIM] = {v[jP][0]-v[iP][0],v[jP][1]-v[iP][1],v[jP][2]-v[iP][2]};
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#ifdef TWO_DIMENSIONAL
					fij[iD] = 8.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#else
					fij[iD] = 10.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#endif
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=0.5*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// diffuse interface force (1st term)
			if(RadiusG*RadiusG - rij2 > 0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double weight = ratio * wg(rij,RadiusG);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*( -GravityCenter[iP][iD])*weight/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
			
			// diffuse interface force (2nd term)
			if(RadiusG*RadiusG - rij2 > 0.0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dw = ratio * dwgdr(rij,RadiusG);
				const double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (                     -GravityCenter[iP][iD])*xij[iD];
				}
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*(gr)*gradw[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}	
	

	#pragma acc kernels present(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM],VirialPressureAtParticle[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#ifdef TWO_DIMENSIONAL
		VirialPressureAtParticle[iP]=-1.0/2.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]);
		#else 
		VirialPressureAtParticle[iP]=-1.0/3.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]+VirialStressAtParticle[iP][2][2]);
		#endif
	}

}



static void calculatePeriodicBoundary( void )
{
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Position[iP][iD] = Mod(Position[iP][iD]-DomainMin[iD],DomainWidth[iD])+DomainMin[iD];
        }
    }
}

