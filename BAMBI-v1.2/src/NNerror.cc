#include "NNopt.h"
#include "myrand.h"
#ifdef PARALLEL
#include "mpi.h"
#endif

void CopyFile(std::string sourcefile, std::string targetfile)
{
	std::ifstream fin(sourcefile.c_str());
	std::ofstream fout(targetfile.c_str());
	for(;;)
	{
		if( fin.eof() ) break;
		std::string line;
		getline(fin,line);
		fout << line << "\n";
	}
	fin.close();
	fout.close();
}

int main(int argc, char **argv)
{
	int myid = 0;
#ifdef PARALLEL
 	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
	
	if (myid==0 && argc!=4 && argc!=3) {
		std::cout << "Usage:\n";
		std::cout << "NNerror <root> <network input file> <# of networks to train>\n";
		return 0;
	}
	
	std::string root = std::string(argv[1]);
	std::string netinfile = std::string(argv[2]);
	int NTrain = 10;
	if (argc==4) NTrain=atoi(argv[3]);
	
	int nNN = 0;
	if( myid == 0 )
	{
		for(;;)
		{
			std::stringstream ss;
			std::string filename = root + "network.txt";
			ss << filename << "." << nNN+1;
			filename = ss.str();
			
			std::ifstream fin(filename.c_str());
			if( fin.fail() ) break;
			fin.close();
		
			nNN++;
		}
	}
		
#ifdef PARALLEL
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&nNN, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

	if( nNN == 0 )
	{
		if (myid==0) std::cout << "No network files found.\n";
#ifdef PARALLEL
		MPI_Finalize();
#else
		exit(-1);
#endif
	} else {
		if (myid==0) printf("%d network files found.\n",nNN);
	}
	
	
	// set up the random no. generator
	unsigned int iseed = (unsigned int)time(NULL);
	srand(iseed);
	long seed = rand();
	seed += 123321;
	seed *= -1;
	
	size_t nlayers,nnodes[10];

	for( size_t n = 0; n < nNN; n++ )
	{
		std::stringstream ss1, ss2, ss3;
		
		std::string filename = root + "network.txt";
		ss1 << filename << "." << n+1;
		filename=ss1.str();
		FeedForwardNeuralNetwork nn;
		float rate, alpha, beta;
		nn.read(filename, &rate, &alpha, &beta);
		//std::cout << nn.nweights << " weights read\n";
				
		filename = root + "train.txt";
		ss2 << filename << "." << n+1;
		filename=ss2.str();
		TrainingData td(filename, false, false);
				
		filename = root + "test.txt";
		ss3 << filename << "." << n+1;
		filename=ss3.str();
		TrainingData vd(filename, false, false);
		
		PredictedData pd(nn.totnnodes, nn.totrnodes, td.ndata, td.nin, td.nout, td.cuminsets[td.ndata], td.cumoutsets[td.ndata]);
		float logL;
		nn.logLike(td, pd, logL, true);
		
		float sigma=0.0;
		int nstart, nend;
		nn.getnstartend(td.ndata, nstart, nend);
		for( size_t i = nstart; i < nend; i++ )
		{
			for( size_t j = td.ntimeignore[i]; j < td.ntime[i]; j++ )
			{
				size_t k = td.cuminsets[i]*pd.totnnodes+(j+1)*pd.totnnodes-pd.nout;
				size_t l = (td.cumoutsets[i]+j-td.ntimeignore[i])*td.nout;
				for( size_t m = 0; m < td.nout; m++ )
				{
					//if (i%100==0) {std::cout << "true=" << td.outputs[l+m] << "\tpred=" << pd.out[k+m] << "\n";}
					sigma += pow( td.outputs[l+m] - pd.out[k+m], 2.0 );
				}
			}
		}

#ifdef PARALLEL
		MPI_Barrier(MPI_COMM_WORLD);
		
		if( myid != 0 )
		{
			MPI_Send(&sigma, 1, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
		}
		else
		{
			float temp;
			MPI_Status status;
			for( size_t i = 1; i < nn.ncpus; i++ )
			{
				MPI_Recv(&temp, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
				sigma += temp;
			}
		}
		
		MPI_Bcast(&sigma, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
		
		sigma = sqrt( sigma / float(td.ndata) );
		if( myid == 0 ) printf("sigma=%g\n",sigma);
		
		// copy the original network file as .1
		{
			std::stringstream ss4, ss5;
			ss4 << std::string(root) + "network.txt" << "." << n+1;
			std::string sourcefile = ss4.str();
			ss5 << sourcefile << "." << 1;
			std::string targetfile = ss5.str();
			if (myid==0) CopyFile(sourcefile, targetfile);
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
#endif
		}
		
		
		for( size_t i = 2; i <= NTrain; i++ )
		{
			if (myid==0) printf("Training network %d\n",(int)i);
			
			// generate new data-set by adding gaussian noise to outputs
			float **data = new float* [td.ndata+vd.ndata];
			for( size_t j = 0; j < td.ndata+vd.ndata; j++ ) data[j] = new float [td.nin+td.nout];
			bool train[td.ndata+vd.ndata];
			size_t ni = 0, no = 0;
			for( size_t j = 0; j < td.ndata; j++ )
			{
				for( size_t k = 0; k < td.nin; k++ ) data[j][k] = td.inputs[ni+k];
				ni += td.nin;
				for( size_t k = 0; k < td.nout; k++ ) data[j][td.nin+k] = td.outputs[no+k] + gasdev(&seed)*sigma;
				no += td.nout;
				
				train[j] = true;
			}
			ni = 0, no = 0;
			for( size_t j = 0; j < vd.ndata; j++ )
			{
				for( size_t k = 0; k < vd.nin; k++ ) data[td.ndata+j][k] = vd.inputs[ni+k];
				ni += vd.nin;
				for( size_t k = 0; k < vd.nout; k++ ) data[td.ndata+j][vd.nin+k] = vd.outputs[no+k] + gasdev(&seed)*sigma;
				no += vd.nout;
				
				train[td.ndata+j] = false;
			}
			
			// copy original network file to root
			{
				std::stringstream ss4, ss5;
				ss4 << std::string(root) + "network.txt" << "." << n+1 << "." << 1;
				std::string sourcefile = ss4.str();
				ss5 << std::string(root) + "network.txt";
				std::string targetfile = ss5.str();
				if (myid==0) CopyFile(sourcefile, targetfile);
#ifdef PARALLEL
				MPI_Barrier(MPI_COMM_WORLD);
#endif
			}
			
			char *croot = &root[0];
			char *cnetinfile = &netinfile[0];
			if (myid==0) AddNewTrainData(croot, td.ndata+vd.ndata, td.nin, data, float(td.ndata)/(td.ndata+vd.ndata),train,3);
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			char dummy[50];
			bool resume;
			ReadInputFile1(cnetinfile,dummy,dummy,&resume);
			TrainNetwork(cnetinfile, croot, croot, &nlayers, nnodes, resume, false);
			
			for( size_t j = 0; j < td.ndata+vd.ndata; j++ ) delete [] data[j];
			delete data;
		
			// copy new network to .i
			{
				std::stringstream ss4, ss5;
				ss4 << std::string(root) + "network.txt";
				std::string sourcefile = ss4.str();
				ss5 << std::string(root) + "network.txt" << "." << n+1 << "." << i;
				std::string targetfile = ss5.str();
				if (myid==0) CopyFile(sourcefile, targetfile);
#ifdef PARALLEL
				MPI_Barrier(MPI_COMM_WORLD);
#endif
			}
		}
	}

#ifdef PARALLEL
	MPI_Finalize();
#endif

}
