module nestwrapper

! Nested sampling includes

use Nested
use params
use like
   
implicit none
   
contains

!-----*-----------------------------------------------------------------

subroutine nest_Sample
	
	implicit none
	
   	integer nclusters				! total number of clusters found
	integer context
   	integer maxNode 				! variables used by the posterior routine
	
	INTERFACE
    		!the likelihood function
    		subroutine getloglikeC(Cube,n_dim,nPar,lnew,context_pass)
			integer n_dim,nPar,context_pass
			double precision lnew,Cube(nPar)
		end subroutine getloglikeC
    	end INTERFACE
	
	INTERFACE
		!this function passes control to the user after every nproc likelihood calls
    		subroutine bambi(ndata, ndim, tdata, lowlike)
			integer ndata, ndim
			double precision, pointer :: tdata(:,:)
			double precision lowlike
		end subroutine bambi
	end INTERFACE
   
   
   	! calling MultiNest
	
   	call nestRun(nest_mmodal,nest_ceff,nest_nlive,nest_tol,nest_efr,sdim,nest_nPar, &
   	nest_nClsPar,nest_maxModes,nest_updInt,nest_Ztol,nest_root,nest_rseed,nest_pWrap, &
   	nest_fb,nest_resume,nest_outfile,nest_initMPI,nest_logZero,nest_maxIter,getLogLikeC, &
	dumper,bambi,context)

end subroutine nest_Sample

!-----*-----------------------------------------------------------------

! dumper, called after every updInt*10 iterations

subroutine dumper(nSamples, nlive, nPar, physLive, posterior, paramConstr, maxLogLike, logZ, logZerr, context)

	implicit none

	integer nSamples				! number of samples in posterior array
	integer nlive					! number of live points
	integer nPar					! number of parameters saved (physical plus derived)
	double precision, pointer :: physLive(:,:)	! array containing the last set of live points
	double precision, pointer :: posterior(:,:)	! array with the posterior distribution
	double precision, pointer :: paramConstr(:)	! array with mean, sigmas, maxlike & MAP parameters
	double precision maxLogLike			! max loglikelihood value
	double precision logZ				! log evidence
	double precision logZerr			! error on log evidence
	integer context					! not required by MultiNest, any additional information user wants to pass
	
end subroutine dumper

!-----*-----------------------------------------------------------------

subroutine getmnparams(root, netfile, resume, logzero, do_bambi, use_NN)
	
	implicit none
	
	character*100 root, netfile
	integer resume, do_bambi, use_NN
	double precision logzero
	integer i
		
	root=nest_root
	netfile=nest_netfile
	if(nest_resume) then
		resume=1
	else
		resume=0
	endif
	use_NN=nest_useNN
	if(nest_bambi) then
		use_NN=0
		do_bambi=1
	else
		do_bambi=0
	endif
	logzero=nest_logZero
		
end subroutine getmnparams

!-----*-----------------------------------------------------------------

end module nestwrapper
