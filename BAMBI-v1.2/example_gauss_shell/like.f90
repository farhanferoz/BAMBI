module like
	
use params
use utils1
implicit none
      
contains
      
      
!=======================================================================

subroutine getloglikens(Cube,n_dim,nPar,slhood,context)
         
	implicit none
      	
	integer n_dim,nPar,context
	double precision Cube(nPar),slhood
	double precision temp(n_dim),dist,loclik
	integer i,j
	double precision TwoPi
         
	TwoPi=6.2831853

	slhood=-huge(1.d0)*epsilon(1.d0)
	
	!rescaling the parameters in unit hypercube according to the prior
	do i=1,n_dim
		temp(i)=(spriorran(i,2)-spriorran(i,1))*Cube(i)+spriorran(i,1)
	end do
	Cube(1:n_dim)=temp(1:n_dim)

	do i=1,sModes
		dist=(sqrt(sum((temp(1:n_dim)-sc(i,1:n_dim))**2.))-sr(i))**2
		loclik=-dist/(2.*(sw(i)**2.))-log(TwoPi*sw(i)**2)/2.
		slhood=logSumExp(slhood,loclik)
	end do

end subroutine getloglikens
      
!=======================================================================

subroutine getphysparams(Cube)
         
	implicit none
      	
	double precision Cube(nest_nPar)
	integer i
	
	!rescaling the parameters in unit hypercube according to the prior
	do i=1,sdim
		Cube(i)=(spriorran(i,2)-spriorran(i,1))*Cube(i)+spriorran(i,1)
	end do

end subroutine getphysparams
      
!=======================================================================

subroutine getallparams(Cube)
         
	implicit none
      	
	double precision Cube(nest_nPar)
	integer i
	
	!rescaling the parameters in unit hypercube according to the prior
	do i=1,sdim
		Cube(i)=(spriorran(i,2)-spriorran(i,1))*Cube(i)+spriorran(i,1)
	end do

end subroutine getallparams
      
!=======================================================================


end module like
