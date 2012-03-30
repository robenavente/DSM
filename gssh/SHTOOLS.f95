module SHTOOLS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!	This module contains an interface block defining all the routines
!	used in the archive SHTOOLS. These are necessary in order to use
!	implicitly shaped arrays with most subroutines.
!
!	Copyright (c) 2005, Mark A. Wieczorek
!	All rights reserved.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	integer, parameter ::	CSPHASE_DEFAULT = 1	! The default is to EXCLUDE the 
							! CONDON-SHORTLEY phase of (-1)^m
							! in front of the Legendre functions.
							! To use this phase function, set
							! CSPHASE_DEFAULT = -1

	interface
	
		subroutine PlmBar_d1(p, dp, lmax, z, csphase, cnorm)
			integer, intent(in) ::	lmax
			real*8, intent(out) ::	p(:), dp(:)
       			real*8, intent(in) ::	z
       			integer, intent(in), optional :: csphase, cnorm
		end subroutine PlmBar_d1
       		
		subroutine PlmBar_d1_m2(p, dp, lmax, z, csphase, cnorm)
			integer, intent(in) ::	lmax
			real*8, intent(out) ::	p(:), dp(:)
       			real*8, intent(in) ::	z
       			integer, intent(in), optional :: csphase, cnorm
		end subroutine PlmBar_d1_m2

	end interface
	
end module SHTOOLS

