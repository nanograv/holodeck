import numpy as np
from holodeck.evolution import _Binary_Evolution

# some cosmological functions

def invE(z):
    OmegaM = 0.3
    Omegak = 0.
    OmegaLambda = 0.7
    return 1./np.sqrt(OmegaM*(1.+z)**3.+Omegak*(1.+z)**2.+OmegaLambda)

def dtdz(z):
    t0 = 14.
    if z == -1.:
        z = -0.99
    return t0/(1.+z)*invE(z)

# extra functions end

class GM_Semi_Analytic(_Binary_Evolution):

    _SELF_CONSISTENT = False

    def __init__(self, Phi0, PhiI, M0, alpha0, alphaI, f0, alphaf, betaf, gammaf,
                 t0, alphatau, betatau, gammatau, Mstar, alphastar):
        """
        M1: mass of primary galaxy
        q: mass ratio between galaxies
        zp: redshift
        f: frequency of the spectrum
        Phi0, PhiI: galaxy stellar mass function renormalisation rate
        M0: scale mass
        alpha0, alphaI: galaxy stellar mass function slope
        pair fraction: f0 rate, alphaf mass power law, betaf redshift power law,
                       gammaf mass ratio power law
        merger time scale: t0 time scale, alphatau mass power law,
                           betatau redshift power law, gammatau mass ratio power law
        Mstar, alphastar: M*-MBH relation
        e0: eccentricity of the binaries
        rho0: galaxy density parameter
        [-2.8,-0.2,11.25,-1.25,0.,0.025,0.,0.8,0.,1.,0.,-0.5,0.,8.25,1.]#,0.4,0.5,0.]
        """
        self.M1 = np.logspace(9,12,25)
        self.M1diff = (M1.max()-M1.min())/(len(M1)-1.)/2.
        self.q = np.linspace(0.25,1,10)
        self.qdiff = (q.max()-q.min())/(len(q)-1.)/2.
        self.alphaf = alphaf
        self.betaf = betaf
        self.gammaf = gammaf
        self.t0 = t0
        self.alphatau = alphatau
        self.betatau = betatau
        self.gammatau = gammatau
        self.Mstar = 10.**Mstar
        self.alphastar = alphastar
        self.zp = np.linspace(0.,1.5,5)
        self.zpdiff = (zp.max()-zp.min())/(len(zp)-1.)/2.
        self._f0 = f0/self.fpair_norm()
        self._Phi0 = Phi0
        self._PhiI = PhiI
        self._M0 = 10.**M0
        self._alpha0 = alpha0
        self._alphaI = alphaI
        self._beta1 = betaf - betatau
        self._gamma1 = gammaf - gammatau
        self.MBH1 = self.MBH(self.M1)
        self.MBH1diff = (self.MBH1.max()-self.MBH1.min())/(len(self.MBH1)-1.)/2.
        self.M2 = np.zeros((len(M1),len(q)))
        self.MBH2 = np.zeros((len(M1),len(q)))
        for i,j in np.ndindex(len(M1),len(q)):
            self.M2[i,j] = self.M1[i]*q[j]
            self.MBH2[i,j] = self.MBH(self.M2[i,j])
        self.mergerrate = np.zeros((len(self.M1),len(self.1),len(self.zp)))
                                    #Mc, q, redshift array lengths
        return

    def _init_step_zero(self):
        super()._init_step_zero()
        self.mergerrate = self.output() #for black hole chirp mass: self.grid()
        return

    def _take_next_step(self, step):
        return EVO.END

    def zprime(self,M1,q,zp):
        """
        redshift condition, need to improve cut at the age of the universe
        """
        t0 = si.quad(dtdz,0,zp)[0]
        tau0 = self.tau(M1,q,zp)
        if t0+tau0 < 13.:
            result = so.fsolve(lambda z: si.quad(dtdz,0,z)[0] - self.tau(M1,q,z) - t0,0.)[0]
        else:
            result = -1.
        return result

    def tau(self,M1,q,z):
        """
        merger time scale
        """
        return self.t0*(M1/5.7e10)**(self.alphatau)*(1.+z)**self.betatau*q**self.gammatau

    def fpair_norm(self,qlow=0.25,qhigh=1.):
        """
        galaxy pair fraction normalization
        """
        return (qhigh**(self.gammaf+1.)-qlow**(self.gammaf+1.))/(self.gammaf+1.)

    def dndM1par(self,M1,q,z,n2,alpha1):
        """
        d3n/dM1dqdz from parameters, missing b^alphatau/a^alphaf
        b = 0.4*h^-1, a = 1
        """
        return (n2*self._M0*(M1/self._M0)**alpha1*np.exp(-M1/self._M0)*q**self._gamma1*
                (1.+z)**self._beta1*dtdz(z)*(0.4/0.7*1.e11/self._M0)**self.alphatau/
                (1.e11/self._M0)**self.alphaf)

    def dMBHdMG(self,M):
        """
        dMBH/dMG
        """
        return self.Mstar*self.alphastar/1.e11*(M/1.e11)**(self.alphastar-1.)

    def dndMc(self,M1,q,z,n2,alpha1):
        """
        d3n/dMcdqdz
        """
        Mred = M1*0.615 #bulge mass to total mass ratio
        return (self.dndM1par(M1,q,z,n2,alpha1)/0.615/
                self.dMBHdMG(Mred)*10.**self.MBH(Mred)*np.log(10.))

    def MBH(self,M):
        """
        mass of the black hole Mstar-Mbulge relation without scattering
        """
        return np.log10(self.Mstar*(M/1.e11)**self.alphastar)

    def output(self,function='dndMc'):
        """
        input 3 x 1d array M1,q,z
        output 3d array (M1,q,z) (galaxy mass, galaxy mass ratio, redshift) of values for function
        """
        output = np.zeros((len(self.M1),len(self.q),len(self.zp)))
        for i,j,k in np.ndindex(len(self.M1),len(self.q),len(self.zp)):
            z = self.zprime(self.M1[i],self.q[j],self.zp[k])
            if z <= 0.:
                output[i,j,k] = 1.e-20
            else:
                Phi0 = 10.**(self._Phi0+self._PhiI*z)
                alpha0 = self._alpha0+self._alphaI*z
                alpha1 = alpha0 + self.alphaf - self.alphatau
                n2 = Phi0*self._f0/self._M0/self._M0/self.t0
                alpha2 = alpha1 - 1. - self._gamma1
                if function=='dndMc':
                    output[i,j,k] = self.dndMc(self.M1[i],self.q[j],z,n2,alpha1)*4.*self.MBH1diff*self.qdiff
                else:
                    raise UserWarning("output function not defined")
        return output

    def grid(self,n0=None,M1=None,M2=None,function='dndMc'):
        """
        input 3d array n0, 1d array MBH1, 2d array MBH2
        output 3d array (Mcbh,qbh,z) (black hole chirp mass,
        black hole mass ratio, redshift) of values for function
        """
        if n0 is None:
            n0 = self.output(function)
        if M1 is None:
            M1 = 10.**self.MBH1
        if M2 is None:
            M2 = 10.**self.MBH2
        Mcbh = np.linspace(5,11,30)
        qbh = np.linspace(0,1,10)
        Mcbhdiff = (Mcbh.max()-Mcbh.min())/(len(Mcbh)-1.)/2.
        qbhdiff = (qbh.max()-qbh.min())/(len(qbh)-1.)/2.
        output = np.zeros((len(Mcbh),len(qbh),len(self.zp)))
        Mc = np.zeros((len(M1),len(M2[0,:])))
        q = np.zeros((len(M1),len(M2[0,:])))
        for i,j in np.ndindex(len(M1),len(M2[0,:])):
            Mc[i,j] = np.log10(mchirp(M1[i],M2[i,j]))
            if M2[i,j] > M1[i]:
                q[i,j] = M1[i]/M2[i,j]
            else:
                q[i,j] = M2[i,j]/M1[i]
        for i,j in np.ndindex(len(M1),len(M2[0,:])):
            for i0,j0 in np.ndindex(len(Mcbh),len(qbh)):
                if abs(Mc[i,j]-Mcbh[i0]) < Mcbhdiff and abs(q[i,j]-qbh[j0]) < qbhdiff:
                    for k in range(len(self.zp)):
                        output[i0,j0,k] += n0[i,j,k]/1.3
                else:
                    pass
        return output

