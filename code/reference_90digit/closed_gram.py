"""Paper 1's closed cosine Gram form, applied without forming a dense matrix.

Finite cosine polynomials use the inherited exact formula. A zero-padded FFT
evaluates the structured quadratic form. Generic-function cosine interpolation
and convergence diagnostics are a separate approximation layer.
"""
import math,time
import mpmath as mp
import numpy as np
from scipy.fft import dct
from scipy.signal import fftconvolve

def evaluate_values(f,x):
    x=np.asarray(x,dtype=float)
    try:y=np.broadcast_to(np.asarray(f(x),dtype=float),x.shape)
    except (TypeError,ValueError):
        y=np.fromiter((float(f(float(t))) for t in x.ravel()),dtype=float,count=x.size).reshape(x.shape)
    if not np.all(np.isfinite(y)):raise ArithmeticError('Nonfinite input at an interior sampling point')
    return y

def kummer_sequence(H,N):
    """M_beta(k*pi): 60-digit low indices plus a 16-term large-index expansion."""
    beta=2*float(H)-1
    if not .5<=float(H)<1 or N<1:raise ValueError('Require 1/2<=H<1 and N>=1')
    if beta==0:return np.ones(N,dtype=complex)
    result=np.empty(N,dtype=complex)
    with mp.workdps(60):
        b=2*mp.mpf(str(H))-1
        for k in range(min(33,N)):result[k]=complex(mp.hyp1f1(b,b+1,1j*mp.pi*k))
    if N>33:
        k=np.arange(33,N);x=np.pi*k
        term=1/(1j*x);boundary=term.copy()
        for j in range(1,16):
            term*=(j-beta)/(1j*x);boundary+=term
        result[33:]=math.gamma(beta+1)*np.exp(.5j*np.pi*beta)*x**(-beta)+beta*np.where(k%2,-1.,1.)*boundary
    return result

class ClosedCosineGram:
    def __init__(self,H,T=1.):
        self.H=float(H);self.T=float(T);self._moments=np.empty(0,dtype=complex)
        if not .5<=self.H<1 or self.T<=0:raise ValueError('Require 1/2<=H<1 and T>0')

    def moments(self,N):
        if len(self._moments)<N:self._moments=kummer_sequence(self.H,N)
        return self._moments[:N]

    def norm2(self,coefficients):
        a=np.asarray(coefficients,dtype=float);N=len(a)
        if a.ndim!=1 or N<1 or not np.all(np.isfinite(a)):raise ValueError('Finite real coefficient vector required')
        if self.H==.5:
            value=float(np.dot(a,a));return value,32*np.finfo(float).eps*value
        beta=2*self.H-1;C=2*self.H*self.T**beta
        if N==1:return float(a[0]**2*self.T**beta),0.
        m=self.moments(N);k=np.arange(1,N);r=m[1:].imag;p=m[1:].real;v=a[1:]
        # Even extension, zero at index 0. The odd convolution kernel retains
        # only even differences, encoding the inherited checkerboard pattern.
        extent=N-1;extended=np.r_[v[::-1],0.,v]
        lag=np.arange(-2*extent,2*extent+1);kernel=np.zeros(len(lag))
        mask=(lag!=0)&(lag%2==0);kernel[mask]=1/lag[mask]
        convolution=fftconvolve(extended,kernel,mode='same')[extent+1:]
        diagonal=C*np.sum((p+beta*r/(np.pi*k))*v*v)
        correction=-2*C/np.pi*np.sum(r*v*convolution)
        even=k%2==0
        constant=a[0]**2*self.T**beta
        cross=-2*np.sqrt(2)*C/np.pi*a[0]*np.sum(r[even]*v[even]/k[even])
        value=math.fsum([constant,diagonal,correction,cross])
        sensitivity=64*np.finfo(float).eps*(abs(constant)+abs(diagonal)+abs(cross)+2*C/np.pi*np.sum(np.abs(r*v*convolution)))
        return value,float(sensitivity)

def spectral_covariance_norm2(f,H,T=1.,absolute_tolerance=1e-10,relative_tolerance=1e-4,
                             initial_terms=64,maximum_terms=262144,evaluation_budget=2000000,gram=None):
    """Same midpoint cosine interpolant and closed Gram rule for every input.

    Refinement tests resolve numerical work only. Input-space membership and a
    rigorous infinite tail do not follow from finite samples.
    """
    if absolute_tolerance<=0 or relative_tolerance<=0:raise ValueError('Positive tolerances required')
    gram=ClosedCosineGram(H,T) if gram is None else gram
    start=time.perf_counter();history=[];count=0;N=initial_terms;resolved=False
    def level(size):
        nonlocal count
        values=np.empty(size)
        for first in range(0,size,8192):
            last=min(first+8192,size);t=T*(np.arange(first,last)+.5)/size
            values[first:last]=evaluate_values(f,t)
        count+=size
        coefficients=np.sqrt(T/size)*dct(values,type=2,norm='ortho')
        return gram.norm2(coefficients)
    while N<=maximum_terms and count+N<=evaluation_budget:
        value,sensitivity=level(N)
        change=abs(value-history[-1]['value']) if history else None
        history.append({'terms':N,'value':value,'change':change,'roundoff_sensitivity':sensitivity})
        if len(history)>=3 and value>0:
            threshold=max(absolute_tolerance,relative_tolerance*value)
            if max(history[-1]['change'],history[-2]['change'])<=threshold and sensitivity<=threshold:
                # Dyadic midpoint grids can share a misleading plateau near
                # a jump. Test a neighboring, nonnested grid for every input.
                if count+N-1>evaluation_budget:break
                neighbor,ns=level(N-1)
                difference=abs(neighbor-value)
                history[-1].update({'neighbor_terms':N-1,'neighbor_value':neighbor,
                    'neighbor_change':difference,'neighbor_roundoff_sensitivity':ns})
                if neighbor>0 and max(difference,ns)<=threshold:
                    resolved=True;break
        if value<0:break
        N*=2
    if not history:raise ValueError('Budget permits no initial expansion')
    last=history[-1]
    estimate=max([r['change'] for r in history[-2:] if r['change'] is not None]+[
        last['roundoff_sensitivity'],last.get('neighbor_change',0),last.get('neighbor_roundoff_sensitivity',0)])
    return {'value':last['value'],'resolved':resolved,'error_estimate':estimate,'history':history,
        'function_evaluations':count,'seconds':time.perf_counter()-start,
        'qualification':'Closed Gram norm of a finite cosine interpolant; interpolation/refinement errors are numerical estimates, not a validated infinite tail.'}
