"""Fourier-style callable interface: fixed basis, fixed dual kernels, one rule.

This module does not classify f or select a formula for a named input family.
All quadrature error reports are numerical estimates, not interval certificates.
"""
from dataclasses import dataclass
import math, warnings, time
import numpy as np
import mpmath as mp
from scipy.integrate import quad_vec,quad,solve_ivp,IntegrationWarning
from general_expansion import NestedExpansion


class EvaluationBudgetExceeded(RuntimeError):
    pass


@dataclass
class ExpansionResult:
    engine: object
    coefficients: np.ndarray
    raw_coefficients: np.ndarray
    diagnostics: dict

    def __call__(self,t):
        return self.engine.raw_values(t,len(self.coefficients))@self.raw_coefficients

    def truncate(self,terms):
        if not 1<=terms<=len(self.coefficients):raise ValueError('Requested prefix is unavailable')
        c=self.coefficients[:terms].copy()
        raw=np.asarray(self.engine.raw_coefficients([str(x) for x in c]))
        return ExpansionResult(self.engine,c,raw,{**self.diagnostics,'prefix_of':len(self.coefficients)})

    def as_dict(self):
        return {'H':str(self.engine.H),'T':str(self.engine.T),'terms':len(self.coefficients),
                'coefficients':self.coefficients.tolist(),'raw_coefficients':self.raw_coefficients.tolist(),
                **self.diagnostics}


class AutomaticExpansion(NestedExpansion):
    """Precompute once for H,T,D; reuse for every admissible real input f."""
    def __init__(self,H,terms=33,T=1,dps=75):
        started=time.perf_counter()
        super().__init__(H,terms,T,dps)
        with mp.workdps(dps):self.whitener=np.array((self.L**-1).tolist(),dtype=float)
        self.beta=2*float(self.H)-1
        self.frequency=np.pi*np.arange(1,terms-2)/float(self.T)
        if self.beta:
            self._origin=.1;self._max_argument=max(1.,(terms-3)*np.pi)
            with mp.workdps(60):
                b=2*self.H-1
                initial=complex(mp.hyp1f1(b,b+1,1j*mp.mpf(str(self._origin))))
            # x M'(x)+beta M(x)=beta exp(ix), M(0)=1.
            self._solution=solve_ivp(lambda x,y:self.beta*(np.exp(1j*x)-y)/x,
                [self._origin,self._max_argument],[initial],method='DOP853',
                rtol=5e-14,atol=2e-15,dense_output=True,max_step=.35)
            if not self._solution.success:raise ArithmeticError('Fixed-kernel ODE did not converge')
            # Fixed Chebyshev panels accelerate repeated vector kernel lookup.
            # This does not depend on f. Reference checks include the interpolation.
            count=max(1,math.ceil(self._max_argument));self._panel_width=self._max_argument/count
            theta=np.pi*(np.arange(12)+.5)/12
            x=(np.arange(count)[:,None]+.5)*self._panel_width+.5*self._panel_width*np.cos(theta)
            values=self._M_ode(x)
            self._cheb=(2/12)*np.cos(np.arange(12)[:,None]*theta)@values.T
            self._cheb[0]*=.5
        self.setup_seconds=time.perf_counter()-started

    def _M_ode(self,x):
        x=np.asarray(x,dtype=float);a=np.abs(x);out=np.empty(x.shape,dtype=complex)
        small=a<self._origin
        if np.any(small):
            z=1j*a[small];term=np.ones_like(z);value=term.copy()
            for n in range(1,24):
                term*=z/n;value+=self.beta/(self.beta+n)*term
            out[small]=value
        if np.any(~small):out[~small]=self._solution.sol(a[~small])[0]
        return np.where(x<0,np.conj(out),out)

    def _M(self,x):
        x=np.asarray(x,dtype=float);a=np.abs(x)
        index=np.minimum((a/self._panel_width).astype(int),self._cheb.shape[1]-1)
        local=2*(a/self._panel_width-index)-1
        value=np.polynomial.chebyshev.chebval(local,self._cheb[:,index],tensor=False)
        return np.where(x<0,np.conj(value),value)

    def kernel_values(self,t):
        """K_H phi at a scalar t, independent of the supplied f."""
        H,T=float(self.H),float(self.T);t=float(t)
        if not 0<=t<=T:raise ValueError('Kernel argument must be in [0,T]')
        if H==.5:return self.raw_values(t)
        beta=self.beta;right=T-t;k=np.empty(self.D)
        k[0]=H*(t**beta+right**beta)/np.sqrt(T)
        for j in [1,2]:
            left=t**(j+beta)*math.gamma(j+1)*math.gamma(beta)/math.gamma(j+beta+1)
            other=sum(math.comb(j,m)*t**(j-m)*right**(beta+m)/(beta+m) for m in range(j+1))
            k[j]=H*beta*(left+other)/T**j
        if self.D>3:
            omega=self.frequency
            k[3:]=np.sqrt(2/T)*H*np.real(np.exp(1j*omega*t)*(
                t**beta*self._M(-omega*t)+right**beta*self._M(omega*right)))
        return k

    def dual_values(self,t):
        return self.whitener@self.kernel_values(t)

    def expand(self,f,coefficient_tolerance=1e-8,max_intervals=2048):
        """Return coefficients and a callable approximation; no norm oracle needed.

        The caller specifies an admissible f. Finite coefficient integrals alone
        cannot establish membership in the covariance Hilbert space.
        """
        if coefficient_tolerance<=0:raise ValueError('Positive tolerance required')
        T=float(self.T);panels=max(8,math.ceil((self.D-3)/4))
        points=np.linspace(0,T,panels+1)[1:-1]
        calls=0
        def integrand(t):
            nonlocal calls
            calls+=1;value=float(f(t))
            if not math.isfinite(value):raise ArithmeticError('Nonfinite input away from an endpoint')
            return value*self.dual_values(t)
        start=time.perf_counter();runs=[]
        for factor in [1.,.2]:
            c,err,info=quad_vec(integrand,0,T,epsabs=coefficient_tolerance*factor,
                epsrel=coefficient_tolerance*factor,norm='2',points=points,
                limit=max_intervals,quadrature='gk21',full_output=True)
            runs.append((c,err,info))
        c,err,info=runs[-1];change=float(np.linalg.norm(c-runs[0][0]))
        threshold=coefficient_tolerance*max(1.,float(np.linalg.norm(c)))
        resolved=all(x[2].success for x in runs) and change<=threshold
        raw=np.asarray(self.raw_coefficients([str(x) for x in c]))
        diagnostics={'status':'coefficients_numerically_resolved' if resolved else 'coefficient_quadrature_unresolved',
            'coefficient_tolerance':coefficient_tolerance,'quadrature_error_estimate':float(err),
            'coefficient_refinement_difference':change,'function_evaluations':calls,
            'quadrature_status':[int(x[2].status) for x in runs],'intervals':[len(x[2].intervals) for x in runs],
            'online_seconds':time.perf_counter()-start,'setup_seconds':self.setup_seconds,
            'qualification':'Numerical coefficient accuracy only; not a truncation-error certificate or an automatic integrability proof.'}
        return ExpansionResult(self,c,raw,diagnostics)

    def covariance_norm2(self,f,tolerance=1e-7,evaluation_budget=2000000):
        """Shared adaptive residual/norm integral, with kernel weights removed.

        For H>1/2 the exact identity is T^(2H) int r(s(z)) int
        r(s(z)*(1-v^(1/beta))) dv dz, s(z)=T*z^(1/(2H)).
        Nested error estimates are diagnostics, not rigorous global bounds.
        """
        if tolerance<=0 or evaluation_budget<1:raise ValueError('Positive tolerance and evaluation budget required')
        H,T=float(self.H),float(self.T);count=0
        def evaluate(t):
            nonlocal count
            count+=1
            if count>evaluation_budget:raise EvaluationBudgetExceeded('Norm/residual evaluation budget exhausted')
            y=float(f(float(t)))
            if not math.isfinite(y):raise ArithmeticError('Nonfinite function in covariance norm integration')
            return y
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',IntegrationWarning)
            if H==.5:
                value,err=quad(lambda t:evaluate(t)**2,0,T,epsabs=tolerance*.1,epsrel=tolerance*.1,limit=512)
                diagnostics={'outer_error_estimate':err,'integrated_inner_error_estimate':0.}
            else:
                beta=self.beta;inner_error_max=0.
                def outer(z):
                    nonlocal inner_error_max
                    s=T*z**(1/(2*H));fs=evaluate(s)
                    def inner(v):
                        fraction=-np.expm1(np.log(v)/beta)
                        return evaluate(s*fraction)
                    inner_value,inner_error=quad(inner,0,1,epsabs=tolerance*.02,
                        epsrel=tolerance*.05,limit=256)
                    inner_error_max=max(inner_error_max,inner_error)
                    return np.array([fs*inner_value,abs(fs)*inner_error])
                panels=max(4,math.ceil((self.D-3)/8))
                points=(np.linspace(0,1,panels+1)[1:-1])**(2*H)
                data,outer_error,info=quad_vec(outer,0,1,epsabs=tolerance*.1,
                    epsrel=tolerance*.1,norm='2',points=points,limit=512,full_output=True)
                value=T**(2*H)*data[0]
                diagnostics={'outer_error_estimate':float(T**(2*H)*outer_error),
                    'integrated_inner_error_estimate':float(T**(2*H)*data[1]),
                    'largest_inner_error_estimate':inner_error_max,'outer_success':bool(info.success)}
        diagnostics.update({'value':float(value),'function_evaluations':count,'seconds':time.perf_counter()-start,
            'integration_warning_count':len(caught),'warnings':list(dict.fromkeys(str(w.message) for w in caught))[:3],
            'qualification':'Nested adaptive quadrature estimates; no interval enclosure.'})
        return diagnostics

    def assess(self,f,expansion,relative_tolerance=.001,norm_tolerance=1e-7,evaluation_budget=2000000,norm=None):
        """Assess a returned approximation without changing its fixed basis."""
        if expansion.engine is not self:raise ValueError('Expansion belongs to another fixed basis')
        if relative_tolerance<=0:raise ValueError('Positive relative tolerance required')
        residual=lambda t:float(f(t))-float(expansion(t))
        norm=self.covariance_norm2(f,norm_tolerance,evaluation_budget) if norm is None else norm
        errors=[self.covariance_norm2(residual,norm_tolerance*factor,evaluation_budget) for factor in [1.,.2]]
        e0,e1=errors[0]['value'],errors[1]['value'];n2=norm['value']
        # Resolve relative to the requested budget, not relative to a tiny
        # residual that does not need near-machine relative accuracy.
        target2=relative_tolerance**2*n2
        stability=abs(e1-e0)<=.02*max(e1,target2)
        diagnostics_ok=all(x['integration_warning_count']==0 and x.get('outer_success',True) for x in [norm]+errors)
        estimated_error=max(x['outer_error_estimate']+x['integrated_inner_error_estimate'] for x in errors)
        norm_error=norm['outer_error_estimate']+norm['integrated_inner_error_estimate']
        stable=n2>0 and e0>=0 and e1>=0 and stability and diagnostics_ok and norm_error<=.001*n2 and estimated_error<=.05*max(e1,target2)
        rel=math.sqrt(e1/n2) if n2>0 and e1>=0 else None
        return {'status':'tolerance_numerically_resolved' if stable and e1<=target2 else ('target_not_reached' if stable else 'residual_quadrature_unresolved'),
            'relative_error':rel,'relative_tolerance':relative_tolerance,'norm':norm,'residual_runs':errors,
            'refinement_stable':bool(stable),'qualification':'Estimated error of the computed approximation; no rigorous certificate and no claim of a universal convergence rate.'}

    def fit(self,f,relative_tolerance=.001,dimensions=(5,9,17,33,65),coefficient_tolerance=1e-8,evaluation_budget=2000000):
        """One automatic procedure: coefficients, screening, residual confirmation.

        Screening selects a candidate, not a certified minimal dimension.
        """
        if relative_tolerance<=0:raise ValueError('Positive relative tolerance required')
        dimensions=sorted(set(d for d in dimensions if 1<=d<=self.D))
        if not dimensions:raise ValueError('No permitted candidate dimensions')
        start=time.perf_counter();full=self.expand(f,coefficient_tolerance)
        try:
            norm=self.covariance_norm2(f,min(1e-8,relative_tolerance**2*.01),evaluation_budget)
        except (EvaluationBudgetExceeded,ArithmeticError) as exc:
            return {'status':'norm_quadrature_unresolved','selected_dimension':None,
                'expansion':full.as_dict(),'reason':str(exc),'total_seconds':time.perf_counter()-start}
        if norm['value']<=0:
            return {'status':'input_norm_unresolved','full_expansion':full.as_dict(),'norm':norm}
        target2=relative_tolerance**2*norm['value']
        screen=[{'D':D,'subtractive_residual2':float(norm['value']-np.dot(full.coefficients[:D],full.coefficients[:D]))} for D in dimensions]
        candidates=[x['D'] for x in screen if x['subtractive_residual2']<=2*target2]
        if dimensions[-1] not in candidates:candidates.append(dimensions[-1])
        assessments=[];selected=None
        for D in candidates:
            prefix=full.truncate(D)
            try:
                assessment=self.assess(f,prefix,relative_tolerance,
                    norm_tolerance=max(1e-13,.02*target2),evaluation_budget=evaluation_budget,norm=norm)
            except (EvaluationBudgetExceeded,ArithmeticError) as exc:
                assessment={'status':'residual_quadrature_unresolved','reason':str(exc)}
            assessments.append({'D':D,**assessment})
            if assessment['status']=='tolerance_numerically_resolved' and full.diagnostics['status']=='coefficients_numerically_resolved':
                selected=prefix;break
        return {'status':'tolerance_numerically_resolved' if selected else 'target_not_resolved_in_budget',
            'selected_dimension':None if selected is None else len(selected.coefficients),
            'expansion':(full if selected is None else selected).as_dict(),
            'screening':screen,'assessments':assessments,'total_seconds':time.perf_counter()-start,
            'qualification':'Automatic numerical workflow for an admissible callable; acceptance uses error estimates, not validated enclosures. No minimal-dimension claim.'}
