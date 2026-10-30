"""One fixed nested expansion; no integrand-name branches or fitted basis.

Core contract: admissible integrand -> norm2 and moments against fixed raw basis.
Callable adapter: common Gauss-Jacobi triangle quadrature for ordinary functions.
Quadrature agreement is a diagnostic, not a rigorous tolerance certificate.
"""
import mpmath as mp
import numpy as np
from scipy.special import roots_jacobi
import fractional_projection as fp

def jacobi01(n,alpha=0,beta=0):
    x,w=roots_jacobi(n,alpha,beta)
    return (x+1)/2,w/2**(alpha+beta+1)

class NestedExpansion:
    """Raw sequence h0, t/T, (t/T)^2, h1, h2, ...; fixed for every f."""
    def __init__(self,H,max_dimension,T=1,dps=70):
        if not isinstance(max_dimension,int) or max_dimension<3:
            raise ValueError('max_dimension must be an integer >=3')
        self.D=max_dimension;self.dps=dps
        with mp.workdps(dps):
            self.H,self.T=fp.parameters(H,T)
            n=self.D-2;C=fp.gram(self.H,n,self.T);G=mp.matrix(self.D)
            def index(k):return 0 if k==0 else k+2
            for k in range(n):
                for l in range(n):G[index(k),index(l)]=C[k,l]
            for j in [1,2]:
                b=fp.power_rhs(self.H,j,n,self.T)
                for k in range(n):G[j,index(k)]=G[index(k),j]=b[k]/self.T**j
                for l in [1,2]:G[j,l]=fp.power_cross(self.H,j,l,self.T)/self.T**(j+l)
            self.G=G;self.L=mp.cholesky(G)

    def raw_values(self,t,D=None):
        D=self.D if D is None else D
        t=np.asarray(t,dtype=float);T=float(self.T);x=t/T
        phi=np.empty(t.shape+(D,));phi[...,0]=1/np.sqrt(T)
        if D>1:phi[...,1]=x
        if D>2:phi[...,2]=x*x
        if D>3:phi[...,3:]=np.sqrt(2/T)*np.cos(np.pi*x[...,None]*np.arange(1,D-2))
        return phi

    def project_moments(self,norm2,moments,dimensions):
        """Same algebra for functions, generalized elements, and analytic oracles.

        Caller owns validity/accuracy of norm2 and moments. No clipping.
        """
        with mp.workdps(self.dps):
            norm2=mp.mpf(str(norm2));b=mp.matrix([mp.mpf(str(x)) for x in moments])
            if len(b)!=self.D or norm2<0 or not mp.isfinite(norm2) or any(not mp.isfinite(x) for x in b):
                raise ValueError('Finite nonnegative norm2 and D finite moments required')
            if norm2==0:
                if any(x!=0 for x in b):raise ValueError('A zero-norm input must have zero moments')
                if any(not 1<=D<=self.D for D in dimensions):raise ValueError('Dimension out of range')
                return [{'D':D,'norm2':'0','error2':'0','relative_error':'0','resolved_positive':False,
                         'exact_zero_norm':True,'coefficients_orthonormal':['0']*D} for D in dimensions]
            c=fp.forward(self.L,b);rows=[]
            for D in dimensions:
                if not 1<=D<=self.D:raise ValueError('Dimension out of range')
                err=norm2-mp.fsum(c[k]**2 for k in range(D))
                rows.append({'D':D,'norm2':str(norm2),'error2':str(err),'relative_error':str(mp.sqrt(err/norm2)) if err>0 else None,
                    'resolved_positive':bool(err>0),'coefficients_orthonormal':[str(c[k]) for k in range(D)]})
            return rows

    def raw_coefficients(self,orthonormal_coefficients):
        with mp.workdps(self.dps):
            D=len(orthonormal_coefficients)
            return [float(x) for x in fp.backward_transpose(self.L[:D,:D],mp.matrix(orthonormal_coefficients))]

    def callable_moments(self,f,nodes):
        """Vectorized f(t); integrability must be established separately.

        The same triangle rule is used for all inputs. It is not asserted to
        resolve arbitrary singularities or discontinuities at a fixed node count.
        """
        H,T=float(self.H),float(self.T)
        def evaluate(t):
            return np.broadcast_to(np.asarray(f(t),dtype=float),np.shape(t))
        if H==.5:
            x,w=jacobi01(nodes);t=T*x;v=evaluate(t)
            b=T*((w*v)@self.raw_values(t));norm2=T*np.dot(w,v*v)
        else:
            beta=2*H-1;x,wx=jacobi01(nodes,0,beta);v,wv=jacobi01(nodes,beta-1,0)
            s=T*x;fs=evaluate(s);ps=self.raw_values(s)
            b=np.zeros(self.D);n2=0.
            for start in range(0,nodes,12):
                sl=slice(start,min(start+12,nodes));t=s[sl,None]*v[None,:]
                ft=evaluate(t);pt=self.raw_values(t)
                avgf=ft@wv;avgp=np.einsum('j,ijk->ik',wv,pt)
                b+=np.einsum('i,ik->k',wx[sl],fs[sl,None]*avgp+ps[sl]*avgf[:,None])
                n2+=np.dot(wx[sl]*fs[sl],avgf)
            factor=H*beta*T**(2*H);b*=factor;norm2=2*factor*n2
        if not np.isfinite(norm2) or not np.all(np.isfinite(b)):
            raise ArithmeticError('Nonfinite quadrature result')
        return float(norm2),b

    def fit_callable(self,f,tolerance,dimensions=(5,9,17,33,65),node_counts=(200,320,480),stability_rtol=.002):
        """Numerically resolved stopping, with explicit failure/unresolved states.

        Uses the same settings for every input. No claimed universal complexity.
        """
        if tolerance<=0:raise ValueError('Tolerance must be positive')
        if len(node_counts)<2:raise ValueError('At least two quadrature levels required')
        dimensions=[d for d in dimensions if d<=self.D]
        if not dimensions:raise ValueError('No candidate dimension within the configured basis')
        runs=[]
        for n in node_counts:
            norm2,b=self.callable_moments(f,n)
            runs.append(self.project_moments(norm2,b,dimensions))
        rows=[]
        for k,D in enumerate(dimensions):
            a,b=runs[-2][k],runs[-1][k]
            ea,eb=float(a['error2']),float(b['error2']);scale=float(b['norm2'])
            if scale==0:
                rows.append({**b,'refinement_stable':False,'tolerance_pass':False,
                             'diagnostic':'All sampled values have zero norm; quadrature alone cannot certify a zero function.'})
                continue
            raw=np.array(self.raw_coefficients(b['coefficients_orthonormal']))
            residual=lambda t:np.asarray(f(t))-self.raw_values(t,D)@raw
            # Evaluate the actual approximation residual, avoiding cancellation
            # in norm2 - coefficient energy. Same operation for every callable.
            direct=[self.callable_moments(residual,n)[0] for n in node_counts[-2:]]
            err=direct[-1];change=abs(direct[-2]-err)
            ca=np.array(list(map(float,a['coefficients_orthonormal'])))
            cb=np.array(list(map(float,b['coefficients_orthonormal'])))
            coefficient_change2=float(np.dot(ca-cb,ca-cb))
            stable=direct[-2]>0 and err>0 and change<=stability_rtol*err and coefficient_change2<=stability_rtol*err
            row={**b,'nodes':node_counts[-1],'last_refinement_absolute':change,'refinement_stable':stable,
                'subtractive_error2':b['error2'],'error2':str(err),
                'relative_error':str(np.sqrt(err/scale)) if err>0 else None,
                'coefficient_refinement_energy':coefficient_change2,
                'resolved_positive':bool(err>0),'evaluation':'direct residual quadrature',
                'tolerance_pass':bool(stable and np.sqrt(err/scale)<=tolerance)}
            rows.append(row)
        first=next((x for x in rows if x['tolerance_pass']),None)
        below_target=any(x['relative_error'] is not None and float(x['relative_error'])<=tolerance for x in rows)
        return {'status':'numerically_resolved' if first else ('quadrature_unresolved' if below_target else 'target_not_reached_in_tested_dimensions'),
                'first_tested_dimension':None if first is None else first['D'],'tolerance':tolerance,'rows':rows,
                'qualification':'Quadrature refinement diagnostic, not a rigorous certificate; general moments API covers broader admissible inputs.'}
