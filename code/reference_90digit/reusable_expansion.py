"""Fixed dual-weight expansion with a reusable closed-Gram residual backend.

The V4 coefficient code remains unchanged. Cosine refinement is tried for every
ordinary callable; the same adaptive-integral fallback applies to every failure.
"""
import time,math
import numpy as np
from automatic_expansion import AutomaticExpansion,EvaluationBudgetExceeded
from closed_gram import spectral_covariance_norm2,ClosedCosineGram,evaluate_values

class ReusableExpansion(AutomaticExpansion):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.auxiliary_gram=ClosedCosineGram(float(self.H),float(self.T))

    def covariance_norm2(self,f,tolerance=1e-7,evaluation_budget=2000000):
        start=time.perf_counter()
        spectral=spectral_covariance_norm2(f,float(self.H),float(self.T),
            absolute_tolerance=tolerance*.001,relative_tolerance=1e-4,
            evaluation_budget=evaluation_budget,gram=self.auxiliary_gram)
        if spectral['resolved']:
            return {'value':spectral['value'],'outer_error_estimate':spectral['error_estimate'],
                'integrated_inner_error_estimate':0.,'outer_success':True,
                'integration_warning_count':0,'warnings':[],'backend':'closed_cosine_gram_fft',
                'function_evaluations':spectral['function_evaluations'],'seconds':time.perf_counter()-start,
                'spectral':spectral,'qualification':spectral['qualification']}
        remaining=evaluation_budget-spectral['function_evaluations']
        if remaining<1:raise EvaluationBudgetExceeded('Spectral refinement exhausted the norm evaluation budget')
        fallback=super().covariance_norm2(f,tolerance,remaining)
        return {**fallback,'backend':'adaptive_integral_after_spectral','spectral':spectral,
            'function_evaluations':spectral['function_evaluations']+fallback['function_evaluations'],
            'seconds':time.perf_counter()-start}

    def assess(self,f,expansion,relative_tolerance=.001,norm_tolerance=1e-7,evaluation_budget=2000000,norm=None):
        if expansion.engine is not self:raise ValueError('Expansion belongs to another basis')
        if relative_tolerance<=0:raise ValueError('Positive relative tolerance required')
        residual=lambda t:evaluate_values(f,t)-expansion(t)
        norm=self.covariance_norm2(f,norm_tolerance,evaluation_budget) if norm is None else norm
        error=self.covariance_norm2(residual,norm_tolerance,evaluation_budget)
        n2,e=norm['value'],error['value']
        ne=norm['outer_error_estimate']+norm['integrated_inner_error_estimate']
        ee=error['outer_error_estimate']+error['integrated_inner_error_estimate']
        diagnostic=all(x['integration_warning_count']==0 and x.get('outer_success',True) for x in [norm,error])
        stable=n2>0 and e>=0 and diagnostic and ne<=.001*n2 and ee<=.05*max(e,relative_tolerance**2*n2)
        if stable and e+ee<=relative_tolerance**2*(n2-ne):status='tolerance_numerically_resolved'
        elif stable and e-ee>relative_tolerance**2*(n2+ne):status='target_not_reached'
        else:status='residual_quadrature_unresolved'
        return {'status':status,'relative_error':math.sqrt(e/n2) if n2>0 and e>=0 else None,
            'relative_tolerance':relative_tolerance,'norm':norm,'residual_runs':[error],
            'refinement_stable':bool(stable),
            'qualification':'Numerical decision with estimated-error margins and auxiliary Fourier refinement; not an interval certificate.'}

    def fit(self,f,relative_tolerance=.001,dimensions=(5,9,17,33,65),coefficient_tolerance=1e-8,evaluation_budget=2000000):
        """Screen with input-error estimates, then check the residual directly."""
        if relative_tolerance<=0:raise ValueError('Positive relative tolerance required')
        dimensions=sorted(set(d for d in dimensions if 1<=d<=self.D))
        if not dimensions:raise ValueError('No permitted candidate dimensions')
        start=time.perf_counter();full=self.expand(f,coefficient_tolerance)
        try:norm=self.covariance_norm2(f,min(1e-8,relative_tolerance**2*.01),evaluation_budget)
        except (EvaluationBudgetExceeded,ArithmeticError,ValueError) as exc:
            return {'status':'norm_quadrature_unresolved','selected_dimension':None,
                'expansion':full.as_dict(),'reason':str(exc),'total_seconds':time.perf_counter()-start}
        if norm['value']<=0:
            return {'status':'input_norm_unresolved','selected_dimension':None,'expansion':full.as_dict(),'norm':norm}
        target2=relative_tolerance**2*norm['value']
        norm_error=norm['outer_error_estimate']+norm['integrated_inner_error_estimate']
        epsilon=max(full.diagnostics['quadrature_error_estimate'],full.diagnostics['coefficient_refinement_difference'])
        uncertainty=norm_error+2*np.linalg.norm(full.coefficients)*epsilon+epsilon*epsilon
        screen=[{'D':D,'subtractive_residual2':float(norm['value']-np.dot(full.coefficients[:D],full.coefficients[:D])),
                 'estimated_input_uncertainty':float(uncertainty)} for D in dimensions]
        candidates=[x['D'] for x in screen if x['subtractive_residual2']<=2*target2+uncertainty]
        if dimensions[-1] not in candidates:candidates.append(dimensions[-1])
        assessments=[];selected=None
        for D in candidates:
            prefix=full.truncate(D)
            try:assessment=self.assess(f,prefix,relative_tolerance,
                norm_tolerance=max(1e-13,.02*target2),evaluation_budget=evaluation_budget,norm=norm)
            except (EvaluationBudgetExceeded,ArithmeticError,ValueError) as exc:
                assessment={'status':'residual_quadrature_unresolved','reason':str(exc)}
            assessments.append({'D':D,**assessment})
            if assessment['status']=='tolerance_numerically_resolved' and full.diagnostics['status']=='coefficients_numerically_resolved':
                selected=prefix;break
        rejected=all(a['status']=='target_not_reached' for a in assessments)
        status='tolerance_numerically_resolved' if selected else ('target_not_reached_in_tested_dimensions' if rejected else 'target_not_resolved_in_budget')
        return {'status':status,'selected_dimension':None if selected is None else len(selected.coefficients),
            'expansion':(full if selected is None else selected).as_dict(),'screening':screen,'assessments':assessments,
            'total_seconds':time.perf_counter()-start,'qualification':'General callable procedure using closed Gram data and estimated-error margins. No rigorous enclosure or minimum-dimension claim.'}
