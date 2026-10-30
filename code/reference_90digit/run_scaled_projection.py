"""90-digit reference values for study B2: the same callable projection in arbitrary precision.

Writes results/scaled_projection_257.json (about 6 minutes on a 12-core workstation);
extract_reference.py then writes results/reference_highprecision_D257.json, read by studies/b2_highprecision_check.py.
Relative to the authors' original driver only the output path and the source-hash location were changed.
"""
from pathlib import Path
import json,time,hashlib
import mpmath as mp
import numpy as np
from reusable_expansion import ReusableExpansion
from general_applications import expand_outputs
from smooth_enrichment import exp_cross
import fractional_projection as fp

HERE=Path(__file__).resolve().parent;OUT=HERE.parents[1]/"results"

def exact_coordinates(engine,kind):
    with mp.workdps(engine.dps):
        H,T,D=engine.H,engine.T,engine.D
        if kind=="singular":
            p=mp.mpf("-.55")
            n2=fp.power_cross(H,p,p,T)
            cos=fp.power_rhs(H,p,D-2,T)
            b=mp.matrix([cos[0],fp.power_cross(H,p,1,T)/T,
                         fp.power_cross(H,p,2,T)/T**2]+list(cos[1:,0]))
        else:
            z=140*mp.pi*1j
            n2=mp.re(-exp_cross(H,z,z,T)/4-exp_cross(H,-z,-z,T)/4+
                     exp_cross(H,z,-z,T)/2)
            b=mp.matrix(D,1)
            b[0]=mp.im(exp_cross(H,z,0,T))/mp.sqrt(T)
            for j in [1,2]:
                b[j]=mp.im(mp.diff(lambda w:exp_cross(H,z,w,T),0,j))/T**j
            for k in range(1,D-2):
                w=1j*k*mp.pi/T
                b[k+2]=mp.sqrt(2/T)/2*mp.im(exp_cross(H,z,w,T)+exp_cross(H,z,-w,T))
        return n2,fp.forward(engine.L,b)

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--terms",type=int,default=257)
    parser.add_argument("--hurst",nargs="+",default=[".5",".75",".95"])
    args=parser.parse_args()
    path=OUT/("scaled_projection_"+str(args.terms)+".json")
    data={"status":"RUNNING","terms":args.terms,"runs":[],"target_relative_rms":.01,
          "qualification":"Same production callable procedure; family-specific identities only validate errors. No universal complexity or convergence-rate claim."}
    start=time.perf_counter()
    def save():
        data["seconds"]=time.perf_counter()-start
        path.write_text(json.dumps(data,indent=2),encoding="utf-8")
    save()
    for h in args.hurst:
        print("PREPARE",h,args.terms,flush=True)
        e=ReusableExpansion(h,args.terms,dps=90)
        print("PREPARED",h,e.setup_seconds,flush=True)
        fs=[("high_frequency",lambda t:np.sin(140*np.pi*t))]
        if float(h)>.55:fs.append(("singular",lambda t:np.asarray(t)**(-.55)))
        batch=expand_outputs(e,[f for _,f in fs],tolerance=1e-8,max_intervals=8192)
        entry={"H":float(h),"setup_seconds":e.setup_seconds,"batch":batch.diagnostics,
               "inputs":[]}
        data["runs"].append(entry);save()
        print("COEFFICIENTS",h,batch.diagnostics["status"],batch.diagnostics["seconds"],flush=True)
        for j,(kind,f) in enumerate(fs):
            st=time.perf_counter();n2,c=exact_coordinates(e,kind)
            rows=[]
            with mp.workdps(90):
                for D in [d for d in [65,129,257,513,1025] if d<=args.terms]:
                    projection_error=n2-mp.fsum(c[k]**2 for k in range(D))
                    coordinate_error=mp.fsum((mp.mpf(str(batch.coefficients[j,k]))-c[k])**2 for k in range(D))
                    assert projection_error>=0
                    actual=projection_error+coordinate_error
                    rows.append({"D":D,"best_projection_error2":float(projection_error),
                        "computed_error2":float(actual),"norm2":float(n2),
                        "relative_rms":float(mp.sqrt(actual/n2)),
                        "coefficient_error_l2":float(mp.sqrt(coordinate_error)),
                        "target_met":bool(actual<=mp.mpf(".0001")*n2)})
            entry["inputs"].append({"input":kind,"reference_seconds":time.perf_counter()-st,
                                    "reference_digits":90,"rows":rows})
            save();print(h,kind,[(r["D"],r["relative_rms"]) for r in rows],flush=True)
    data["status"]="PASS"
    names=["run_scaled_projection.py","general_applications.py","automatic_expansion.py",
           "general_expansion.py","fractional_projection.py","smooth_enrichment.py"]
    data["source_hashes"]={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
    save();print("PASS",path,flush=True)

if __name__=="__main__":main()

