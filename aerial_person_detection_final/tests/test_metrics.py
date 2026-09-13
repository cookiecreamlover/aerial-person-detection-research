from dataclasses import dataclass
from evaluation.metrics import box_iou, match, prf
import math

@dataclass
class B:
    class_id:int; label:str; x1:float; y1:float; x2:float; y2:float

def run():
    p=B(0,'person',0,0,10,10); t=B(0,'person',0,0,10,10)
    assert box_iou(p,t)==1.0
    assert match([p],[t])==(1,0,0)
    assert all(math.isclose(x, 0.8, rel_tol=1e-9) for x in prf(8,2,2))
    print('Comparative metrics tests passed.')
if __name__=='__main__': run()
