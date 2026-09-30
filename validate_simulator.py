from pathlib import Path
import sys
import copy
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capde_sim import CAPDESimulator, Sensor, SimulationConfig, Packet


def sensor_one(success=1.0, cls='real-time'):
    return Sensor('T01','test',cls,0.0,1.0,0.99,success,False,1.0,0.0,1.0)


def test_packet_accounting():
    cfg=SimulationConfig(duration_s=2.0)
    s=sensor_one()
    sim=CAPDESimulator([s],cfg,'capde',1.0,123)
    m,_=sim.run()
    assert m.generated == m.delivered + m.missed
    return ('packet_accounting_identity', True, f'{m.generated}={m.delivered}+{m.missed}')


def test_deterministic_delivery():
    cfg=SimulationConfig(duration_s=0.2, posture_bad_start_prob=0.0, load_success_penalty_per_unit=0.0)
    s=sensor_one(success=1.0)
    sim=CAPDESimulator([s],cfg,'capde',1.0,1)
    p=Packet(1,s.sensor_id,s.traffic_class,0.0,1.0)
    s.queue.append(p); s.generated=1
    ok=sim._attempt(s,p,0.02,False)
    assert ok and s.delivered==1 and len(s.queue)==0
    return ('single_node_error_free_delivery', True, 'one attempt -> one delivery')


def test_collision_failure():
    cfg=SimulationConfig(duration_s=0.2)
    s=sensor_one(success=1.0)
    sim=CAPDESimulator([s],cfg,'priority_csma',1.0,1)
    p=Packet(1,s.sensor_id,s.traffic_class,0.0,1.0)
    s.queue.append(p); s.generated=1
    ok=sim._attempt(s,p,0.02,True)
    assert (not ok) and s.collisions==1 and len(s.queue)==1
    return ('collision_consumes_attempt_without_delivery', True, 'collision leaves packet queued')


def test_token_bound():
    cfg=SimulationConfig(duration_s=0.2,token_capacity=2.0,token_replenish=1.0)
    sensors=[sensor_one(cls='emergency'),sensor_one(cls='emergency')]
    sensors[0].sensor_id='E1'; sensors[1].sensor_id='E2'
    for i,s in enumerate(sensors):
        s.base_success=1.0
        p=Packet(i+1,s.sensor_id,'emergency',0.0,1.0)
        s.queue.append(p); s.generated=1; s.recent_emergency_times.append(0.0)
    sim=CAPDESimulator(sensors,cfg,'capde',1.0,2)
    sim._service_capde_frame(0.0,1)
    assert sim.token_uses <= int(cfg.token_capacity)
    assert 0.0 <= sim.token_count <= cfg.token_capacity
    return ('emergency_token_bound', True, f'token_uses={sim.token_uses}, remaining={sim.token_count}')


def test_recovery_reserve():
    cfg=SimulationConfig(duration_s=0.2,recovery_period_frames=1,posture_bad_start_prob=0.0)
    bg=sensor_one(success=1.0,cls='background'); bg.sensor_id='BG'
    rt=sensor_one(success=1.0,cls='real-time'); rt.sensor_id='RT'
    for i,s in enumerate([bg,rt]):
        s.queue.append(Packet(i+1,s.sensor_id,s.traffic_class,0.0,5.0)); s.generated=1
    sim=CAPDESimulator([bg,rt],cfg,'capde',1.0,3)
    sim._service_capde_frame(0.0,1)
    assert bg.delivered==1
    return ('recovery_reserve_services_background', True, 'background queue received reserved opportunity')


def main():
    tests=[test_packet_accounting,test_deterministic_delivery,test_collision_failure,test_token_bound,test_recovery_reserve]
    rows=[]
    for fn in tests:
        try:
            rows.append(fn())
        except Exception as e:
            rows.append((fn.__name__,False,repr(e)))
    df=pd.DataFrame(rows,columns=['test','passed','detail'])
    out=ROOT/'data'/'derived'/'validation_results.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    print(df.to_string(index=False))
    if not df.passed.all():
        raise SystemExit(1)

if __name__=='__main__':
    main()
