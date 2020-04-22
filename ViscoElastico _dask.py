import numpy as np
from argparse import ArgumentParser
from dask.distributed import Client, wait, LocalCluster
from devito import configuration
from devito.logger import info
from examples.seismic.viscoelastic import ViscoelasticWaveSolver
from examples.seismic import demo_model, setup_geometry, plot_shotrecord
import time
import random
import os
from random import randint

solver = None
nshots = 10


def kill_a_worker(cluster):
    non_preemptible_workers = []
    preemptible_workers = [
        w.pid for w in cluster.scheduler.workers.values()
        if w.pid not in non_preemptible_workers]
    if preemptible_workers:
        os.kill(random.choice(preemptible_workers), 15)

def viscoelastic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4,
                       nbl=10, constant=True, **kwargs):

    preset = 'constant-viscoelastic' if constant else 'layers-viscoelastic'

    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    #geometry_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, geometry.src_positions[i, :], 
     #       geometry.t0, geometry.tn, f0=geometry.f0, src_type=geometry.src_type)
    
    # Create solver object to provide relevant operators
    solver = ViscoelasticWaveSolver(model, geometry, space_order=space_order, **kwargs)
    return solver, model

def one_shot_run(src_coordinates, i, autotune):
    
    solver.geometry.src_positions[0, :] = src_coordinates[i, :]
    info("Applying Forward")
    rec1, rec2, v, tau, summary = solver.forward(autotune=autotune)
    a = 2
    return rec1, rec2, summary


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, constant=False, **kwargs):

    # Start Dask cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker = 1, death_timeout=1000)
    client = Client(cluster)

    global solver
    solver, model = viscoelastic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                                space_order=space_order, constant=constant, **kwargs)
    
    src_coordinates = np.empty((nshots, 2), dtype=np.float32)
    src_coordinates[:, 0] = np.linspace(0., solver.model.domain_size[0], num=nshots)
    src_coordinates[:, 1] = 30.
    futures = []
    r = randint(1,nshots)
    inicio=  time.time()

    for i in range(nshots):

        future = client.submit(one_shot_run, src_coordinates, i, autotune)
        futures.append(future)
        
            
    #print("Vai matar processo")
    
    time.sleep(19)
    for j in range(5):               
        kill_a_worker(cluster)

   
    

    wait(futures)
    
    
    for i in range(nshots):
        
        dados = futures[i].result()[0]
        #plot_shotrecord(dados.data, solver.model, solver.geometry.t0, solver.geometry.tn)  
        
    fim =  time.time()
    tempo = fim - inicio
    print("Demorou - ",tempo)
    #client.restart()


    return (summary.gflopss, summary.oi, summary.timings,
            [rec1, rec2, v, tau])


def test_viscoelastic():
    _, _, _, [rec1, rec2, v, tau] = run()
    norm = lambda x: np.linalg.norm(x.data.reshape(-1))
    assert np.isclose(norm(rec1), 12.9297, atol=1e-3, rtol=0)
    assert np.isclose(norm(rec2), 0.42867, atol=1e-3, rtol=0)


if __name__ == "__main__":
    description = ("Example script for a set of viscoelastic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=True, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced", choices=["noop", "advanced"],
                        help="Devito loop engine (DLEE) mode")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    args = parser.parse_args()

    # 2D preset parameters
    if args.dim2:
        shape = (150, 150)
        spacing = (10.0, 10.0)
        tn = 750.0
    # 3D preset parameters
    else:
        shape = (150, 150, 150)
        spacing = (10.0, 10.0, 10.0)
        tn = 1250.0

  

    _, _, _, [rec1, rec2, v, tau]= run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, dle=args.dle,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant,
        dse=args.dse)