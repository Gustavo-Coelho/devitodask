import numpy as np
from examples.seismic import plot_velocity, plot_perturbation
from scipy import ndimage
import devito
from examples.seismic.acoustic import AcousticWaveSolver
from devito import TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource
from examples.seismic import plot_image
from examples.seismic import AcquisitionGeometry
from devito import configuration
from distributed import Client, LocalCluster, wait
from devito import Function
from devito import Function
from examples.seismic import plot_image
import time
from examples.seismic import Receiver
configuration['log-level'] = 'WARNING'


# Configure model presets
from examples.seismic import demo_model


def processing1():
    # Enable model presets here:
    preset = 'layers-isotropic'  # A simple but cheap model (recommended)
    # preset = 'marmousi2d-isotropic'  # A larger more realistic model

    # Standard preset with a simple two-layer model
    if preset == 'layers-isotropic':
        def create_model(grid=None):
            return demo_model('layers-isotropic', origin=(0., 0.), shape=(101, 101),
                            spacing=(10., 10.), nbl=20, grid=grid, nlayers=2)
        filter_sigma = (1, 1)
        nshots = 21
        nreceivers = 101
        t0 = 0.
        tn = 1000.  # Simulation last 1 second (1000 ms)
        f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)


    # A more computationally demanding preset based on the 2D Marmousi model
    if preset == 'marmousi2d-isotropic':
        def create_model(grid=None):
            return demo_model('marmousi2d-isotropic', data_path='../../../../data/',
                            grid=grid, nbl=20)
        filter_sigma = (6, 6)
        nshots = 301  # Need good covergae in shots, one every two grid points
        nreceivers = 601  # One recevier every grid point
        t0 = 0.
        tn = 3500.  # Simulation last 3.5 second (3500 ms)
        f0 = 0.025  # Source peak frequency is 25Hz (0.025 kHz)

        #NBVAL_IGNORE_OUTPUT
   

    # Create true model from a preset
    model = create_model()

    # Create initial model and smooth the boundaries
    model0 = create_model(grid=model.grid)
    model0.vp = ndimage.gaussian_filter(model0.vp.data, sigma=filter_sigma, order=0)

    # Plot the true and initial model and the perturbation between them
    #plot_velocity(model)
    #plot_velocity(model0)
    #plot_perturbation(model0, model)

    #NBVAL_IGNORE_OUTPUT
    # Define acquisition geometry: source
    

    # First, position source centrally in all dimensions, then set depth
    src_coordinates = np.empty((nshots, 2), dtype=np.float32)
    src_coordinates[:, 0] = np.linspace(0., 1000, num=nshots)
    src_coordinates[:, 1] = 30.


    # Define acquisition geometry: receivers

    # Initialize receivers for synthetic and imaging data
    rec_coordinates = np.empty((nreceivers, 2))
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
    rec_coordinates[:, 1] = 30.

    # Geometry

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=.010, src_type='Ricker')
    # We can plot the time signature to see the wavelet
    #geometry.src.show()

    # Plot acquisition geometry
    #plot_velocity(model, source=geometry.src_positions,
                #receiver=geometry.rec_positions[::4, :])

    # Compute synthetic data with forward operator 

    solver = AcousticWaveSolver(model, geometry, space_order=4)
    true_d , _, _ = solver.forward(vp=model.vp)

    # Compute initial data with forward operator 
    smooth_d, _, _ = solver.forward(vp=model0.vp)

    # Define gradient operator for imaging
    

    def ImagingOperator(model, image):
        # Define the wavefield with the size of the model and the time dimension
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)

        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                        save=geometry.nt)
        
        # Defie the wave equation, but with a negated damping term
        eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T

        # Use `solve` to rearrange the equation into a stencil expression
        stencil = Eq(v.backward, solve(eqn, v.backward))
        
        # Define residual injection at the location of the forward receivers
        dt = model.critical_dt
        residual = PointSource(name='residual', grid=model.grid,
                            time_range=geometry.time_axis,
                            coordinates=geometry.rec_positions)    
        res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m)

        # Correlate u and v for the current time step and add it to the image
        image_update = Eq(image, image - u * v)

        return Operator([stencil] + res_term + [image_update],
                        subs=model.spacing_map)


    # Start Dask cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    all_current_workers = [w.pid for w in cluster.scheduler.workers.values()]

    #NBVAL_IGNORE_OUTPUT

    # Prepare the varying source locations
    source_locations = np.empty((nshots, 2), dtype=np.float32)
    source_locations[:, 0] = np.linspace(0., 1000, num=nshots)
    source_locations[:, 1] = 30.

    #plot_velocity(model, source=source_locations)

    #d_obs é o model.data

    def one_shot_rtm(geometry, model0):
        
        # Devito objects for gradient and data residual
        grad = Function(name="grad", grid=geometry.model.grid)
        residual = Receiver(name='rec', grid=geometry.model.grid,
                            time_range=geometry.time_axis, 
                            coordinates=geometry.rec_positions)
        solver = AcousticWaveSolver(geometry.model, geometry, space_order=4)

        # Predicted data and residual
        true_d, u0 = solver.forward(vp=geometry.model.vp, save=True)[0:2]
        smooth_d, _ =  solver.forward(vp=model0.vp, save=True)[0:2]

        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
        residual = smooth_d.data - true_d.data
        
        
        return u0, v, residual



    def multi_shots_rtm(geometry, model0):
        image = Function(name='image', grid=model.grid)
        op_imaging = ImagingOperator(model, image)
        futures = []
        inicio = time.time()
        for i in range(geometry.nsrc):

            # Geometry for current shot
            geometry_i = AcquisitionGeometry(geometry.model, geometry.rec_positions, geometry.src_positions[i, :], 
                geometry.t0, geometry.tn, f0=geometry.f0, src_type=geometry.src_type)
            
            # Call serial FWI objective function for each shot location
            futures.append(client.submit(one_shot_rtm, geometry_i, model0))

        # Wait for all workers to finish and collect function values and gradients
        wait(futures)
        fim = time.time()
        tempo = fim - inicio
        print("Demorou - Dask: ",tempo) 
        for i in range(geometry.nsrc):
            #print('iteração', (i+1))
        # plot_image(np.diff(image.data, axis=1))

            op_imaging(u=futures[i].result()[0], v=futures[i].result()[1], vp=model0.vp, dt=model0.critical_dt, 
                residual=futures[i].result()[2])      


        return image.data

    # Compute FWI gradient for 5 shots
    inicio = time.time()
    print("Começou Processing 1")
    imageData = multi_shots_rtm(geometry, model0)
    c = imageData.data

    fim = time.time()
    tempo = fim - inicio
    client.close()
    #NBVAL_IGNORE_OUTPUT
    #from examples.seismic import plot_image
    print("Demorou: ",tempo)
    # Plot the inverted image
    plot_image(np.diff(imageData, axis=1))

    return imageData



if __name__ == '__main__':
    image1 = processing1()
    c = 2
    