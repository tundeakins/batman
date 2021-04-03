#Adding TTV functionality to Batman
#by Tunde Akinsanmi Feb 2021


import numpy as np
import matplotlib.pyplot as plt

from copy import copy, deepcopy
from collections import namedtuple
from . import TransitModel


__all__ = ['split_transits', 'TTV_TransitModel']


def split_transits(t, P, t_ref=None, flux=None, input_t0s=None, find_peaks=False, 
                   find_peaks_kw={"flux_value": 0.995, "distance": None},
                  show_plot=False):
    
    """
    Funtion to split the transits in data into individual transits so that each transit can have a 
    unique mid transit time t0. Recommended to set show_plot=True to visually ensure that transits are well separated.
    
    Parameters:
    -----------
    
    t : np.array;
        times of the full transit data to be splitted.
    
    P : float;
        Orbital period in same unit as t.
    
    t_ref : float;
        reference time of transit - from literature or visual estimate of a mid-transit time in the data 
        Used to calculate expected time of transits in the data assuming linear ephemerides.
        If you don't trust t_ref to accurately find the transits. Use find_peaks=True to instead find peaks in data.
    
    flux : np.array, (optional); 
        Flux value at time t.

    input_t0s: array, list, (optional);
        split transit using these mid-transit times


    show_plot: bool;
        set true to plot the data and show split points. Requires flux to be given. 
        
    find_peaks: Bool;
        Set to True to use scipy's find_peaks to find mid transit times. t_ref is not required if find_peaks is being used to identify transits.
    
    find_peaks_kw: dict;
        Keyword arguments to pass to find_peaks function. Takes the following keys:
        "flux_value" - the value of the flux below which peaks can be identified
        "distance" - optional, float; distance between neighboring peaks in number of samples. 
        Default is to estimate the number of samples within a Period by calculating the mean cadence.
        "distance": P/np.mean(np.diff(t))
        
    Returns:
    --------
    
    tr_times, tr_edges, indices, t0s ;
        tr_time : list of array of the different transits
        tr_edges : end times of each time array
        indices : index values for each time in array
        t0s : list of mid transit times in data.
    """
    if find_peaks:
        from scipy.signal import find_peaks as fp
        assert flux is not None, f"finding peaks in data requires inputting the flux array at times t"
        assert len(flux) == len(t), "t and flux need to have same length"
        assert np.sort(t) == t, "input t has to be sorted"
        
        
        if "distance" not in find_peaks_kw or find_peaks_kw["distance"] is None:
            find_peaks_kw["distance"] = P/np.mean(np.diff(t))
        
        peaks,_ = fp(-1*flux, height=(-find_peaks_kw["flux_value"], None), distance = find_peaks_kw["distance"])
        t0s = t[peaks]
    
    elif input_t0s is not None:
        t0s = list(input_t0s)

    else:
        tref = t_ref
        if t_ref < t.min() or t.max() < t_ref:        #if reference time t0 is not within this timeseries
            #find transit time that falls around middle of the data
            ntrans = int((np.median(t) - tref)/P)   
            tref = t_ref + ntrans*P
            
        nt = int( (tref-t.min())/P )                            #how many transits behind tref is the first transit
        tr_first = tref - nt*P                                    #time of first transit in data
        tr_last = tr_first + int((t.max() - tr_first)/P)*P        #time of last transit in data
    
        n_tot_tr = int((tr_last - tr_first)/P)                  #total nmumber of transits in data_range
        t0s = [tr_first + P*n for n in range(n_tot_tr+1) ]        #expected tmid of transits in data (if no TTV)
        #remove tmid without sufficient transit data around it
        t0s = list(filter(lambda t0: ( t[ (t<t0+0.1*P) & (t>t0-0.1*P)] ).size>0, t0s))

    
    #split data into individual transits. taking points around each tmid    
    tr_times= []
    indz = []
    for i in range(len(t0s)):
        if i==0: 
            tr_times.append(t[( t<(t0s[i]+0.5*P) )])
            indz.append( np.argwhere(t<(t0s[i]+0.5*P)).reshape(-1) )
            #print("doing 0")
        elif i == len(t0s)-1: 
            tr_times.append( t[ t>(tr_times[i-1][-1]) ])
            indz.append( np.argwhere(  t>(tr_times[i-1][-1]) ).reshape(-1) )
            #print("doing last")
        else: 
            tr_times.append( t[( t>(tr_times[i-1][-1]) ) & ( t<(t0s[i]+0.5*P) )] )
            indz.append( np.argwhere( ( t>(tr_times[i-1][-1]) ) & ( t<(t0s[i]+0.5*P) ) ).reshape(-1))
            #print(f"doing middle {i}")    
    
    tr_edges = [tr_t[-1] for tr_t in tr_times]    #take last time in each timebin as breakpts
    
    if show_plot:
        assert flux is not None, f"plotting requires input flux"
        plt.figure(figsize=(15,3))
        plt.plot(t,flux,".")
        for edg in tr_edges: plt.axvline(edg, ls="dashed", c="k", alpha=0.3)
        plt.plot(t0s, (0.997*np.min(flux))*np.ones_like(t0s),"k^")
        plt.xlabel("Time (days)", fontsize=14)
        if find_peaks: plt.title("Using find_peaks: dashed vertical lines = transit splitting times;  triangles = identified transits");
        else: plt.title("Using t_ref: dashed vertical lines = transit splitting times;  triangles = identified transits");

    assert len(np.concatenate(tr_times)) == len(t), "Problem with splitting, len(concatenate(split_time)) != len(t)"

    return tr_times, tr_edges, indz, t0s


'''
def TTV_TransitModel(params, time, flux = None, find_peaks=False,find_peaks_kw=None,
                    max_err=1.0, nthreads = 1, fac = None, transittype = "primary",
                    supersample_factor = 1, exp_time = 0., debug=False):
    """
    Funtion to model transits with TTV. It runs the usual batman TransitModel on data splitted into several arrays 
    each consisting of  a single transit. The data is splitted using the `split_transits` function which by default
    uses a given reference time (params.t_ref) or alternatively scipy's `find_peaks` function to identify transits.
    Ensure you run the `split_transits` function setting show_plots=True to be certain the splitting is reasonable.

    Paramters:
    ---------
    params : object;
        object containing the physical parameters of the transit. An instance of `TransitParams`

    time : np.array;
        Array of times at which to calculate transit model

    max_err: float, optional;
        Error tolerance (in parts per million) for the model.
	
	nthreads: int, optional;
        Number of threads to use for parallelization. 

	fac: float, optional
        Scale factor for integration step size

	transittype: string, optional
        Type of transit ("primary" or "secondary")

	supersample_factor:	integer, optional
        Number of points subdividing exposure

	exp_time: double, optional
        Exposure time (in same units as `t`)
	
    Example:
    --------
	
	>>> flux = batman.TTV_TransitModel(params, max_err = 0.5, nthreads=4)

    """
    if params.split_time is True:
        tr_times, _, _, _ = split_transits(time, params.per, params.t_ref, flux, find_peaks, find_peaks_kw)   #params.t_ref #break data into time bins 
        time_array_used = "splitting time within function"
    else:
        tr_times = time
        time_array_used = "using already splitted input time array"
    
    if debug: print(time_array_used)

    flux_batman = []
    for i, t0 in enumerate(params.t0):
        batparams = deepcopy(params)
        batparams.t0 = t0

        m = TransitModel(batparams, tr_times[i], max_err=max_err, nthreads = nthreads, fac = fac, 
                        transittype=transittype, supersample_factor=supersample_factor, exp_time=exp_time)   #initializes model
        flux_batman.append(m.light_curve(batparams) ) #calculates light curve
        
        
    return np.concatenate(flux_batman)
'''    


def TTV_TransitModel(params, time, flux = None, find_peaks=False,find_peaks_kw=None,
                    max_err=1.0, nthreads = 1, fac = None, transittype = "primary",
                    supersample_factor = 1, exp_time = 0., debug=False):
    """
    Function to model transits with TTV. It runs the usual batman.TransitModel on data splitted into several arrays 
    each consisting of  a single transit. The data is splitted using the `split_transits` function which by default
    uses a given reference time (params.t_ref) or alternatively scipy's `find_peaks` function to identify transits.
    Ensure you run the `split_transits` function setting show_plots=True to be certain the splitting is reasonable.

    Paramters:
    ---------
    params : object;
        object containing the physical parameters of the transit. An instance of `TransitParams`

    time : np.array;
        Array of times at which to calculate transit model

    max_err: float, optional;
        Error tolerance (in parts per million) for the model.
	
	nthreads: int, optional;
        Number of threads to use for parallelization. 

	fac: float, optional
        Scale factor for integration step size

	transittype: string, optional
        Type of transit ("primary" or "secondary")

	supersample_factor:	integer, optional
        Number of points subdividing exposure

	exp_time: double, optional
        Exposure time (in same units as `t`)

    Returns:
    --------
    m : object;
        object containing transit model for individual transits in the data and a function for generating the model light curves given some parameters
	
    Example:
    --------	
	>>> m = batman.TTV_TransitModel(params, time, max_err = 0.5, nthreads=4)

    """
    ttv_model = namedtuple("ttv_model", ['models', 'ttv_light_curve'])

    if params.split_time is True:
        assert params.t_ref is not None, "Set a value for params.t_ref"
        tr_times, _, _, _ = split_transits(time, params.per, params.t_ref, flux, find_peaks, find_peaks_kw)   #params.t_ref #break data into time bins 
        time_array_used = "splitting time within function"
    else:
        tr_times = time
        time_array_used = "using already splitted input time array"
    
    if debug: print(time_array_used)

    
    models = []
    assert np.iterable(params.t0) and len(params.t0)>1, f"params.t0 has to be an iterable with length > 1"
    for i, t0 in enumerate(params.t0):
        assert isinstance(tr_times, list), f"input time is {type(tr_times)}. It should be a list of time arrays for each transit. Set params.split_time to True if you want to split the transits. Or use batman.split_transits function to split the time before input"
        assert isinstance(tr_times[i], np.ndarray), f"elements of the time list should be of type numpy.ndarray and not {type(tr_times[i])}"
        
        batparams = deepcopy(params)
        batparams.t0 = t0

        m = TransitModel(batparams, tr_times[i], max_err=max_err, nthreads = nthreads, fac = fac, 
                        transittype=transittype, supersample_factor=supersample_factor, exp_time=exp_time)   #initializes model
        models.append(m)
        
    ttv_model.models = models
    
    def ttv_light_curve(params):
        """
        Calculates the light curve for each transit using initialized TTV_models from TTV_TransitModel.
        After the TTV model has be initialized using TTV_TransitModel(), ttv_light_curve can be called with new parameters without re-initializing.

        Parameters:
        -----------
        params: object;
            A `TransitParams` instance containing the transit parameters

        Returns:
        --------
        flux : ndarray;
            Returns the transit model flux.

        Example:
        ---------
        >>> m = batman.TTV_TransitModel(params, time, max_err = 0.5, nthreads=4)
        >>> flux = m.ttv_light_curve(params)

        """
        flux_batman = []
        assert np.iterable(params.t0) and len(params.t0)>1, f"params.t0 has to be an iterable with length > 1"
        for i, t0 in enumerate(params.t0):
            bat_params = deepcopy(params)
            bat_params.t0 = t0

            flux_batman.append(ttv_model.models[i].light_curve(bat_params))
        
        return np.concatenate(flux_batman)

    ttv_model.ttv_light_curve = ttv_light_curve  
       
    return ttv_model 

'''
def TTV_light_curve(TTV_model, params):
    """
    Calculates the light curve for each transit using initializes TTV_models from TTV_TransitModel.

    Parameters:
    -----------
    TTV_model : list;
        List of same length as params.t0. It contains transit models for each transit in the data.

    params: object;
        A `TransitParams` instance containing the transit parameters

    Returns:
    --------
    relative flux: ndarray;
        a contatentated array of all the modeled transits in the data.

    Example:

    TTV_model = batman.TTV_TransitModel2(params, time)

    flux = batman.light_curve(TTVmodel, params)
    """

    flux_batman = []
    for i, t0 in enumerate(params.t0):
        batparams = deepcopy(params)
        batparams.t0 = t0

        flux_batman.append(TTV_model[i].light_curve(batparams))
    
    return np.concatenate(flux_batman)

'''
