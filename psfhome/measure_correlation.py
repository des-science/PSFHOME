import emcee
import numpy as np
import treecorr
import pickle
import os
import fitsio as fio

def parse_args():

    import argparse
    parser = argparse.ArgumentParser(description='Measure correlation functions of galaxy and PSF.')
    parser.add_argument('--theta_min',
                        default=1.0, type=float,
                        nargs='+',
                        help='minimum separation')
    parser.add_argument('--theta_max',
                        default=200, type=float,
                        nargs='+',
                        help='maximum separation')
    parser.add_argument('--nbins',
                        default=32, type=float,
                        nargs='+',
                        help='number of bins to use')
    parser.add_argument('--gal_shape_file',
                        default='/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V5b/metadetect_desdmv5a_cutsv5_patchesv5b.h5',
                        type=str,
                        help='galaxy catalogue')
    parser.add_argument('--psf_file',
                        default='/global/cfs/cdirs/des/schutt20/catalogs/y6a2_piff_v3_HOMs_v1_rhotau_input_riz.fits',
                        type=str,
                        help='PSF catalogue')
    parser.add_argument('--patch_centers',
                        default='/global/cfs/cdirs/des/y6-shear-catalogs/patches-centers-altrem-npatch200-seed8888.fits',
                        type=str,
                        help='Jackknife patch center file')
    parser.add_argument('--outpath',
                        default='/pscratch/sd/m/myamamot',
                        type=str,
                        help='output path')
    parser.add_argument('--cov_file',
                        default='gp_covariance_psf.txt',
                        type=str,
                        help='Covariance file')
    parser.add_argument('--hdf5', default=True,
                        action='store_const', const=True, help='use of HDF5 file')
    parser.add_argument('--subtract_mean_shear', default=False,
                        action='store_const', const=True, help='whether or not to subtract mean shear')
    parser.add_argument('--var_method',
                        default='bootstrap',
                        type=str,
                        help='Covariance estimation method')
    parser.add_argument('--save_gp_cov', default=True,
                        action='store_const', const=True, help='whether or not to save gp covariance')
    parser.add_argument('--num_corr',
                        default=4, type=int,
                        nargs='+',
                        help='number of correlations to compute')
    parser.add_argument('--parallel', default=False,
                        action='store_const', const=True, help='parallelize the process')
    parser.add_argument('--simulations', default=False, 
                        action='store_const', const=True, help='run on mocks')
    parser.add_argument('--const', default=False, 
                        action='store_const', const=True, help='whether or not to model constant')
    # Which columns in psf catalogue to use
    parser.add_argument('--p1', default='G1_MODEL')
    parser.add_argument('--p2', default='G2_MODEL')
    parser.add_argument('--q1', default='DELTA_G1')
    parser.add_argument('--q2', default='DELTA_G2')
    parser.add_argument('--p41', default='G41_MODEL')
    parser.add_argument('--p42', default='G42_MODEL')
    parser.add_argument('--q41', default='DELTA_G41')
    parser.add_argument('--q42', default='DELTA_G42')

    args = parser.parse_args()

    return args


def read_mdet_h5(datafile, keys, response=False, subtract_mean_shear=False):

    def _get_shear_weights(dat):
        return 1/(0.17**2 + 0.5*(np.array(dat['gauss_g_cov_1_1']) + np.array(dat['gauss_g_cov_2_2'])))
    def _wmean(q,w):
        return np.sum(q*w)/np.sum(w)
    
    import h5py as h5
    f = h5.File(datafile, 'r')
    d = f.get('/mdet/noshear')
    nrows = len(np.array( d['ra'] ))
    formats = []
    for key in keys:
        formats.append('f4')
    data = np.recarray(shape=(nrows,), formats=formats, names=keys)
    for key in keys:  
        if key == 'w':
            data['w'] = _get_shear_weights(d)
        elif key in ('g1', 'g2'):
            data[key] = np.array(d['gauss_'+key[0]+'_'+key[1]])
        else:
            data[key] = np.array(d[key])
    print('made recarray with hdf5 file')
    
    # response correction
    if response:
        d_2p = f.get('/mdet/2p')
        d_1p = f.get('/mdet/1p')
        d_2m = f.get('/mdet/2m')
        d_1m = f.get('/mdet/1m')
        # compute response with weights
        g1p = _wmean(np.array(d_1p["gauss_g_1"]), _get_shear_weights(d_1p))                                     
        g1m = _wmean(np.array(d_1m["gauss_g_1"]), _get_shear_weights(d_1m))
        R11 = (g1p - g1m) / 0.02

        g2p = _wmean(np.array(d_2p["gauss_g_2"]), _get_shear_weights(d_2p))
        g2m = _wmean(np.array(d_2m["gauss_g_2"]), _get_shear_weights(d_2m))
        R22 = (g2p - g2m) / 0.02

        R = (R11 + R22)/2.
        data['g1'] /= R
        data['g2'] /= R

    # mean shear subtraction
    if subtract_mean_shear:
        g1 = _wmean(data["g1"],data["w"])
        g2 = _wmean(data["g2"],data["w"])
        print('mean g1 g2 =(%1.8f,%1.8f)'%(g1, g2))          
        data['g1'] -= g1
        data['g2'] -= g2   

    return data


def get_corr(cat1, cat2, var_method, min_sep=0.5, max_sep=250, nbins=32, ximmode=False):

    """
    Get the correlation functions
    cat1: catalog 1
    cat2: catalog 2
    min_sep: minimum angular separation in arcmin
    max_sep: maximum angular separation in arcmin
    nbins: number of bins
    """

    gg_ij = treecorr.GGCorrelation(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        sep_units="arcmin",
        var_method=var_method,
    )

    gg_ij.process(cat1, cat2)

    if ximmode == True:
        return gg_ij.meanlogr, gg_ij.xim, gg_ij.cov[nbins:, nbins:], gg_ij
    else:
        return gg_ij.meanlogr, gg_ij.xip, gg_ij.cov[:nbins, :nbins], gg_ij
    

def run_mocks(comm, rank, size, num_of_corr, nbins, psf_cat_list, patch_centers, theta_min, theta_max, var_method, outpath):

    import pickle
    print('running correlations for sims...')
    for seed in range(1,201):
        if seed % size != rank:
            continue

        with open('/pscratch/sd/m/myamamot/sample_variance/v5_catalog/seed__fid_'+str(seed)+'.pkl', 'rb') as f:
            d_sim = pickle.load(f)['sources'][0]
        cat2 = treecorr.Catalog(ra=d_sim['ra'], dec=d_sim['dec'], ra_units='deg', dec_units='deg', g1=d_sim['e1'], g2=d_sim['e2'], patch_centers=patch_centers)

        e_gal_1_mean = np.mean(np.array(cat2.g1))
        e_gal_2_mean = np.mean(np.array(cat2.g2))
        egal_mean = np.array([e_gal_1_mean, e_gal_2_mean])

        # measure the psf moments average, e1 and e2
        psf_const1 = np.array(
            [np.mean(psf_cat_list[i].g1) for i in range(num_of_corr)]
        )
        psf_const2 = np.array(
            [np.mean(psf_cat_list[i].g2) for i in range(num_of_corr)]
        )

        gp_corr_xip = np.zeros(shape=(num_of_corr, nbins))
        gp_corr_cov = np.zeros(shape = (num_of_corr, nbins, nbins))
        gp_corr_list = []
        for i in range(num_of_corr):
            print('running gp correlations...', i)
            logr, this_xip, this_cov, gp_corr_item = get_corr(cat2, psf_cat_list[i], var_method, min_sep=theta_min, max_sep=theta_max, 
                                                            nbins=nbins)
            gp_corr_xip[i] = this_xip
            gp_corr_cov[i] = this_cov
            gp_corr_list.append(gp_corr_item)

        slice_ = []
        for i in range(num_of_corr):
            slice_.append(np.arange(0, nbins) + i*2*nbins)
        slice_ = np.array(slice_).reshape(-1)
        # gp_joint_cov = treecorr.estimate_multi_cov(gp_corr_list, var_method)[
        #     slice_, :
        # ][:, slice_]


        # measure the PSF-PSF correlations
        pp_corr_xip = np.zeros(shape=(num_of_corr, num_of_corr, nbins))
        pp_corr_cov = np.zeros(shape=(num_of_corr, num_of_corr, nbins, nbins))

        pp_corr_list = []

        for i in range(num_of_corr):
            for j in range(num_of_corr):
                print('running pp correlations...', i, j)
                logr, this_xip, this_cov, pp_corr_item = get_corr(
                    psf_cat_list[i], psf_cat_list[j], var_method, 
                    min_sep=theta_min, max_sep=theta_max, nbins=nbins)
                pp_corr_cov[i][j] = this_cov
                pp_corr_xip[i][j] = this_xip
                pp_corr_list.append(pp_corr_item)

        slice_ = []
        for i in range(num_of_corr**2):
            slice_.append(np.arange(0, nbins) + i*2*nbins)
        slice_ = np.array(slice_).reshape(-1)

        # pp_joint_cov = treecorr.estimate_multi_cov(pp_corr_list, var_method)[
        #     slice_, :
        # ][:, slice_]

        r = np.exp(logr)

        pp_corr = pp_corr_xip
        pp_corr_cov = pp_corr_cov
        gp_corr = gp_corr_xip

        outpath_sims = os.path.join(outpath, 'sims')
        with open(os.path.join(outpath_sims, "full_correlation_psf_"+str(seed)+".pkl"), 'wb') as f:
            pickle.dump([r, gp_corr, pp_corr,psf_const1, psf_const2, egal_mean], f)

def main():

    """
    This function measures the correlation function between the galaxy-PSF and PSF-PSF
    """
    args = parse_args()

    psf_file = args.psf_file
    gal_shape_file = args.gal_shape_file
    star_weight_file = None
    patch_centers = args.patch_centers
    num_of_corr = args.num_corr
    p1 = args.p1 
    p2 = args.p2
    q1 = args.q1 
    q2 = args.q2
    p41 = args.p41 
    p42 = args.p42
    q41 = args.q41 
    q42 = args.q42

    theta_min = args.theta_min
    theta_max = args.theta_max
    nbins = args.nbins
    var_method = args.var_method

    hdf5 = args.hdf5
    subtract_mean_shear = args.subtract_mean_shear
    save_gp_cov = args.save_gp_cov
    cov_file = args.cov_file
    full_cov_file = "gp_full_covariance_psf.txt"
    outpath = args.outpath
    const = args.const

    if args.parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
    print(rank)

    # read some variables for later use in this function
    num_of_corr = num_of_corr

    if star_weight_file is None:
        l = len(fio.read(psf_file))
        wstar = np.ones(l)
    else:
        wstar = np.load(star_weight_file)

    # load galaxy shape and PSF moments tables
    if hdf5:
        keys = ['ra', 'dec', 'g1', 'g2', 'w']
        gal_data = read_mdet_h5(gal_shape_file, keys, response=True, subtract_mean_shear=subtract_mean_shear)
        cat_egal = treecorr.Catalog(ra=gal_data['ra'], 
                                dec=gal_data['dec'], 
                                ra_units='deg', 
                                dec_units='deg', 
                                g1=gal_data['g1'], 
                                g2=gal_data['g2'], 
                                w=gal_data['w'], 
                                patch_centers=patch_centers)
    else:
        if subtract_mean_shear:
            d_gal = fio.read(gal_shape_file)
            mean_e1 = np.average(d_gal['g1']/d_gal['R_all'], weights=d_gal['w'])
            mean_e2 = np.average(d_gal['g2']/d_gal['R_all'], weights=d_gal['w'])
            print('mean galaxy shapes', mean_e1, mean_e2)
            cat_egal = treecorr.Catalog(ra=d_gal['ra'], 
                                        dec=d_gal['dec'], 
                                        ra_units='deg', 
                                        dec_units='deg', 
                                        g1=d_gal['g1']/d_gal['R_all'] - mean_e1, 
                                        g2=d_gal['g2']/d_gal['R_all'] - mean_e2, 
                                        w=d_gal['w'], 
                                        patch_centers=patch_centers)
        else:
            cat_egal = treecorr.Catalog(
                gal_shape_file,
                w_col="w",
                ra_col="ra",
                dec_col="dec",
                ra_units="deg",
                dec_units="deg",
                g1_col="g1",
                g2_col="g2",
                patch_centers=patch_centers
            )

    # PSF catalog
    cat = fio.read(psf_file)
    cat_epsf = treecorr.Catalog(
        ra=cat['RA'],
        dec=cat['DEC'],
        g1=cat[p1],
        g2=cat[p2],
        ra_units="deg",
        dec_units="deg",
        patch_centers=patch_centers,
        w=wstar
    )
    cat_depsf = treecorr.Catalog(
        ra=cat['RA'],
        dec=cat['DEC'],
        g1=cat[q1],
        g2=cat[q2],
        ra_units="deg",
        dec_units="deg",
        patch_centers=patch_centers,
        w=wstar
    )
    cat_Mpsf = treecorr.Catalog(
        ra=cat['RA'],
        dec=cat['DEC'],
        g1=cat[p41],
        g2=cat[p42],
        ra_units="deg",
        dec_units="deg",
        patch_centers=patch_centers,
        w=wstar
    )
    cat_dMpsf = treecorr.Catalog(
        ra=cat['RA'],
        dec=cat['DEC'],
        g1=cat[q41],
        g2=cat[q42],
        ra_units="deg",
        dec_units="deg",
        patch_centers=patch_centers,
        w=wstar
    )

    gal_cat = cat_egal

    # measure the mean galxy shape, that's the last two data points
    e_gal_1_mean = np.mean(np.array(gal_cat.g1))
    e_gal_1_var = np.var(np.array(gal_cat.g1))
    e_gal_2_mean = np.mean(np.array(gal_cat.g2))
    e_gal_2_var = np.var(np.array(gal_cat.g2))

    egal_mean = np.array([e_gal_1_mean, e_gal_2_mean])

    # define the PSF moments list, [second leakage, second modeling, fourth leakage, fourth modeling]
    psf_cat_list = [cat_epsf, cat_depsf, cat_Mpsf, cat_dMpsf]

    # measure the psf moments average, e1 and e2
    psf_const1 = np.array(
        [np.mean(psf_cat_list[i].g1) for i in range(num_of_corr)]
    )
    psf_const2 = np.array(
        [np.mean(psf_cat_list[i].g2) for i in range(num_of_corr)]
    )

    # measure the galaxy-PSF cross correlations
    if args.simulations:
        run_mocks(comm, rank, size, num_of_corr, nbins, psf_cat_list, patch_centers, theta_min, theta_max, var_method, outpath)
    else:
        gp_corr_xip = np.zeros(shape=(num_of_corr, nbins))
        gp_corr_cov = np.zeros(shape = (num_of_corr, nbins, nbins))
        gp_corr_list = []
        for i in range(num_of_corr):
            print('running gp correlations...', i)
            logr, this_xip, this_cov, gp_corr_item = get_corr(gal_cat, psf_cat_list[i], var_method, min_sep=theta_min, max_sep=theta_max, nbins=nbins)
            gp_corr_xip[i] = this_xip
            gp_corr_cov[i] = this_cov
            gp_corr_list.append(gp_corr_item)

        slice_ = []
        for i in range(num_of_corr):
            slice_.append(np.arange(0, nbins) + i*2*nbins)
        slice_ = np.array(slice_).reshape(-1)

        gp_joint_cov = treecorr.estimate_multi_cov(gp_corr_list, var_method)
        if save_gp_cov:
            np.savetxt(os.path.join(outpath, full_cov_file), gp_joint_cov)
        gp_joint_cov = treecorr.estimate_multi_cov(gp_corr_list, var_method)[
            slice_, :
        ][:, slice_]
        if save_gp_cov:
            if const:
                full_cov = np.zeros((num_of_corr*nbins + 2, num_of_corr*nbins + 2))
                full_cov[:num_of_corr*nbins, :num_of_corr*nbins] = gp_joint_cov
                full_cov[num_of_corr*nbins, num_of_corr*nbins] = e_gal_1_var
                full_cov[num_of_corr*nbins + 1, num_of_corr*nbins + 1] = e_gal_2_var
                np.savetxt(os.path.join(outpath, cov_file), full_cov)
            else:
                np.savetxt(os.path.join(outpath, cov_file), gp_joint_cov)


        # measure the PSF-PSF correlations
        pp_corr_xip = np.zeros(shape=(num_of_corr, num_of_corr, nbins))
        pp_corr_cov = np.zeros(shape=(num_of_corr, num_of_corr, nbins, nbins))

        pp_corr_list = []

        for i in range(num_of_corr):
            for j in range(num_of_corr):
                print('running pp correlations...', i, j)
                logr, this_xip, this_cov, pp_corr_item = get_corr(
                    psf_cat_list[i], psf_cat_list[j], var_method, 
                    min_sep=theta_min, max_sep=theta_max, nbins=nbins)
                pp_corr_cov[i][j] = this_cov
                pp_corr_xip[i][j] = this_xip
                pp_corr_list.append(pp_corr_item)

        slice_ = []
        for i in range(num_of_corr**2):
            slice_.append(np.arange(0, nbins) + i*2*nbins)
        slice_ = np.array(slice_).reshape(-1)

        pp_joint_cov = treecorr.estimate_multi_cov(pp_corr_list, var_method)[
            slice_, :
        ][:, slice_]

        r = np.exp(logr)

        pp_corr = pp_corr_xip
        pp_corr_cov = pp_corr_cov
        gp_corr = gp_corr_xip

        with open(os.path.join(outpath, "full_correlation_psf.pkl"), 'wb') as f:
            pickle.dump([r, gp_corr, pp_corr,  psf_const1, psf_const2, egal_mean, pp_corr_cov, pp_joint_cov, gp_corr_cov ], f)

if __name__ == "__main__":
    main()