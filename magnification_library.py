import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
import scipy.integrate as integrate
from astropy import units as u
from astropy import constants as const
from clmm import Modeling as mod
from clmm import utils 
import scipy.interpolate as itp
from scipy import stats


#____________________utils

def plot_profile(r, profile_vals, profile_label='rho', linestyle=None, label=None):
    plt.loglog(r, profile_vals,linestyle=linestyle, label=label)
    plt.xlabel('r [Mpc]', fontsize='xx-large')
    plt.ylabel(profile_label, fontsize='xx-large')
    
def bin_center(array):
    bin_center = array[:-1] + 0.5*(array[1:]-array[:-1])
    return bin_center

    
#____________________cluster 

def scaled_radius(Delta, mass, z,  cosmo):
    """Return the scaled radius corresponding to a certain halo mass and spherical overdensity contrast wrt to the critical density of the universe.
    parameters=======================
    Delta = spherical overdensity contrast
    mass = halo mass in Msun
    z = redshift of the cluster
    cosmo = astropy cosmology object
    
    return ==========================
    R_c = scaled radius in Mpc
"""
    rho = cosmo.critical_density(z).to(u.Msun/(u.Mpc)**3).value
    R_c = (3.* mass /(4 * np.pi)  *1./ rho * 1/Delta)**(1/3.)
    return R_c

#____________________lensing

def DA_ls_over_DA_s(z1, z2, cosmo):
    "wrapper to return aa fixed value when z1>z2"
    if isinstance(z2, (int, float)):
        if z2>=z1:
            return (cosmo.eval_da_z1z2(z1, z2)/cosmo.eval_da(z2))   
        elif z2<z1:
            return -99 
    elif isinstance(z2, (list, np.ndarray)):
        check = (z2>=z1)
        result = np.ones(z2.size) * -99
        result[check] = (cosmo.eval_da_z1z2(z1, z2[check])/cosmo.eval_da(z2[check]))  
        return result
    

def beta(z_cl, z_s, cosmo):
    """Geometric lensing efficicency  beta = max(0, Dang_ls/Dang_s)  Eq.2 in https://arxiv.org/pdf/1611.03866.pdf"""
    beta = np.heaviside(z_s-z_cl,0) * DA_ls_over_DA_s(z_cl, z_s, cosmo)
    return beta

def beta_s(z_cl, z_s, z_inf, cosmo):
    """Geometric lensing efficicency ratio beta_s =beta(z_s)/beta(z_inf)"""
    beta_s = beta(z_cl,z_s,cosmo) / beta(z_cl,z_inf,cosmo)
    return beta_s

def compute_B_mean(lens_redshift, pdz, cosmo, zmin=None, zmax=4.0, nsteps=1000, delta_z_cut=0.1):
    if zmin==None:
        zmin = lens_redshift + delta_z_cut
    z_int = np.linspace(zmin, zmax, nsteps)
    B_mean = np.nansum( beta(lens_redshift, z_int, cosmo) * pdz(z_int)) / np.nansum(pdz(z_int))
    return B_mean

def compute_Bs_mean(lens_redshift, z_inf, pdz, cosmo,  zmin=None, zmax=4.0, nsteps=1000, delta_z_cut=0.1):
    if zmin==None:
        zmin = lens_redshift + delta_z_cut
    z_int = np.linspace(zmin, zmax, nsteps)
    Bs_mean = np.nansum( beta_s(lens_redshift, z_int, z_inf, cosmo) * pdz(z_int)) / np.nansum(pdz(z_int))
    return Bs_mean

#def theta_einstein(M, z_l, z_s, cosmo):
#    """Einstein radius for a point mass in radian"""
    
#     aexp_cluster = mod._get_a_from_z(z_l)
#     aexp_src = mod._get_a_from_z(z_s)
    
#     D_l = mod.angular_diameter_dist_a1a2(cosmo, aexp_cluster, 1.0)
#     D_s = mod.angular_diameter_dist_a1a2(cosmo, aexp_src, 1.0)
#     D_ls = mod.angular_diameter_dist_a1a2(cosmo, aexp_src, aexp_cluster)
#     beta = D_ls/D_s
#     G_c2 = (const.G/((const.c)**2)).to(u.Mpc/u.Msun).value
#     theta_e = np.sqrt(4*G_c2*M*(beta/D_l))
#     return theta_e / cosmo.h

#-----------------magnification

def mu_wl(kappa): 
    "magnification with WL approximation"
    mu_wl = 1 + 2*kappa
    return mu_wl

#magnification bias : number of lensed source over the number of unlensed source
#beta = slope of the power law luminosity function around the limiting flux of the survey, with N ~ AS^(-beta)
def mu_bias(mu, beta):
    mu_bias = mu**(beta-1)
    return mu_bias

#-----------------SNR 
def compute_source_number_per_bin(rmin, rmax, radial_unit, lens_redshift, source_pdz, source_density, nbins=10, method='evenwidth', cosmo=None, delta_z_cut = 0.1):
    """
    """    
    binedges = utils.make_bins(rmin, rmax, nbins, method=method)
    bin_center = binedges[0:-1] + (binedges[1:]  - binedges[0:-1])/2.
    binedges_arcmin = utils.convert_units(binedges, radial_unit, 'arcmin', lens_redshift, cosmo)
    bin_center_arcmin = binedges_arcmin[0:-1] + (binedges_arcmin[1:]  - binedges_arcmin[0:-1])/2.
    area = (np.pi * (binedges_arcmin[1:]**2 - binedges_arcmin[0:-1]**2))
    
    Ngal = integrate.quad(source_pdz , lens_redshift + delta_z_cut, np.inf)[0] * (source_density * area).value
    
    return bin_center, binedges, Ngal

def modele_determination(bin_center, radial_unit, lens_redshift, mass, profile_type, dict_profile, clmm_cosmo, conc=3.0, delta_mdef=200, zinf=1e10, verbose=True):
    
    """Computes the model at the position of the bin_center. This is not precise enough (biased) when their is only few galaxies per bin. Rather take the mean radius of the galaxies in the bin (not yet implemented).
    'conc', the concentration, can be a float for a fixed value or an array with the same size as the mass in case each concentration apply to a different mass."""
    ###########################
    #PATCH TO AVOID CLMM TO CRASH
    if np.any(mass<0) or np.any(conc<0):
        model_inf = np.ones(bin_center.size)*np.inf
        if verbose:
            print("No negative value allowed")
    ###########################
    else:
    
        if profile_type not in dict_profile.keys():
            print("Wrong profile type")  
            
        rad_Mpc = utils.convert_units(bin_center, radial_unit, 'Mpc', lens_redshift, clmm_cosmo)
        
        if isinstance(mass, (list, tuple, np.ndarray)) and not isinstance(conc, (list, tuple, np.ndarray)) :
            conc = np.ones(len(mass)) * conc
        elif not isinstance(mass, (list, tuple, np.ndarray)) and isinstance(conc, (list, tuple, np.ndarray)):
            mass = np.ones(len(conc)) * mass
            
        if profile_type != "redshift depth contrast":
            
            if isinstance(mass, (list, tuple, np.ndarray)):
                model_inf = np.zeros((rad_Mpc.size, len(mass)))

                for i in range(len(mass)):
                    model_inf[:,i] = dict_profile[profile_type]['model_arg']  * \
                                                dict_profile[profile_type]['model_func'](rad_Mpc, mdelta=mass[i], 
                                                cdelta=conc[i], z_cluster=lens_redshift, z_source=zinf, 
                                                cosmo= clmm_cosmo, 
                                                delta_mdef=delta_mdef, 
                                                halo_profile_model='nfw', 
                                                z_src_model='single_plane')   
            else:    

                model_inf = dict_profile[profile_type]['model_arg']  * \
                                                dict_profile[profile_type]['model_func'](rad_Mpc, mdelta=mass, 
                                                cdelta=conc, z_cluster=lens_redshift, z_source=zinf, 
                                                cosmo= clmm_cosmo, 
                                                delta_mdef=delta_mdef, 
                                                halo_profile_model='nfw', 
                                                z_src_model='single_plane')  


            model = compute_Bs_mean(lens_redshift, zinf, dict_profile[profile_type]['source_pdz'], clmm_cosmo) * model_inf
            
            return model
    
        if profile_type == "redshift depth contrast":
            
            func = dict_profile[profile_type]['source_pdz']
            
            zmin, zmax, nz = 0.001, 5, 10000
            z = np.linspace(zmin, zmax, nz)
            
            if isinstance(mass, (list, tuple, np.ndarray)):
                
                model_z = np.zeros((rad_Mpc.size, len(mass), z.size))     
                
                for i in range(rad_Mpc.size):
                    for j in range(len(mass)):
                        #dict_profile[profile_type]['model_arg']  * \
                        model_z[i,j,:] = dict_profile[profile_type]['model_func'](rad_Mpc[i], mdelta=mass[j], 
                                                    cdelta=conc[j], z_cluster=lens_redshift, z_source=z, 
                                                    cosmo= clmm_cosmo, 
                                                    delta_mdef=delta_mdef, 
                                                    halo_profile_model='nfw', 
                                                    z_src_model='single_plane')**(dict_profile[profile_type]['model_arg']-1)

            else : 

                model_z = np.zeros((rad_Mpc.size, z.size))  
                
                for i in range(rad_Mpc.size):
                        #dict_profile[profile_type]['model_arg']  * \
                    model_z[i,:] = dict_profile[profile_type]['model_func'](rad_Mpc[i], mdelta=mass, 
                                                    cdelta=conc, z_cluster=lens_redshift, z_source=z, 
                                                    cosmo= clmm_cosmo, 
                                                    delta_mdef=delta_mdef, 
                                                    halo_profile_model='nfw', 
                                                    z_src_model='single_plane')**(dict_profile[profile_type]['model_arg']-1)
                    
                    
                        
        ax =  len(np.shape(model_z))-1  
        z_cut = dict_profile[profile_type]['delta_z_cut'] +  lens_redshift
        if len(model_z.shape)>2:
            model_z_cut = model_z[:,:,np.where(z>z_cut)[0]]
        else:
            model_z_cut = model_z[:,np.where(z>z_cut)[0]]
        
        zmean_mu = (np.sum(z[z>z_cut]*func(z[z>z_cut])*model_z_cut, axis=ax)/np.sum(func(z[z>z_cut]) * model_z_cut, axis=ax))
        zmean_tot = np.sum(z[z>z_cut]*func(z[z>z_cut]))/np.sum(func(z[z>z_cut]))
        model = zmean_mu/zmean_tot - 1
            
        return model, model_z


def profile_determination(rmin, rmax, radial_unit, lens_redshift, mass, profile_type, dict_profile, clmm_cosmo, nbins=10, method='evenwidth', conc=3.0, delta_mdef=200, zinf=1e10):

    if profile_type not in dict_profile.keys():
        print("Wrong profile type")        
    
    bin_center, bin_edges, Ngal = compute_source_number_per_bin(rmin, rmax, radial_unit , lens_redshift, dict_profile[profile_type]['source_pdz'], dict_profile[profile_type]['source_density'], nbins=nbins, method=method, cosmo=clmm_cosmo)
    noise = dict_profile[profile_type]['noise_func'](Ngal)
    model = modele_determination(bin_center, radial_unit, lens_redshift, mass, profile_type, dict_profile, clmm_cosmo, conc, delta_mdef, zinf)
    
    if profile_type == "redshift depth contrast":
        return bin_center, bin_edges, noise, model[0], model[1]
    else:
        return bin_center, bin_edges, noise, model


def noise_shear(ngal,s_e):
    return s_e / np.sqrt(ngal)

def noise_mag(ngal):
    return 1 / np.sqrt(ngal)

def noise_zdepth(ngal, zpdf, a=0, b=np.inf):
    return z_std(zpdf, a, b) / np.sqrt(ngal)

def SNR_shear(shear,ngal,s_e):
    SNR_s = shear / noise_shear(ngal,s_e)
    return SNR_s

def SNR_mag(kappa,ngal,alpha):
    SNR_mu = kappa * 2 * abs(alpha - 1) / noise_mag(ngal)
    return SNR_mu

def SNR_ratio(shear,ngal_s,s_e,kappa,ngal_mu,alpha):
    "ratio of SNr of the shear over SNR of the magnification"
    SNR_ratio = SNR_shear(shear,ngal_s,s_e)/SNR_mag(kappa,ngal_mu,alpha)
    return SNR_ratio

#____________________luminosity function

def schechterM(magnitude, phiStar, alpha, MStar): 
    """Schechter luminosity function by magnitudes.""" 
    MStarMinM = 0.4 * (MStar - magnitude) 
    return (0.4 * np.log(10) * phiStar * 10.0**(MStarMinM * (alpha + 1.)) * np.exp(-10.**MStarMinM)) 


def PLE(magnitude, phiStar, alpha, beta, MStar): 
    """double power law as in https://arxiv.org/pdf/1509.05607.pdf""" 
    MStarMinM = 0.4 * (MStar - magnitude) 
    return phiStar / (10.0**(-MStarMinM * (alpha + 1.)) + 10.0**(-MStarMinM * (beta + 1.))) 


def slope(magnitude, alpha, MStar, beta=None,fct="schechter"):
    "slope of dlog10(phi)/dm"
    MStarMinM = 0.4 * (MStar - magnitude) 
    if fct=="schechter":
        slope = 0.4 *(10**MStarMinM - (alpha + 1))
    elif fct=="PLE":
        slope = -0.4 * ((alpha + 1) * 10.0**(-MStarMinM * (alpha + 1.)) + (beta + 1) * 10.0**(-MStarMinM * (beta + 1.))) / (10.0**(-MStarMinM * (alpha + 1.)) + 10.0**(-MStarMinM * (beta + 1.))) 
    else:
        print ("Wrong LF paramerisation fonction")
        slope = np.nan
    return slope

#redshif evolution of the parameters
def LF_param(z, a0, a1, m0, m1, m2=None, b0=None, method="Faber07"):
    betaz = None
    if method=="Ricci18":
        alphaz = a0 * np.log10(1 + z) + a1
        Mstarz = m0 * np.log10(1 + z) + m1
    if method=="Faber07":  
        zp = 0.5
        alphaz = a0 + a1 * (z - zp)
        Mstarz = m0 + m1 * (z - zp)   
    if method =="PLE":
        zp = 2.2
        alphaz, Mstarz, betaz = np.zeros((3, z.size))
        if isinstance(z, np.ndarray):
            alphaz[z<=zp]= a0[0]
            alphaz[z>zp] = a0[1]
            
            betaz[z<=zp] = b0[0]
            betaz[z>zp]  = b0[1]
            
            Mstarz[z<=zp] = m0[0] - 2.5 * (m1[0] * (z[z<=zp] - zp)  + m2[0] * (z[z<=zp] - zp)**2)
            Mstarz[z>zp]  = m0[1] - 2.5 * (m1[1] * (z[z>zp] - zp)  + m2[1] * (z[z>zp] - zp)**2)
        elif z<=zp:
            alphaz = a0[0]
            betaz = b0[1]
            Mstarz = m0[0] - 2.5 * (m1[0] * (z - zp)  + m2[0] * (z - zp)**2)
        else :
            alphaz = a0[0]
            betaz = b0[1]
            Mstarz = m0[1] - 2.5 * (m1[1] * (z - zp)  + m2[1] * (z - zp)**2)
            
    if method =="PLE_LEDE":
        zp = 2.2
        alphaz, Mstarz, betaz = np.zeros((3, z.size))
        if isinstance(z, np.ndarray):
            alphaz[z<=zp]= a0[0] +  a1[0] * (z[z<=zp] -zp)
            alphaz[z>zp] = a0[1] +  a1[1] * (z[z>zp] -zp)
            
            betaz[z<=zp] = b0[0]
            betaz[z>zp]  = b0[1]
            
            Mstarz[z<=zp] = m0[0] - 2.5 * (m1[0] * (z[z<=zp] - zp)  + m2[0] * (z[z<=zp] - zp)**2)
            Mstarz[z>zp]  = m0[1] - 2.5 * (m1[1] * (z[z>zp] - zp)  + m2[1] * (z[z>zp] - zp)**2)
        elif z<=zp:
            alphaz = a0[0] +  a1[0] * (z-zp) 
            betaz = b0[1]
            Mstarz = m0[0] - 2.5 * (m1[0] * (z - zp)  + m2[0] * (z - zp)**2)
        else :
            alphaz = a0[1] +  a1[1] * (z-zp)
            betaz = b0[1]
            Mstarz = m0[1] - 2.5 * (m1[1] * (z - zp)  + m2[1] * (z - zp)**2)            
    return alphaz, Mstarz, betaz


def mlim_to_Mlim(mlim, z, astropy_cosmo_object, Kcorr="simple"):
    dl = astropy_cosmo_object.luminosity_distance(z).to(u.pc).value
    if Kcorr=="simple":
        Kcorr = -2.5*np.log10(1+z)
        Mlim = mlim - 5*np.log10(dl/10) - Kcorr
    else :
        Mlim = mlim - 5*np.log10(dl/10) - Kcorr
    return Mlim


def Mlim_to_mlim(Mlim, z, astropy_cosmo_object, Kcorr="simple"):
    dl = astropy_cosmo_object.luminosity_distance(z).to(u.pc).value
    if Kcorr=="simple":
        Kcorr = -2.5*np.log10(1+z)
        mlim = Mlim + 5*np.log10(dl/10) + Kcorr 
    else :
        mlim = Mlim + 5*np.log10(dl/10) + Kcorr
    return mlim

#____________________redshift dstribution

def pzfxn(z):
    """Redshift distribution function Chang et al. 2013"""
    alpha, beta, z0 = 1.24, 1.01, 0.51
    return (z**alpha)*np.exp(-(z/z0)**beta)

def pdf_z(z, alpha, beta, z0):
    """Redshift distribution function"""
    return (z**alpha)*np.exp(-(z/z0)**beta)

def trunc_pdf(z, alpha, beta, z0, zmin, zmax):
    """Redshift distribution function"""
    return (z**alpha)*np.exp(-(z/z0)**beta)*np.heaviside((z>zmin),0)*np.heaviside((z<zmax),0)

def QSO_pdf_z(z):
    """Redshift distribution function following Chang et al. 2013 arXiv:1305.0793 and fitted on DESI expectations from Table 2.3 and Table 2.7  in https://arxiv.org/pdf/1611.00036.pdf"""
    alpha, beta, z0 = 3.8338536 , 0.93783884 , 0.31593764
    return pdf_z(z, 3.8338536 , 0.93783884 , 0.31593764)

def gaussian(x, a,mu, sig):
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def skewn(z, a, loc, scale):
    """Redshift distribution function"""
    return skewnorm.pdf(z, a, loc, scale)

def trunc_skewn(z, a, loc, scale, zmin, zmax):
    """Redshift distribution function"""
    return skewn(z, a, loc, scale)*np.heaviside((z>zmin),0)*np.heaviside((z<zmax),0)

#mean redshift of the distribution
def z_mean(func, a, b):
    num = integrate.quad(lambda x: func(x)*x, a, b)
    den = integrate.quad(lambda x: func(x), a, b)
    return num[0]/den[0]

#standard deviation of the distribution
def z_std(func, a, b):
    mean_x = z_mean(func, a, b)
    num = integrate.quad(lambda x: func(x)*(x - mean_x)**2, a, b)
    den = integrate.quad(lambda x: func(x), a, b)
    return num[0]/den[0]

def zpdf_from_hist(hist, zmin=0, zmax=10):
    """'hist' must be defined with density=True, stacked=True""" 
    zbinc = np.insert(bin_center(hist[1]), [0, bin_center(hist[1]).size], [zmin,zmax])
    zdf_val = np.insert(hist[0], [0, hist[0].size], [0,0])
    pdf_zsource = itp.interp1d(zbinc, zdf_val)
    return pdf_zsource

def truncated_zpdf(z, zpdz, zmin, zmax=None):

    return zpdz() * np.heaviside((z>zmin),0)*np.heaviside((z<zmax),0)

#_______________mass_fitting

def sum_chi2(data, error, model, step_low=0, step_high=None, cov=False):
    if step_high==None:
        step_high=len(data)
        
    data = data[step_low:step_high]
    model = model[step_low:step_high]
    if cov==True:
        error = error[step_low:step_high, step_low:step_high]
        y = (data - model).T * np.linalg.inv(error) * (data - model)
    else :
        error = error[step_low:step_high]
        y = (data - model)**2/error**2
    return np.sum(y)


def mass_minimization(chi2_val, mass_guess, s = np.array([1.,2.,3.]) ):
    
    if np.sum(np.isnan(chi2_val))==len(mass_guess):
        print('Chi2val contains only nan!')
        eval_mass, eval_mass_min, eval_mass_max, nanarg = np.nan, np.nan, np.nan, np.nan
    else:
        nanarg = np.nanargmin(chi2_val)
        chi2_val_itp_up = itp.interp1d(chi2_val[nanarg :], mass_guess[nanarg :], bounds_error= False)
        chi2_val_itp_low = itp.interp1d(chi2_val[: nanarg + 1], mass_guess[: nanarg + 1], bounds_error= False)

        chi2_eval = stats.chi2.ppf(stats.chi2.cdf(s**2,1), 1)

        eval_mass = mass_guess[np.nanargmin(chi2_val)]
        eval_mass_min = chi2_val_itp_low(np.nanmin(chi2_val) + stats.chi2.ppf(stats.chi2.cdf(s**2,1), 1))
        eval_mass_max = chi2_val_itp_up(np.nanmin(chi2_val)  + stats.chi2.ppf(stats.chi2.cdf(s**2,1), 1))

    return eval_mass, eval_mass_min, eval_mass_max, nanarg