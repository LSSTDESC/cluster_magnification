import GCRCatalogs
from GCR import GCRQuery
import numpy as np
from astropy.table import Table
import pandas as pd
from datetime import datetime

halo = "False"

print("Hello! it's "+ str(datetime.now()))

print("halo = ", halo)

#Load cosmoDC2 catalog
extragalactic_cat = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_image')

print('Catalog loaded')
print("it's "+ str(datetime.now()))


if halo == "True":
    filters = ['halo_mass > 0.5e13','is_central==True']
    halo_selection = extragalactic_cat.get_quantities(['halo_mass', 'redshift','ra', 'dec', 'halo_id'],\
                                                filters=filters)
    halo_cat = Table(halo_selection)

    print(len(halo_cat)/1e6)
    print("it's "+ str(datetime.now()))

    sel = halo_cat['redshift']>0
    halo_cat_pd = halo_cat[sel].to_pandas()
    halo_cat_pd.to_hdf('cat_halos.h5' ,key='halo_cat')
    print("halo cat saved")

    exit()

else : 
    
    filters = ['redshift<10.', 'mag_i_lsst<26.', 'halo_id>-1']

    gal_selection = extragalactic_cat.get_quantities(['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst', 'redshift','ra', 'dec', 'shear_1', 'shear_2', 'convergence', 'magnification'], filters=filters)

    gal_cat = Table(gal_selection)

    print(len(gal_cat)/1e6)
    print("it's "+ str(datetime.now()))
    
    #Make redshift selection 
    high_z = gal_cat['redshift']>1.5

    #Make color selection 
    u_min_g = gal_cat['mag_u_lsst'] - gal_cat['mag_g_lsst']
    g_min_r = gal_cat['mag_g_lsst'] - gal_cat['mag_r_lsst']
    r_min_i = gal_cat['mag_r_lsst'] - gal_cat['mag_i_lsst']
    r_min_y = gal_cat['mag_r_lsst'] - gal_cat['mag_y_lsst']
    r_min_z = gal_cat['mag_r_lsst'] - gal_cat['mag_z_lsst']
    #g_min_i = gal_cat['mag_g_lsst'] - gal_cat['mag_i_lsst']


    #LBG = (u_min_g > 0.33 * r_min_i + 0.87) * (r_min_i<0.05)  
    LBGp =   (u_min_g > 0.33 * r_min_i + 0.87) *(r_min_i<0.05)  * (g_min_r<0.6)
    LBGpp =  (u_min_g > 0.33 * r_min_i + 0.87) * (r_min_y<0.1)  
    #Udrop = (u_min_g>1.5) * (g_min_r> -1.) * (g_min_r<1.2) * (1.5*g_min_r < u_min_g-0.75)
    Udropp = (u_min_g>1.5) * (g_min_r> -1.) * (g_min_r<1.2) * (1.5*g_min_r < u_min_g-0.75) * (r_min_z<0.5)

    #CCseq_ll = 2.276 * (r_min_z) - 0.152
    #CCseq_T = -1/2.276 * (r_min_z) - 0.152/2.276**2

    #rzseq_ll = - 0.0248 * gal_cat['mag_z_lsst'] + 1.604

    #red_color1 = g_min_i - CCseq_ll

    #red_color2 = (g_min_i - CCseq_T) / (1 + 1/2.276**2)

    #blue_color1 = (r_min_z - rzseq_ll)
    #blue_color2 = red_color2

    #sel_red1 = (red_color1 < -0.7) * (red_color2 < 4.) * (r_min_z>0.5) 
    #sel_blue1 = ((blue_color1 < -0.8) | ((blue_color2 < 0.5) * (g_min_i < 4)))  * (r_min_z<0.5) 

    #sel_red2 = (red_color1 < -0.8) * (red_color2 < 1.7) * (r_min_z>0.5) 
    #sel_blue2 = ((blue_color1 < -0.9) | ((blue_color2 < 0.3) * (g_min_i < 4)))  * (r_min_z<0.5)
    
    
    #Save hdf files
    print("it's "+ str(datetime.now()))
    print('Cat size :', len(gal_cat[high_z]) )
    highz_cat = gal_cat[high_z].to_pandas()
    highz_cat.to_hdf('cat_highz.h5' ,key='highz')
    print("high_z cat saved") 

    #Save hdf files
    print("it's "+ str(datetime.now()))
    print('Cat size :', len(gal_cat[LBGp]) )
    LBGp_cat = gal_cat[LBGp].to_pandas()
    LBGp_cat.to_hdf('cat_LBGp.h5' ,key='LBGp')
    print("LBGp cat saved")

    print("it's "+ str(datetime.now()))
    print('Cat size :', len(gal_cat[LBGpp]) )
    LBGpp_cat = gal_cat[LBGpp].to_pandas()
    LBGpp_cat.to_hdf('cat_LBGpp.h5' ,key='LBGpp')
    print("LBGpp cat saved")

    print("it's "+ str(datetime.now()))
    print('Cat size :', len(gal_cat[Udropp]) )
    Udropp_cat = gal_cat[Udropp].to_pandas()
    Udropp_cat.to_hdf('cat_Udropp.h5' ,key='Udropp')
    print("Udropp cat saved")

    #print("it's "+ str(datetime.now()))
    #print('Cat size :', len(gal_cat[sel_red1]) )
    #sel_red1_cat = gal_cat[sel_red1].to_pandas()
    #sel_red1_cat.to_hdf('cat_sel_red1.h5' ,key='sel_red1')
    #print("sel_red1 cat saved")

    #print("it's "+ str(datetime.now()))
    #print('Cat size :', len(gal_cat[sel_red2]) )
    #sel_red2_cat = gal_cat[sel_red2].to_pandas()
    #sel_red2_cat.to_hdf('cat_sel_red2.h5' ,key='sel_red2')
    #print("sel_red2 cat saved")

    #print("it's "+ str(datetime.now()))
    #print('Cat size :', len(gal_cat[sel_blue1]) )
    #sel_blue1_cat = gal_cat[sel_blue1].to_pandas()
    #sel_blue1_cat.to_hdf('cat_sel_blue1.h5' ,key='sel_blue1')
    #print("sel_blue1 cat saved")

    #print("it's "+ str(datetime.now()))
    #print('Cat size :', len(gal_cat[sel_blue2]) )
    #sel_blue2_cat = gal_cat[sel_blue2].to_pandas()
    #sel_blue2_cat.to_hdf('cat_sel_blue2.h5' ,key='sel_blue2')
    #print("sel_blue2 cat saved")

    #print("it's "+ str(datetime.now()))
    #print("Done, good job!")