# cluster_magnification
Author : M. Ricci

**Developpement of the use of magnification biais to constrain cluster masses, tests on cosmoDC2.** 

### The main branch :
Contains all notebooks associated with the cluster mass measurements from magnification in cosmoDC2. The results are presented in the DESC Note [Ricci et al. 2021](https://fr.overleaf.com/read/vskkvpmjbrpw).

- The investigation of the background source selection is done in `source_selection_DC2.ipynb`
- The source and lens catalogs are created by running `DC2_catalog_save.py`
- The source number counts slope is measured in `alpha_derivation.ipynb`
- The magnification profiles are computed and investigated in `cosmoDC2_magnification_profiles_in_Mpc.ipynb` and `cosmoDC2_magnification_profiles_in_arcmin.ipynb`
- The cluster selection and binning and the measurements of the magnification profiles are performed in `correlations_halo_sources_Mpc.ipynb` and `correlations_halo_sources_arcmin.ipynb`
- The magnification and magnification bias profiles are compared and the masses derived in `cosmoDC2_comparison_profiles_in_Mpc.ipynb` and `cosmoDC2_comparison_profiles_in_arcmin.ipynb`
- The three source samples are compared in `cosmoDC2_comparison_source_samples.ipynb`

### Data :
The data can be obtained by running the different codes, in the order defined above, in a DESC specific environmnt (to acess the cosmoDC2 catalog). 

### Requirements : 
Running the different notebooks requires in particular the following packages: 
- CLMM (version >= '1.0')
- TreeCorr (version >= '4.2.4')
- CCL
- GCRCatalog

