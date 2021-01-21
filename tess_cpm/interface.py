# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:25:26 2021

@author: jlbus
"""
import pandas as pd
import numpy as np

import os
import shutil

from astroquery.mast import Catalogs
from astroquery.mast import Tesscut

from astropy import units as u
from astropy.coordinates import SkyCoord
#from astropy.wcs import WCS

from astropy.io import fits

from astropy.table import Table
#from astropy.table import Column

#import tess_cpm
#import lightkurve as lk


class cpm_interface:
    """
    Used for downloading TESS data products from MAST.
    """
    
    def __init__(self, tic=None, ra=None, dec=None, download_dir=None):#, products = "all"):
        """
        Takes in TIC and/or RA/Dec, download directory, and product list.
        Updates: 
            - make tic,ra,dec flexible for float/str input
            - make sure download dir is proper format
            - make sure products is "all" or a list
            - specify ResolveError and No Data Products error exceptions
        """
        self.tic  = tic
        self.ra = ra
        self.dec = dec
        self.parent_folder = download_dir
         
        if tic == None:
            
            radii = np.linspace(start = 0.0001, stop = 0.001, num = 19)
            #for i,row in self.df.iterrows():
            tic_found = False
            for rad in radii:
                if tic_found == False:
                    query_string = str(row['ra']) + " " + str(row['dec']) # make sure to have a space between the strings!#SkyCoord(ra = row['ra'], dec = row['dec'], frame = 'icrs') str(row['ra']) + " " + str(row['dec']) # make sure to have a space between the strings!
                    obs_table = Catalogs.query_region(coordinates = query_string, radius = rad*u.deg, catalog = "TIC")
                    obs_df = self.tab2df(obs_table)
                    if len(obs_table['ID']) == 1:
                        tic = obs_table['ID'][0]
                        tic_found = True
                        continue
                    if len(obs_df[obs_df['GAIA'].to_numpy(dtype = 'str') != '']) == 1:
                        temp_obs_df = obs_df[obs_df['GAIA'].to_numpy(dtype = 'str') != '']
                        tic = temp_obs_df['ID'].iloc[0]
                        tic_found = True
                        continue
                    if len(np.unique(obs_df[obs_df['HIP'].to_numpy(dtype = 'str') != '']['HIP'])) == 1:
                        tic = obs_table['ID'][0]
                        tic_found = True
                        continue
            if tic_found == False:
                self.tic = "tic issue"
                self.use_tic = False
            else:
                self.tic = obs_table['ID'][0]
                self.use_tic = True
        else:
            self.use_tic = True
            
        if ra == None:
            query_string = "tic " + self.tic # make sure to have a space between the strings!
            obs_table = Catalogs.query_object(query_string, radius = 0.001*u.deg, catalog = "TIC")
            #obs_df = self.tab2df(obs_table)
            self.ra = obs_table['ra'][0]
            self.dec = obs_table['dec'][0]
            
        download_path = os.path.join(self.parent_folder,str(self.tic))
        
        if os.path.exists(download_path):
            self.download_path = download_path
        else:
            self.download_path = download_path
            os.mkdir(download_path)
            
    def download(self, k=5, n=35, size = 32, exlusion_size = 5, pred_pix_method = "cosine_similarity", save_lc = True, keep_tesscut = False, add_poly = False, poly_scale = 2, poly_num_terms = 4):   
        '''
        This function downloads the TESS Cut for the desired target. If TIC ID 
        is available, it downloads by TIC ID. Otherwise it downloads by ra/dec.
        Then, it will do the desired extraction given the extraction 
        parameters for each available sector.
        Optionally deletes the TESS Cut to save memory.
        Optionally saves the extracted CPM light curves as pickled Pandas 
        dataframe.
        
        To add:
        If list of parameters are provided, loop through all desired CPM 
        extraction methods.

        Parameters
        ----------
        k : TYPE, optional
            Number of contiguous sections to split LC into for the holdout fit
            prediction. The default is 5 for normal targets, 50 for targets 
            with long term astrophysical phenomena (such as Supernova).
        n : TYPE, optional
            Number of pixels to be selected by "pred_pix_method" pixel 
            selection to be used to build the CPM LC. The default is 35 for a 
            32x32 TESS Cut cutout, 64 for a 50x50 TESS Cut cutout.
        size : TYPE, optional
            The size of the TESS Cut cutout. A "size" x "size" cutout will be
            downloaded. The default is 32, possibly 50, max suggested is 100.
        exlusion_size : TYPE, optional
            The "exclusion_size" x "exclusion_size" box centered around the
            target that prevents pixels associated with the target from 
            being picked as one of the 'n' predictor pixels.The default is 5.
        pred_pix_method : TYPE, optional
            Method for selecting the 'n' predictive pixels. Options are 
            "random" for random selection (seed integer is hard coded),
            "cosine_similarity" for selecting pixels with similar trends as 
            the target's trend, and "similar_brightness" for selecting pixels 
            with the most similar flux levels to the target. Note: "similar 
            brightness" method could introduce false variability into resulting
            CPM extraction. The default is "cosine_similarity".
        save_lc : TYPE, optional
            "True" saves the extracted CPM LC as a pickled Pandas dataframe. 
            The default is True.
        keep_tesscut : TYPE, optional
            "True" does not delete the downloaded, large TESS Cut .fits file.
            The default is False.
        add_poly : TYPE, optional
            "True" adds a polynomial model to retain long term astrophysical
            phenomena (such as a supernova). The default is False.
        poly_scale : TYPE, optional
            Scaling factor to help flexibility of polynomial model fit. Larger
            values allow for more flexibility. The default is 2.
        poly_num_terms : TYPE, optional
            Number of polynomial terms for the polynomial model. Includes a 
            constant term, so the degree of teh fit is "poly_num_terms" - 1. 
            The default is 4, for a cubic polynomial fit. 

        Returns
        -------
        The CPM LC pandas dataframe as an attribute of the cpm_interface
        object.

        '''
        
        
        def lc_extract():
            if self.use_tic == True:
                cpm_lc_df_fn = "tic" + str(self.tic) + "_cpm_LC.pkl"
                self.save_fn = cpm_lc_df_fn
            if self.use_tic == False:
                cpm_lc_df_fn = "ra" + str(self.ra) + "_dec" + str(self.dec) + "_cpm_LC.pkl"
                self.save_fn = cpm_lc_df_fn
            
            TESS_cuts = os.listdir(self.download_path)
            cpm_lc_df_list = []
            
            for cut in TESS_cuts:
                sector = cut.split("-")[1][-2:]
                
                temp_cut_fn = os.path.join(self.download_path,cut)
                
                with fits.open(temp_cut_fn, mode="readonly") as hdu:
                    x_cen = int(round(hdu[1].header["1CRPX4"]))
                    y_cen = int(round(hdu[1].header["2CRPX4"]))
                             
                temp_source = tess_cpm.Source(temp_cut_fn, remove_bad=True)            
                temp_source.set_aperture(rowlims=[y_cen-1,y_cen+1], collims=[x_cen-1, y_cen+1])            
                temp_source.add_cpm_model(exclusion_size = exclusion_size, n=n, predictor_method = choose_pred_pix);  
                #_ = s.models[0][0].plot_model() #plot selected pixels 
                if add_poly == True:
                    temp_source.add_poly_model(scale = poly_scale, num_terms = poly_num_terms); 
                    temp_source.set_regs([0.01,0.1])#first in list is for cpm model, second in list is for poly model          
                else:
                    temp_source.set_regs([0.1]) #In the current implementation the value is the reciprocal of the prior variance (i.e., precision) on the coefficients
                temp_source.holdout_fit_predict(k=k)            
                time = temp_source.time            
                flux = temp_source.get_aperture_lc(data_type="cpm_subtracted_flux")            
                sector = np.repeat(a=sector, repeats = len(time))            
                lc_table = Table([time,flux,sector], names = ['time','cpm','sector'])            
                lc_df = lc_table.to_pandas()            
                cpm_lc_df_list.append(lc_df) 
            
            del temp_source
            
            cpm_lc_df = pd.concat(cpm_lc_df_list)
            if save_lc == True:
                if keep_tesscut == False:
                    save_path = os.path.join(self.parent_folder,cpm_lc_df_fn)
                    with open(save_path,'wb') as outfile:
                        pkl.dump(cpm_lc_df,outfile)
                    shutil.rmtree(self.download_path)
                else:
                    save_path = os.path.join(self.download_path,cpm_lc_df_fn)
                    with open(save_path,'wb') as outfile:
                        pkl.dump(cpm_lc_df,outfile)
            self.lc_df = cpm_lc_df
        
        
        # do the downloading
        try:        
            if self.use_tic == True: #if tic is available, get tesscut by tic
                manifest = Tesscut.download_cutouts(size=size, sector=None, path=self.download_path, inflate=True, objectname="TIC " + str(self.tic))
                self.manifest = manifest
            else: #otherwise use the ra/dec coordinates
                cutout_coord = SkyCoord(float(self.ra), float(self.dec), unit="deg")
                manifest = Tesscut.download_products(coordinates=cutout_coord, size = size, sector = None, path=self.download_path, inflate=True)
                self.manifest = manifest
                
            if len(manifest) > 0:
                self.query_success = "success"
            if len(manifest) == 0:
                self.query_success = "fail"
                shutil.rmtree(self.download_path)
        except:
            self.query_success = "fail"
            
        #run all the desired extractions
        if self.query_success == "success":
            self.lc_extract()
            