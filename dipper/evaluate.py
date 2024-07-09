from dipper import *
from tools import *
import astropy.stats as astro_stats
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')

# TODO: undo this action
import warnings
warnings.filterwarnings('ignore')

# feature evaluation 
column_names = ['Nphot',
    'biweight_scale',
    'frac_above_2_sigma', # in deviation
    'Ndips',
    'rate',
    'chi2dof',
    'skew', 
    'kurtosis',
    'mad',
    'stetson_i',
    'stetson_j',
    'stetson_k',
    'invNeumann',    
    'best_dip_power',
    'best_dip_time_loc',
    'best_dip_start',
    'best_dip_end',
    'best_dip_dt',
    'best_dip_ndet',
    'lc_score']

def half_eval(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, ra_cat, dec_cat, custom_cols=column_names, min_phot=10):
    """Perform half evaluation of the light curve."""
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat,  band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Evaluate biweight location and scale & other obvious statistics
    R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
    adf = adf_tests(mag) # ADF test for stationarity
    chi2dof = chidof(mag) # chi2dof

    # Running deviation
    running_deviation = deviation(mag, mag_err, R, S)

    # Peak detection summary per light curve
    peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=1, pk_2_pk_cut=1)

    return peak_detections


def evaluate_updated(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, custom_cols=column_names, min_phot=10):
    """ Evaluate the time-series features of the light curve, as of May 28, 2024. 

    Parameters
    ----------
    time_cat : array-like
        Time array of the light curve.
    mag_cat : array-like
        Magnitude array of the light curve.
    mag_err_cat : array-like
        Magnitude error array of the light curve.
    flag_cat : array-like
        Flag array of the light curve.
    band_cat : array-like
        Band array of the light curve (supports ZTF-r or ZTF-g).
    custom_cols : list
        List of custom column names.
    min_phot : int
        Minimum number of detections required for evaluation.
    
    Returns
    -------
    pd.Series
        A pandas series containing the evaluated features.

    General Notes
    -------------
    This evaluation function takes in the ZTF (gr) detections and performs a dipper selection function.  
        
    """

    # Summary information
    summary_ = {}
    
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, 
                                    band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Don't evaluate if there are less than 10 detections
    if len(time) < min_phot:
        summary_['Nphot'] = len(time)
        for col in custom_cols[1::]:
            summary_[col] = np.nan
    else:
        # Evaluate biweight location and scale & other obvious statistics
        R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
        chi2dof = chidof(mag) # chi2dof

        # Running deviation
        running_deviation = deviation(mag, mag_err, R, S)

        # Peak detection summary per light curve
        peak_detections = peak_detector(time, running_deviation, power_thresh=4, peak_close_rmv=20, pk_2_pk_cut=20)

        # Calculate other summary statistics
        other_stats = other_summary_stats(time, mag, mag_err, len(mag), R, S)
            
        # If there's no detected peaks or time array is empty or no peaks detected...
        if peak_detections[0]==0 or len(time)==0 or peak_detections[0]==0:
            # If failing; set all values to NaN
            for col in custom_cols:
                summary_[col] = np.nan
            
            # Replace nan's with values
            summary_['Nphot'] = len(time)
            summary_['biweight_scale'] = S

            if len(running_deviation)==0:
                summary_['frac_above_2_sigma'] = 0
            else:
                summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
            
            summary_['Ndips'] = 0
            summary_['rate'] = 0
            summary_['chi2dof'] = chi2dof
            summary_['skew'] = other_stats['skew']
            summary_['kurtosis'] = other_stats['kurtosis']
            summary_['mad'] = other_stats['mad']
            summary_['stetson_i'] = other_stats['stetson_I']
            summary_['stetson_j'] = other_stats['stetson_J']
            summary_['stetson_k'] = other_stats['stetson_K']
            summary_['invNeumann'] = other_stats['invNeumann']
        else: # If there are significant peaks...
            
            # From the r-band data select a good peak...
            bp = best_peak_detector(peak_detections, min_in_dip=3)
            
            # Investigate the g-band data and ensure we see a ~significant~ event 
            g_validate, out_g = False, 0
            
            time_g, mag_g, mag_err_g = prepare_lc(time_cat, mag_cat, mag_err_cat,
                                                   flag_cat, band_cat, band_of_study='g', flag_good=0, q=None, custom_q=False)
            #plt.scatter(time_g, mag_g, c='g', s=10, label='g-band')
            #plt.scatter(time, mag, c='r', s=10, label='r-band')
            
            # minimum number of g-band detections after processing
            if len(time_g) > 10:
                g_validate = True
                
            Rg, Sg = astro_stats.biweight.biweight_location(mag_g), astro_stats.biweight.biweight_scale(mag_g)
            
            running_deviation_g = deviation(mag_g, mag_err_g, Rg, Sg)

            # TODO: major big in the dipper detection pipeline is not as efficient here!
            try:
                best_peak_time = bp['peak_loc'].values[0]
                close_g_dets = np.isclose(best_peak_time, time_g, atol=30) # Selection find nearest point within 20 days

                if sum(close_g_dets)==0:
                    g_validate = False
                    xg = [] # empty array...
                else:
                    close_pair_time = time_g[close_g_dets]
                    close_pair_mag = mag_g[close_g_dets]
                    close_pair_mag_err = mag_err_g[close_g_dets]
                    close_pair_dev = running_deviation_g[close_g_dets]

                    phase_close_g = abs(best_peak_time-close_pair_time) # closest neighbour
                    srt_pairs = np.argsort(phase_close_g) # first index will be the nearest pair to best peak time!!

                    _time_close_g, _mag_close_g, _magerr_close_g, _running_deviation_g = close_pair_time[srt_pairs], close_pair_mag[srt_pairs], close_pair_mag_err[srt_pairs], close_pair_dev[srt_pairs]

                    # final selection!!
                    close_pair_time = _time_close_g[0]
                    close_pair_mag = _mag_close_g[0]
                    close_pair_mag_err = _magerr_close_g[0]
                    close_pair_dev = _running_deviation_g[0]
                    xg = [0]
            except:
                g_validate = False
                xg = [] # empty array...
            
            if (len(xg) == 0) or (g_validate==False): # reject if there's no detections...
                g_validate = False
            else:
                g_validate = True
                # Calculate the significance of this g-band bump...
                out_g = (close_pair_dev-np.nanmean(running_deviation_g))/(np.nanstd(running_deviation_g))
                print (out_g)

            # Impose a 3 sigma cut on the g-band data...
            if g_validate and out_g >3:
        
                _score_ = calc_sum_score(time, mag, mag_err, peak_detections, R, S)

                # If failing; set all values to NaN
                for col in custom_cols:
                    summary_[col] = np.nan

                ######## Final appending data ########
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                summary_['chi2dof'] = chi2dof
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['mad'] = other_stats['mad']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']
                summary_['best_dip_power'] = bp['dip_power'].values[0]
                summary_['best_dip_time_loc'] = bp['peak_loc'].values[0]
                summary_['best_dip_start'] = bp['window_start'].values[0]
                summary_['best_dip_end'] = bp['window_end'].values[0]
                summary_['best_dip_dt'] = bp['average_dt_dif'].values[0]
                summary_['best_dip_ndet'] = bp['N_in_dip'].values[0]
                summary_['lc_score'] = _score_
            
            else:
                # If failing; set all values to NaN
                for col in custom_cols:
                    summary_[col] = np.nan
                    
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                summary_['chi2dof'] = chi2dof
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['mad'] = other_stats['mad']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']
                
    return pd.Series(list(summary_.values()), index=custom_cols)
