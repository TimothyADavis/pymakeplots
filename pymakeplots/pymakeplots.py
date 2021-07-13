# coding: utf-8
import numpy as np
import scipy.interpolate as interpolate
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib
from pymakeplots.sauron_colormap import sauron
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse,Rectangle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse,AnchoredSizeBar
from astropy.coordinates import ICRS
import matplotlib.gridspec as gridspec
from astropy.table import Table
from pafit.fit_kinematic_pa import fit_kinematic_pa
import warnings
from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning
warnings.filterwarnings(action='ignore', category=SpectralCubeWarning, append=True)
warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning, append=True)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def rotateImage(img, angle, pivot):
    padX = [img.shape[0] - pivot[0], pivot[0]]
    padY = [img.shape[1] - pivot[1], pivot[1]]
    padZ = [0,0]
    imgP = np.pad(img, [padY, padX, padZ], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR


class pymakeplots:
    def __init__(self,cube_flat=None,pb=None,cube=None):
        self.galname=None
        self.gal_distance=None
        self.posang=None
        self.vsys=None
        self.moment1=None
        self.rms=None
        self.flat_cube=None
        self.pbcorr_cube=None
        self.spectralcube=None
        self.mask=None
        self.flat_cube_trim=None
        self.pbcorr_cube_trim=None
        self.mask_trim=None
        self.bmaj=None
        self.bmin=None
        self.bpa=None
        self.xcoord,self.ycoord,self.vcoord = None, None, None
        self.xcoord_trim,self.ycoord_trim,self.vcoord_trim = None, None, None
        self.dv=None
        self.cellsize=None
        self.silent=False # rig for silent running if true
        self.bardist=None
        self.rmsfac=3
        self.restfreq=None
        self.obj_ra=None
        self.obj_dec=None
        self.imagesize=None
        self.xc=None
        self.yc=None
        self.all_axes_physical=False
        self.bunit=None
        self.linefree_chans_start, self.linefree_chans_end = 1, 6
        self.chans2do=None
        self.spatial_trim = None
        self.maxvdisp=None
        self.cliplevel=None
        self.fits=False
        self.pvdthick=5.    
        self.flipped=False
        self.make_square=True
        self.useallpixels = False
        self.wcs=None
        
        if (cube != None)&(pb==None)&(cube_flat==None):
            # only one cube given
            self.input_cube_nopb(cube)
            
        if (cube != None)&(pb!=None):
            # pbcorred cube and pb given
            self.input_cube_pbcorr(cube,pb)
        
        if (cube_flat != None)&(pb!=None):
            # flat cube and pb given
            if np.any(self.pbcorr_cube) == None: #check if the user gave all three cubes, in which case this call is redundant
                self.input_cube_flat(cube_flat,pb)  
              
        if (cube != None)&(pb==None)&(cube_flat!=None):
            # pbcorred cube and flat cube given
            self.input_cube_pbcorr_and_flat(cube,cube_flat)        
            
    def vsystrans_inv(self,val):
        return val +self.vsys

    def vsystrans(self,val):
        return val - self.vsys
        
    def ang2pctrans_inv(self,val):
        return val/(4.84*self.gal_distance)

    def ang2pctrans(self,val):
        return val*4.84*self.gal_distance
        
    def ang2kpctrans_inv(self,val):
        return val/(4.84e-3*self.gal_distance)

    def ang2kpctrans(self,val):
        return val*4.84e-3*self.gal_distance    
            
        
    def beam_area(self):
        return (np.pi*(self.bmaj/self.cellsize)*(self.bmin/self.cellsize))/(4*np.log(2))
        
    def input_cube_pbcorr(self,path_to_pbcorr_cube,path_to_pb):
       
       self.pbcorr_cube = self.read_primary_cube(path_to_pbcorr_cube) 
       
       pb,hdr,_= self.read_in_a_cube(path_to_pb)
       if self.flipped: pb=np.flip(pb,axis=2)

       self.flat_cube = self.pbcorr_cube*pb
                          
    
    def input_cube_flat(self,path_to_flat_cube,path_to_pb):
       
       self.flat_cube = self.read_primary_cube(path_to_flat_cube)
       
       pb,hdr,_= self.read_in_a_cube(path_to_pb)
       if self.flipped: pb=np.flip(pb,axis=2)
       
       self.pbcorr_cube = self.flat_cube.copy()*0.0
       self.pbcorr_cube[np.isfinite(pb) & (pb != 0)] = self.flat_cube[np.isfinite(pb) & (pb != 0)] / pb[np.isfinite(pb) & (pb != 0)]
       


    def input_cube_nopb(self,path_to_cube):
       
       self.pbcorr_cube = self.read_primary_cube(path_to_cube) 
       
       self.flat_cube = self.pbcorr_cube
       


    def input_cube_pbcorr_and_flat(self,path_to_pbcorr_cube,path_to_flat_cube):
       
       self.pbcorr_cube = self.read_primary_cube(path_to_pbcorr_cube) 
       
       self.flat_cube,hdr,_ = self.read_in_a_cube(path_to_flat_cube)
       if self.flipped: self.flat_cube=np.flip(self.flat_cube,axis=2)
           
    def smooth_mask(self,cube):
        """
        Apply a Gaussian blur, using sigma = 4 in the velocity direction (seems to work best), to the uncorrected cube.
        The mode 'nearest' seems to give the best results.
        :return: (ndarray) mask to apply to the un-clipped cube
        """
        sigma = 1.5 * self.bmaj / self.cellsize
        smooth_cube = ndimage.uniform_filter(cube, size=[sigma, sigma,4], mode='constant')  # mode='nearest'
        newrms= self.rms_estimate(smooth_cube,0,1) 
        self.cliplevel=newrms*self.rmsfac   
        mask=(smooth_cube > self.cliplevel)
        # print("Clip level:",((3e20*1.6014457E-20*91.9)/(self.bmaj*self.bmin))*self.cliplevel*self.dv)
        # import ipdb
        # ipdb.set_trace()
        return mask      
       
    def get_header_coord_arrays(self,hdr):
        self.wcs=wcs.WCS(hdr)

        maxsize=np.max([hdr['NAXIS1'],hdr['NAXIS2'],hdr['NAXIS3']])

        xp,yp=np.meshgrid(np.arange(0,maxsize),np.arange(0,maxsize))
        zp = xp.copy()
        # import ipdb
        # ipdb.set_trace()
        # #coord = wcs.celestial.pixel_to_world(50, 50)
        try:
            x,y,spectral = self.wcs.all_pix2world(xp,yp,zp, 0)
        except:
            x,y,spectral,_ = self.wcs.all_pix2world(xp,yp,zp, 0,0) ## if stokes axis remains


        x1=np.median(x[0:hdr['NAXIS2'],0:hdr['NAXIS1']],0)
        y1=np.median(y[0:hdr['NAXIS2'],0:hdr['NAXIS1']],1)
        spectral1=spectral[0,0:hdr['NAXIS3']]

        if (hdr['CTYPE3'] =='VRAD') or (hdr['CTYPE3'] =='VELO-LSR') or (hdr['CTYPE3'] =='VOPT'):
            v1=spectral1
            try:
                if hdr['CUNIT3']=='m/s':
                     v1/=1e3
                     #cd3/=1e3
            except:
                 if np.max(v1) > 1e5:
                     v1/=1e3
                     #cd3/=1e3

        else:
           f1=spectral1*u.Hz
           try:
               self.restfreq = hdr['RESTFRQ']*u.Hz
           except:
               self.restfreq = hdr['RESTFREQ']*u.Hz
           v1=f1.to(u.km/u.s, equivalencies=u.doppler_radio(self.restfreq))
           v1=v1.value

        cd3= np.median(np.diff(v1))
        cd1= np.median(np.diff(x1))
        return x1,y1,v1,np.abs(cd1*3600),cd3
        
    # def get_header_coord_arrays(self,hdr,cube_or_mom):
    #    try:
    #        cd1=hdr['CDELT1']
    #        cd2=hdr['CDELT2']
    #
    #    except:
    #        cd1=hdr['CD1_1']
    #        cd2=hdr['CD2_2']
    #
    #    x1=((np.arange(0,hdr['NAXIS1'])-(hdr['CRPIX1']-1))*cd1) + hdr['CRVAL1']
    #    y1=((np.arange(0,hdr['NAXIS2'])-(hdr['CRPIX2']-1))*cd2) + hdr['CRVAL2']
    #
    #    try:
    #        cd3=hdr['CDELT3']
    #    except:
    #        cd3=hdr['CD3_3']
    #
    #
    #    if (hdr['CTYPE3'] =='VRAD') or (hdr['CTYPE3'] =='VELO-LSR'):
    #        v1=((np.arange(0,hdr['NAXIS3'])-(hdr['CRPIX3']-1))*cd3) + hdr['CRVAL3']
    #        try:
    #            if hdr['CUNIT3']=='m/s':
    #                 v1/=1e3
    #                 cd3/=1e3
    #        except:
    #             if np.max(v1) > 1e5:
    #                 v1/=1e3
    #                 cd3/=1e3
    #    else:
    #        f1=(((np.arange(0,hdr['NAXIS3'])-(hdr['CRPIX3']-1))*cd3) + hdr['CRVAL3'])*u.Hz
    #        try:
    #            restfreq = hdr['RESTFRQ']*u.Hz
    #        except:
    #            restfreq = hdr['RESTFREQ']*u.Hz
    #        v1=f1.to(u.km/u.s, equivalencies=u.doppler_radio(restfreq))
    #        v1=v1.value
    #        cd3= v1[1]-v1[0]
    #
    #    return x1,y1,v1,np.abs(cd1*3600),cd3
           

    def set_rc_params(self,mult=1):
        #matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['font.family'] = 'Latin Modern Roman'
        matplotlib.rcParams.update({'font.size': 20*mult})
        matplotlib.rcParams['legend.fontsize'] = 17.5*mult
        matplotlib.rcParams['axes.linewidth'] = 1.5
        matplotlib.rcParams['xtick.labelsize'] = 20*mult
        matplotlib.rcParams['ytick.labelsize'] = 20*mult
        matplotlib.rcParams['xtick.major.size'] = 10
        matplotlib.rcParams['ytick.major.size'] = 10
        matplotlib.rcParams['xtick.major.width'] = 2
        matplotlib.rcParams['ytick.major.width'] = 2
        matplotlib.rcParams['xtick.minor.size'] = 5
        matplotlib.rcParams['ytick.minor.size'] = 5
        matplotlib.rcParams['xtick.minor.width'] = 1
        matplotlib.rcParams['ytick.minor.width'] = 1
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        matplotlib.rcParams['xtick.bottom'] = True
        matplotlib.rcParams['ytick.left'] = True
        #matplotlib.rcParams['ytick.right'] = True
        matplotlib.rcParams["xtick.minor.visible"] = True
        matplotlib.rcParams["ytick.minor.visible"] = True
        #params = {'mathtext.default': 'regular'}
        #matplotlib.rcParams.update(params)
        matplotlib.rcParams['axes.labelsize'] = 20*mult
        
           
    def rms_estimate(self,cube,chanstart,chanend):
        quarterx=np.array(cube.shape[0]/4.).astype(np.int)
        quartery=np.array(cube.shape[1]/4.).astype(np.int)
        return np.nanstd(cube[quarterx*1:3*quarterx,1*quartery:3*quartery,chanstart:chanend])
        
                    
        
    def read_in_a_cube(self,path):
        hdulist=fits.open(path)
        hdr=hdulist[0].header
        cube = np.squeeze(hdulist[0].data.T) #squeeze to remove singular stokes axis if present
        cube[np.isfinite(cube) == False] = 0.0
        
        try:
            if hdr['CASAMBM']:
                beamtab = hdulist[1].data
        except:
            beamtab=None
            
        return cube, hdr, beamtab
        
    def read_primary_cube(self,cube):
        
        ### read in cube ###
        datacube,hdr,beamtab = self.read_in_a_cube(cube)
        
        try:
           self.bmaj=np.median(beamtab['BMAJ'])
           self.bmin=np.median(beamtab['BMIN'])
           self.bpa=np.median(beamtab['BPA'])
        except:     
           self.bmaj=hdr['BMAJ']*3600.
           self.bmin=hdr['BMIN']*3600.
           self.bpa=hdr['BPA']
        

           
        try:
            self.galname=hdr['OBJECT']
        except:
            self.galname="Galaxy"
            
        try:
            self.bunit=hdr['BUNIT']
        except:
            self.bunit="Unknown"
                          
        self.xcoord,self.ycoord,self.vcoord,self.cellsize,self.dv = self.get_header_coord_arrays(hdr)
        
        self.spectralcube= SpectralCube.read(cube).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=self.restfreq)
        
        try:
            self.obj_ra=hdr['OBSRA']
            self.obj_dec=hdr['OBSDEC']
            if (self.obj_ra > np.max(self.xcoord)) or (self.obj_ra < np.min(self.xcoord)) or (self.obj_dec < np.min(self.ycoord)) or (self.obj_dec > np.max(self.ycoord)):
                # obsra/dec given in the headers arent in the observed field! Fall back on medians.
                if not self.silent: print("OBSRA/OBSDEC keywords dont seem correct! Assuming galaxy centre is at pointing centre")
                self.obj_ra=np.median(self.xcoord)
                self.obj_dec=np.median(self.ycoord)
        except:
            self.obj_ra=np.median(self.xcoord)
            self.obj_dec=np.median(self.ycoord)
        
            
            
        
        if self.dv < 0:
            datacube = np.flip(datacube,axis=2)
            self.dv*=(-1)
            self.vcoord = np.flip(self.vcoord)
            self.flipped=True
        datacube[~np.isfinite(datacube)]=0.0
        
        self.rms= self.rms_estimate(datacube,self.linefree_chans_start,self.linefree_chans_end) 
        return datacube
    
    def prepare_cubes(self):
        
        self.clip_cube()
            
        self.mask_trim=self.smooth_mask(self.flat_cube_trim)
        
        
        self.xc=(self.xcoord_trim-self.obj_ra)*(-1) * 3600.
        self.yc=(self.ycoord_trim-self.obj_dec) * 3600. / np.cos(np.deg2rad(self.obj_dec))
        

        if self.gal_distance == None:
            self.gal_distance = self.vsys/70.
            if not self.silent: print("Warning! Estimating galaxy distance using Hubble expansion. Set `gal_distance` if this is not appropriate.")
            
        if self.all_axes_physical:
            self.xc=self.ang2kpctrans(self.xc)
            self.yc=self.ang2kpctrans(self.yc)

            
    
    def make_all(self,pdf=False,fits=False):
        self.set_rc_params()

        fig = plt.figure( figsize=(7*3,14))

        gs0 = gridspec.GridSpec(2, 1, figure=fig)
        gs0.tight_layout(fig)
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[0], wspace=0.0, hspace=1.3)

        ax1 = fig.add_subplot(gs00[0, 0])
        ax2 = fig.add_subplot(gs00[0,1], sharey=ax1)
        ax3 = fig.add_subplot(gs00[0, 2], sharey=ax1)
        textaxes = fig.add_subplot(gs00[0, 3])
        textaxes.axis('off')


        gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=0.3, hspace=1.3)

        ax4 = fig.add_subplot(gs01[0, 0])
        ax5 = fig.add_subplot(gs01[0, 1])

        plt.tight_layout()
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        gs0.tight_layout(fig, rect=[0, 0, 1, 0.97])




        self.make_moments(axes=np.array([ax1,ax2,ax3]),fits=fits)
        self.make_pvd(axes=ax4,fits=fits)
        self.make_spec(axes=ax5,fits=fits)
        
        ### plotting PA on mom1
        ypv=self.yc
        xpv=self.yc*0.0
        ang=self.posang
        c = np.cos(np.deg2rad(ang))
        s = np.sin(np.deg2rad(ang))
        x2 =  c*xpv - s*ypv
        y2 =  s*xpv + c*ypv
        ax2.plot(x2,y2,'k--')


        ###### make summary box
        
        rjustnum=35

        c = ICRS(self.obj_ra*u.degree, self.obj_dec*u.degree)

        thetext = (self.galname)+'\n \n'
        thetext += (("RA: "+c.ra.to_string(u.hour, sep=':')))+'\n'
        thetext += ("Dec: "+c.dec.to_string(u.degree, sep=':', alwayssign=True))+'\n \n'
        thetext += ("Vsys: "+str(int(self.vsys))+" km/s")+'\n'
        thetext += ("Dist: "+str(round(self.gal_distance,1))+" Mpc")+'\n'
        if self.gal_distance == self.vsys/70.:
            thetext+=("(Est. from Vsys)")



        at2 = AnchoredText(thetext,
                           loc='upper right', prop=dict(size=30,multialignment='right'), frameon=False,
                           bbox_transform=textaxes.transAxes
                           )
        textaxes.add_artist(at2)



        if pdf:
            plt.savefig(self.galname+"_allplots.pdf", bbox_inches = 'tight')
        else:
            plt.show()

    
  
    
    def make_moments(self,axes=None,mom=[0,1,2],pdf=False,fits=False):
        mom=np.array(mom)
        self.fits=fits
        
        if np.any(self.xc) == None:
            self.prepare_cubes()
        
        self.set_rc_params()
       
        nplots=mom.size
        if np.any(axes) == None:
            if self.make_square:
                fig,axes=plt.subplots(1,nplots,sharey=True,figsize=(7*nplots,7), gridspec_kw = {'wspace':0, 'hspace':0})
            else:
                fig,axes=plt.subplots(1,nplots,sharey=True,figsize=(7*nplots*(self.imagesize[0]/self.imagesize[1]),7), gridspec_kw = {'wspace':0, 'hspace':0})
                
            outsideaxis=0
        else:
            outsideaxis=1
            
            
        if nplots == 1:
            axes=np.array([axes])
        
        for i in range(0,nplots):
            if mom[i] == 0:
                self.mom0(axes[i],first=not i,last=(i==nplots-1))
            if mom[i] == 1:
                self.mom1(axes[i],first=not i,last=(i==nplots-1))
            if mom[i] == 2:
                self.mom2(axes[i],first=not i,last=(i==nplots-1))

        
        if self.gal_distance != None and not self.all_axes_physical:
            self.scalebar(axes[i]) # plot onto last axes

                    
        if pdf:
            plt.savefig(self.galname+"_moment"+"".join(mom.astype(np.str))+".pdf", bbox_inches = 'tight')
        else:
            if not outsideaxis: plt.show()
        
    
    def scalebar(self,ax,loc='lower right'):
        barlength_pc = np.ceil((np.abs(self.xc[-1]-self.xc[0])*4.84*self.gal_distance)/1000.)*100
        barlength_arc=  barlength_pc/(4.84*self.gal_distance)
        
        if barlength_arc > 0.3*(self.xc[-1]-self.xc[0]): # go to 10 pc rounding
            barlength_pc = np.ceil((np.abs(self.xc[-1]-self.xc[0])*4.84*self.gal_distance)/100.)*10
            barlength_arc=  barlength_pc/(4.84*self.gal_distance)

        if barlength_arc > 0.3*(self.xc[-1]-self.xc[0]): # go to 1 pc rounding
            barlength_pc = np.ceil((np.abs(self.xc[-1]-self.xc[0])*4.84*self.gal_distance)/10.)*1
            barlength_arc=  barlength_pc/(4.84*self.gal_distance)
            
            
        
        if np.log10(barlength_pc) > 3:
            label=(barlength_pc/1e3).astype(np.str)+ " kpc"
        else:
            label=barlength_pc.astype(np.int).astype(np.str)+ " pc"
            
        asb = AnchoredSizeBar(ax.transData,  barlength_arc,   label,  loc=loc,  pad=0.25, borderpad=0.5, sep=5, frameon=False)
        ax.add_artist(asb)
        
        
        
        
            
    def clip_cube(self):
        
        if self.chans2do == None:
            # use the mask to try and guess the channels with signal.
            mask_cumsum=np.nancumsum((self.pbcorr_cube > self.rmsfac*self.rms).sum(axis=0).sum(axis=0))
            w_low,=np.where(mask_cumsum/np.max(mask_cumsum) < 0.02)
            w_high,=np.where(mask_cumsum/np.max(mask_cumsum) > 0.98)
            
            if w_low.size ==0: w_low=np.array([0])
            if w_high.size ==0: w_high=np.array([self.vcoord.size])
            self.chans2do=[np.clip(np.max(w_low)-2,0,self.vcoord.size),np.clip(np.min(w_high)+2,0,self.vcoord.size)]
    
        if self.vsys == None:
            # use the cube to try and guess the vsys
            self.vsys=((self.pbcorr_cube*(self.pbcorr_cube > self.rmsfac*self.rms)).sum(axis=0).sum(axis=0)*self.vcoord).sum()/((self.pbcorr_cube*(self.pbcorr_cube > self.rmsfac*self.rms)).sum(axis=0).sum(axis=0)).sum()
        
        if self.imagesize != None:
            if np.array(self.imagesize).size == 1:
                self.imagesize=[self.imagesize,self.imagesize]
            

            wx,=np.where((np.abs((self.xcoord-self.obj_ra)*3600.) <= self.imagesize[0]))
            wy,=np.where((np.abs((self.ycoord-self.obj_dec)*3600.) <= self.imagesize[1]*np.cos(np.deg2rad(self.obj_dec))))
            self.spatial_trim=[np.min(wx),np.max(wx),np.min(wy),np.max(wy)]        
        
        if self.spatial_trim == None:
            
            mom0=(self.pbcorr_cube > self.rmsfac*self.rms).sum(axis=2)
            mom0[mom0>0]=1
            
            cumulative_x = np.nancumsum(mom0.sum(axis=1),dtype=np.float)
            cumulative_x /= np.nanmax(cumulative_x)
            cumulative_y = np.nancumsum(mom0.sum(axis=0),dtype=np.float)
            cumulative_y /= np.nanmax(cumulative_y)
            
            wx_low,=np.where(cumulative_x < 0.02)
            if wx_low.size ==0: wx_low=np.array([0])
            wx_high,=np.where(cumulative_x > 0.98)
            if wx_high.size ==0: wx_high=np.array([cumulative_x.size])
            wy_low,=np.where(cumulative_y < 0.02)
            if wy_low.size ==0: wy_low=np.array([0])
            wy_high,=np.where(cumulative_y > 0.98)
            if wy_high.size ==0: wy_high=np.array([cumulative_y.size])
            

            
            beam_in_pix = np.int(np.ceil(self.bmaj/self.cellsize))
            

            self.spatial_trim = [np.clip(np.max(wx_low) - 2*beam_in_pix,0,self.xcoord.size),np.clip(np.min(wx_high) + 2*beam_in_pix,0,self.xcoord.size)\
                                , np.clip(np.max(wy_low) - 2*beam_in_pix,0,self.ycoord.size), np.clip(np.min(wy_high) + 2*beam_in_pix,0,self.ycoord.size)]
            
            
        self.flat_cube_trim=self.flat_cube[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]]
        self.pbcorr_cube_trim=self.pbcorr_cube[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]]
        #self.mask_trim=self.mask[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]] 
        self.spectralcube=self.spectralcube[self.chans2do[0]:self.chans2do[1],self.spatial_trim[2]:self.spatial_trim[3],self.spatial_trim[0]:self.spatial_trim[1]] 
        self.xcoord_trim=self.xcoord[self.spatial_trim[0]:self.spatial_trim[1]]
        self.ycoord_trim=self.ycoord[self.spatial_trim[2]:self.spatial_trim[3]]
        self.vcoord_trim=self.vcoord[self.chans2do[0]:self.chans2do[1]]  
            
           
            
    

    def colorbar(self,mappable,ticks=None):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cb=fig.colorbar(mappable, cax=cax,ticks=ticks,orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position('top')
        return cb         
            
        
    def add_beam(self,ax):    
        if self.all_axes_physical:
            ae = AnchoredEllipse(ax.transData, width=self.ang2kpctrans(self.bmaj), height=self.ang2kpctrans(self.bmin), angle=self.bpa,
                             loc='lower left', pad=0.5, borderpad=0.4,
                             frameon=False)
        else:
            ae = AnchoredEllipse(ax.transData, width=self.bmaj, height=self.bmin, angle=self.bpa,
                             loc='lower left', pad=0.5, borderpad=0.4,
                             frameon=False)                   
        ae.ellipse.set_edgecolor('black')
        ae.ellipse.set_facecolor('none')
        ae.ellipse.set_linewidth(1.5)
        ax.add_artist(ae)
        
            
        
    def mom0(self,ax1,first=True,last=True):
        mom0=(self.pbcorr_cube_trim*self.mask_trim).sum(axis=2)*self.dv
        
        
        oldcmp = cm.get_cmap("YlOrBr", 512)
        newcmp = ListedColormap(oldcmp(np.linspace(0.15, 1, 256)))
        


        im1=ax1.contourf(self.xc,self.yc,mom0.T,levels=np.linspace(np.nanmin(mom0[mom0 > 0]),np.nanmax(mom0),10),cmap=newcmp)
        
        if self.all_axes_physical:
            ax1.set_xlabel('RA offset (kpc)')
            if first: ax1.set_ylabel('Dec offset (kpc)')
        else:
            ax1.set_xlabel('RA offset (")')
            if first: ax1.set_ylabel('Dec offset (")')
        
        
        maxmom0=np.nanmax(mom0)
        
        
        vticks=np.linspace(0,(np.round((maxmom0 / 10**np.floor(np.log10(maxmom0))))*10**np.floor(np.log10(maxmom0))),4)
        
        cb=self.colorbar(im1,ticks=vticks)
        
        if self.bunit.lower() == "Jy/beam".lower():
            cb.set_label("I$_{\\rm CO}$ (Jy beam$^{-1}$ km s$^{-1}$)")
        if self.bunit.lower() == "K".lower():
            cb.set_label("I$_{\\rm CO}$ (K km s$^{-1}$)")
            
            
        self.add_beam(ax1)
        if self.make_square:
            ax1.set_xlim(np.min([self.xc[0],self.yc[0]]),np.max([self.xc[-1],self.yc[-1]]))
            ax1.set_ylim(np.min([self.xc[0],self.yc[0]]),np.max([self.xc[-1],self.yc[-1]]))
        ax1.set_aspect('equal')
        
        if last and not self.all_axes_physical:
            if np.log10(self.ang2pctrans(np.max([self.xc,self.yc]))) > 3:
                secax = ax1.secondary_yaxis('right', functions=(self.ang2kpctrans, self.ang2kpctrans_inv))
                secax.set_ylabel(r'Dec offset (kpc)')
            else:
                secax = ax1.secondary_yaxis('right', functions=(self.ang2pctrans, self.ang2pctrans_inv))
                secax.set_ylabel(r'Dec offset (pc)')
            
        
        if self.fits:
            self.write_fits(mom0.T,0)
        
        
    def mom1(self,ax1,first=True,last=True):

        mom0=(self.pbcorr_cube_trim*self.mask_trim).sum(axis=2)
        mom1=mom0.copy()*np.nan
        mom1[mom0 != 0.0] = (((self.pbcorr_cube_trim*self.mask_trim)*self.vcoord_trim).sum(axis=2))[mom0 != 0.0]/mom0[mom0 != 0.0]
        
        
        vticks=np.linspace((-1)*np.ceil(np.max(np.abs(self.vcoord_trim-self.vsys))/10.)*10.,np.ceil(np.max(np.abs(self.vcoord_trim-self.vsys))/10.)*10.,5)
        
        im1=ax1.contourf(self.xc,self.yc,mom1.T-self.vsys,levels=self.vcoord_trim-self.vsys,cmap=sauron,vmin=vticks[0],vmax=vticks[-1])
        
        if self.all_axes_physical:
            ax1.set_xlabel('RA offset (kpc)')
            if first: ax1.set_ylabel('Dec offset (kpc)')
        else:
            ax1.set_xlabel('RA offset (")')
            if first: ax1.set_ylabel('Dec offset (")')
        
                
        cb=self.colorbar(im1,ticks=vticks)
        cb.set_label("V$_{\\rm obs}$ - V$_{\\rm sys}$ (km s$^{-1}$)")
        self.add_beam(ax1)
        
        ax1.set_aspect('equal')
        if self.make_square:
            ax1.set_xlim(np.min([self.xc[0],self.yc[0]]),np.max([self.xc[-1],self.yc[-1]]))
            ax1.set_ylim(np.min([self.xc[0],self.yc[0]]),np.max([self.xc[-1],self.yc[-1]]))
        
        if last and not self.all_axes_physical:
            if np.log10(self.ang2pctrans(np.max([self.xc,self.yc]))) > 3:
                secax = ax1.secondary_yaxis('right', functions=(self.ang2kpctrans, self.ang2kpctrans_inv))
                secax.set_ylabel(r'Dec offset (kpc)')
            else:
                secax = ax1.secondary_yaxis('right', functions=(self.ang2pctrans, self.ang2pctrans_inv))
                secax.set_ylabel(r'Dec offset (pc)')    
            
        if self.fits:
            self.write_fits(mom1.T,1)
                

        
    def mom2(self,ax1,first=True,last=True):
        mom0=(self.pbcorr_cube_trim*self.mask_trim).sum(axis=2)
        mom1=mom0.copy()*0.0 +np.nan
        mom1[mom0 != 0.0] = (((self.pbcorr_cube_trim*self.mask_trim)*self.vcoord_trim).sum(axis=2))[mom0 != 0.0]/mom0[mom0 != 0.0]
        mom2=mom1.copy()*0.0
        for i in range(0,self.xcoord_trim.size):
            for j in range(0,self.ycoord_trim.size):
                if mom0[i,j] != 0.0:
                    mom2[i,j]=np.sqrt(np.sum(np.abs(self.pbcorr_cube_trim[i,j,:]*self.mask_trim[i,j,:]) * (self.vcoord_trim - mom1[i,j]) ** 2, axis=0) / np.sum(abs(self.pbcorr_cube_trim[i,j]*self.mask_trim[i,j,:]), axis=0))
        
        if self.maxvdisp == None:
            self.maxvdisp=np.ceil(np.clip(np.nanstd(mom2)*4,0,np.nanmax(mom2))/10.)*10.
        
        
        
        mom2levs=np.linspace(0,self.maxvdisp,10)
        
        im1=ax1.contourf(self.xc,self.yc,mom2.T,levels=mom2levs,cmap=sauron,vmax=self.maxvdisp)
        
        if self.all_axes_physical:
            ax1.set_xlabel('RA offset (kpc)')
            if first: ax1.set_ylabel('Dec offset (kpc)')
        else:
            ax1.set_xlabel('RA offset (")')
            if first: ax1.set_ylabel('Dec offset (")')
        
        if self.maxvdisp < 50:
            dvticks=10
        else:
            dvticks=20    
        if self.maxvdisp > 100:
            dvticks=30    
    
        vticks=np.arange(0,5)*dvticks
        
        cb=self.colorbar(im1,ticks=vticks)
        cb.set_label("$\sigma_{obs}$ (km s$^{-1}$)")

        
        self.add_beam(ax1)
        
        ax1.set_aspect('equal')
        if self.make_square:
            ax1.set_xlim(np.min([self.xc[0],self.yc[0]]),np.max([self.xc[-1],self.yc[-1]]))
            ax1.set_ylim(np.min([self.xc[0],self.yc[0]]),np.max([self.xc[-1],self.yc[-1]]))
            
        if last and not self.all_axes_physical:
            if np.log10(self.ang2pctrans(np.max([np.max(self.xc),np.max(self.yc)]))) > 3.3:
                secax = ax1.secondary_yaxis('right', functions=(self.ang2kpctrans, self.ang2kpctrans_inv))
                secax.set_ylabel(r'Dec offset (kpc)')
            else:
                secax = ax1.secondary_yaxis('right', functions=(self.ang2pctrans, self.ang2pctrans_inv))
                secax.set_ylabel(r'Dec offset (pc)',rotation=270,labelpad=10)
                
                    
        if self.fits:
            self.write_fits(mom2.T,2)
      
      
    def write_fits(self,array,whichmoment):
        if self.fits == True:
            filename=self.galname+"_mom"+"".join(np.array([whichmoment]).astype(np.str))+".fits"
        else:
            filename=self.fits+"_mom"+"".join(np.array([whichmoment]).astype(np.str))+".fits"
        
  
        newhdu = fits.PrimaryHDU(array)

        newhdu.header['CRPIX1']=self.spectralcube.header['CRPIX1']
        newhdu.header['CRVAL1']=self.spectralcube.header['CRVAL1']
        newhdu.header['CDELT1']=self.spectralcube.header['CDELT1']
        newhdu.header['CTYPE1']=self.spectralcube.header['CTYPE1']
        newhdu.header['CUNIT1']=self.spectralcube.header['CUNIT1']
        newhdu.header['CRPIX2']=self.spectralcube.header['CRPIX2']
        newhdu.header['CRVAL2']=self.spectralcube.header['CRVAL2']
        newhdu.header['CDELT2']=self.spectralcube.header['CDELT2']
        try:
            newhdu.header['PV2_1']=self.spectralcube.header['PV2_1']
            newhdu.header['PV2_2']=self.spectralcube.header['PV2_2']
        except:
            pass
            
        try:
            newhdu.header['RADESYS']=self.spectralcube.header['RADESYS']
        except:
            pass
            
        try:
            newhdu.header['SPECSYS'] = self.spectralcube.header['SPECSYS']
        except:
            pass    
        try:
            newhdu.header['LONPOLE']=self.spectralcube.header['LONPOLE']
            newhdu.header['LATPOLE']=self.spectralcube.header['LATPOLE']
        except:
            pass    
        newhdu.header['CTYPE2']=self.spectralcube.header['CTYPE2']
        newhdu.header['CUNIT2']=self.spectralcube.header['CUNIT2']
        newhdu.header['BMAJ']=self.bmaj/3600.
        newhdu.header['BMIN']=self.bmin/3600.
        newhdu.header['BPA']=self.bpa/3600.
        newhdu.header['MOMCLIP']=(self.cliplevel, self.bunit+' km/s')
        newhdu.header['VSYS']=(self.vsys,'km/s')
        newhdu.header['comment'] = 'Moment map created with pymakeplots'

        if whichmoment == 0:
            newhdu.header['BUNIT']=self.bunit+' km/s'
        
        else:
            newhdu.header['BUNIT']='km/s'
            
        if whichmoment == 2:
            newhdu.header['BUNIT']='km/s'        
        
        newhdu.writeto(filename,overwrite=True)
    
    def write_pvd_fits(self,xx,vv,pvd):
        if self.fits == True:
            filename=self.galname+"_pvd.fits"
        else:
            filename=self.fits+"_pvd.fits"
            
        newhdu = fits.PrimaryHDU(pvd)
        newhdu.header['CRPIX1']=1
        newhdu.header['CRVAL1']=xx[0]
        newhdu.header['CDELT1']=xx[1]-xx[0]
        newhdu.header['CTYPE1']='OFFSET'
        newhdu.header['CUNIT1']='arcsec'
        newhdu.header['CRPIX2']=1
        newhdu.header['CRVAL2']=vv[0]
        newhdu.header['CDELT2']=vv[1]-vv[0]
        newhdu.header['CTYPE2']='VRAD'
        newhdu.header['CUNIT2']='km/s'
        newhdu.header['BMAJ']=self.bmaj/3600.
        newhdu.header['BMIN']=self.bmin/3600.
        newhdu.header['BPA']=self.bpa/3600.
        newhdu.header['PVDANGLE']=(self.posang,'deg')
        newhdu.header['PVDTHICK']=(self.pvdthick,'pixels')
        newhdu.header['MOMCLIP']=(self.cliplevel, self.bunit+' km/s')
        newhdu.header['VSYS']=(self.vsys,'km/s')
        newhdu.header['comment'] = 'Moment map created with pymakeplots'
        newhdu.header['BUNIT'] = self.bunit+' km/s'
        
        newhdu.writeto(filename,overwrite=True)   
            
    def make_pvd(self,axes=None,fits=False,pdf=False):
        
        if np.any(self.xc) == None:
            self.prepare_cubes()
        
        if np.any(axes) == None:    
            self.set_rc_params(mult=0.75)   
            fig,axes=plt.subplots(1,figsize=(7,5))
            outsideaxis=0
        else:
            outsideaxis=1
            

        if self.posang==None:
            # try fitting the moment one to get the kinematic pa
            if not self.silent: print("No position angle given, estimating using the observed moment one.")
            mom0=(self.pbcorr_cube_trim*self.mask_trim).sum(axis=2)
            mom1=mom0.copy()*np.nan
            mom1[mom0 != 0.0] = (((self.pbcorr_cube_trim*self.mask_trim)*self.vcoord_trim).sum(axis=2))[mom0 != 0.0]/mom0[mom0 != 0.0]
            mom1=mom1.T
            

            # if the cube is small, use it directly to estimate posang. If its large, then interpolate down to keep runtime low.
            if (self.pbcorr_cube_trim[:,:,0].size < 50*50) or (self.useallpixels):
                xv, yv = np.meshgrid(self.xc,self.yc)
                x,y,v = xv[np.isfinite(mom1)],yv[np.isfinite(mom1)],mom1[np.isfinite(mom1)]
            else:
                print("Downsampling the observed moment one in PA estimate for speed. Set `useallpixels` to override.")    
                mom1[np.isfinite(mom1) == False] = self.vsys
                interper = interpolate.interp2d(self.xc,self.yc,mom1-self.vsys,bounds_error=False,fill_value=np.nan)
                x=np.linspace(np.min(self.xc),np.max(self.xc),50)
                y=np.linspace(np.min(self.yc),np.max(self.yc),50)
                v= interper(x,y)
                xv, yv = np.meshgrid(x,y)
                x,y,v = xv.flatten(),yv.flatten(),v.flatten()
                
            self.posang,_,_ = fit_kinematic_pa(x[np.isfinite(v)],y[np.isfinite(v)],v[np.isfinite(v)],nsteps=36,plot=False,quiet=True)
            
            if np.sin(np.deg2rad((self.posang+45)*2)) > 0:
                # do y axis cut
                if np.nanmean(mom1[self.yc > 0,:]) > np.nanmean(mom1[self.yc < 0,:]):
                    # posang should be gt 180
                    if self.posang < 180: self.posang += 180
                else:
                     # posang should be lt 180
                    if self.posang > 180: self.posang -= 180    
            else:
                # do x axis cut
                if np.nanmean(mom1[:,self.xc > 0]) > np.nanmean(mom1[:,self.xc < 0]):
                    # posang should be gt 180
                    if self.posang < 180: self.posang += 180
                else:
                     # posang should be lt 180
                    if self.posang > 180: self.posang -= 180
            if not self.silent: print("PA estimate (degrees): ",np.round(self.posang,1))        
        
                    
        centpix_x=np.where(np.isclose(self.xc,0.0,atol=self.cellsize/1.9))[0]
        centpix_y=np.where(np.isclose(self.yc,0.0,atol=self.cellsize/1.9))[0]
        

        
        rotcube= rotateImage(self.pbcorr_cube_trim*self.mask_trim,90-self.posang,[centpix_x[0],centpix_y[0]])


        pvd=rotcube[:,np.array(rotcube.shape[1]//2-self.pvdthick).astype(np.int):np.array(rotcube.shape[1]//2+self.pvdthick).astype(np.int),:].sum(axis=1)
        

        
        if self.posang < 180:
            loc1="upper left"
            loc2="lower right"
        else:
            loc1="upper right"
            loc2="lower left"
        
        pvdaxis=(np.arange(0,pvd.shape[0])-pvd.shape[0]/2)*self.cellsize    
        if self.all_axes_physical:
                pvdaxis=self.ang2kpctrans(pvdaxis)
        
            
        vaxis=self.vcoord_trim
        

        pvd=pvd[np.abs(pvdaxis) < np.max([np.max(abs(self.xc)),np.max(abs(self.yc))]),:]
        pvdaxis=pvdaxis[np.abs(pvdaxis) < np.max([np.max(abs(self.xc)),np.max(abs(self.yc))])]
        
     
        
        oldcmp = cm.get_cmap("YlOrBr", 512)
        newcmp = ListedColormap(oldcmp(np.linspace(0.15, 1, 256)))
        

        
        axes.contourf(pvdaxis,vaxis,pvd.T,levels=np.linspace(self.cliplevel,np.nanmax(pvd),10),cmap=newcmp)
        axes.contour(pvdaxis,vaxis,pvd.T,levels=np.linspace(self.cliplevel,np.nanmax(pvd),10),colors='black')
        
        if self.all_axes_physical:
             axes.set_xlabel('Offset (kpc)')
        else:
             axes.set_xlabel('Offset (")')
       
        axes.set_ylabel('Velocity (km s$^{-1}$)')
        
        secax = axes.secondary_yaxis('right', functions=(self.vsystrans, self.vsystrans_inv))
        secax.set_ylabel(r'V$_{\rm offset}$ (km s$^{-1}$)')
        
        

        anchored_text = AnchoredText("PA: "+str(round(self.posang,1))+"$^{\circ}$", loc=loc1,frameon=False)
        axes.add_artist(anchored_text)
        
        if self.gal_distance != None and not self.all_axes_physical:
            self.scalebar(axes,loc=loc2)
        
        if self.fits:
            self.write_pvd_fits(pvdaxis,vaxis,pvd.T)
        
        if pdf:
            plt.savefig(self.galname+"_pvd.pdf", bbox_inches = 'tight')
        else:
            if not outsideaxis: plt.show()
    
    def make_spec(self,axes=None,fits=False,pdf=False,onlydata=False,nsum=False,highlight=False):
        if np.any(self.xc) == None:
            self.prepare_cubes()
        
        if axes == None:    
            self.set_rc_params(mult=0.75)   
            fig,axes=plt.subplots(1,figsize=(7,5))
            outsideaxis=0
        else:
            outsideaxis=1
        spec=self.pbcorr_cube[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],:].sum(axis=0).sum(axis=0)
        spec_mask=(self.pbcorr_cube_trim*self.mask_trim).sum(axis=0).sum(axis=0)
    
        
        if self.bunit.lower() == "Jy/beam".lower():
            
            spec*=1/self.beam_area()
            spec_mask*=1/self.beam_area()
            
            if spec_mask.max() < 1:
                spec*=1e3
                spec_mask*=1e3
                ylab="Flux Density (mJy)"
            else:
                ylab="Flux Density (Jy)"
        if self.bunit == "K":
            ylab="Brightness Temp. (K)"
            
        if nsum:
            spec=np.append(running_mean(spec,nsum),spec[-1])
            spec_mask=np.append(running_mean(spec_mask,nsum),spec_mask[-1])
                
                    
        
        if onlydata:
            axes.step(self.vcoord,spec,c='k')
            
            if np.any(highlight):
                area=(self.vcoord >= highlight[0])&(self.vcoord <= highlight[1])
                plt.fill_between(self.vcoord[area],spec[area], step="pre", alpha=0.6,color='grey')
                
                
        else:
            axes.step(self.vcoord,spec,c='lightgrey',linestyle='--')
            axes.step(self.vcoord_trim,spec_mask,c='k')
            if np.any(highlight):
                area=(self.vcoord_trim >= highlight[0])&(self.vcoord_trim <= highlight[1])
                plt.fill_between(self.vcoord_trim[area],spec_mask[area], step="pre", alpha=0.6,color='grey')
        
        
        axes.axhline(y=0,linestyle='-.',color='k',alpha=0.5)        
        axes.set_xlabel('Velocity (km s$^{-1}$)')
        axes.set_ylabel(ylab)
        if np.log10(np.nanmedian(self.vcoord_trim)) >= 5:
            plt.locator_params(axis='x', nbins=4)


            
            
        secax = axes.secondary_xaxis('top', functions=(self.vsystrans, self.vsystrans_inv))
        secax.set_xlabel(r'V$_{\rm obs}$-V$_{\rm sys}$ (km s$^{-1}$)')
        
        if self.fits:
            self.write_spectrum(self.vcoord,spec,self.vcoord_trim,spec_mask,ylab)
        
        if pdf:
            plt.savefig(self.galname+"_spec.pdf", bbox_inches = 'tight')
        else:
            if not outsideaxis: plt.show()
        
            
        
    def write_spectrum(self,v1,spec1,vmask,specmask,descrip):
       if self.fits == True:
           filename=self.galname
       else:
           filename=self.fits
       
       
       t = Table([v1,spec1],names=('Velocity (km/s)', descrip))   
       t.write(filename+"_spec.csv", format='csv',overwrite=True)
       
       t1 = Table([vmask,specmask],names=('Velocity (km/s)', descrip))   
       t1.write(filename+"_specmask.csv", format='csv',overwrite=True)
       
       
    
           