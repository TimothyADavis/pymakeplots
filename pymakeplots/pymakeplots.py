# coding: utf-8
import numpy as np
import scipy.interpolate as interpolate
from astropy.io import fits
#from stackarator.dist_ellipse import dist_ellipse
import astropy.units as u
from scipy import ndimage
# from tqdm import tqdm
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

def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    padZ = [0,0]
    imgP = np.pad(img, [padY, padX, padZ], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]



class pymakeplots:
    def __init__(self):
        self.galname=None
        self.gal_distance=None
        self.posang=0
        self.vsys=None
        self.moment1=None
        self.rms=None
        self.flat_cube=None
        self.pbcorr_cube=None
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
        self.hdr=None
        self.rmsfac=3
        self.obj_ra=None
        self.obj_dec=None
        self.imagesize=None
        self.xc=None
        self.yc=None
        self.bunit=None
        self.ignore_firstlast_chans=5
        self.chans2do=None
        self.spatial_trim = None
        self.maxvdisp=None
        self.cliplevel=None
        self.fits=False
        self.pvdthick=5.    
    
    def vsystrans_inv(self,val):
        return val +self.vsys

    def vsystrans(self,val):
        return val - self.vsys
        
    def beam_area(self):
        return (np.pi*(self.bmaj/self.cellsize)*(self.bmin/self.cellsize))/(4*np.log(2))
        
    def input_cube_pbcorr(self,path_to_pbcorr_cube,path_to_pb):
       
       self.pbcorr_cube = self.read_primary_cube(path_to_pbcorr_cube) 
       
       pb,hdr= self.read_in_a_cube(path_to_pb)
       
       self.flat_cube = self.pbcorr_cube*pb
                          
    
    def input_cube_flat(self,path_to_flat_cube,path_to_pb):
       
       self.flat_cube = self.read_primary_cube(path_to_flat_cube)
       
       pb,hdr= self.read_in_a_cube(path_to_pb)
       
       self.pbcorr_cube = self.flat_cube / pb
       


    def input_cube_nopb(self,path_to_cube):
       
       self.pbcorr_cube = self.read_primary_cube(path_to_cube) 
       
       self.flat_cube = self.pbcorr_cube
       


    def input_cube_pbcorr_and_flat(self,path_to_pbcorr_cube,path_to_flat_cube):
       
       self.pbcorr_cube = self.read_primary_cube(path_to_pbcorr_cube) 
       
       self.flat_cube,hdr = self.read_in_a_cube(path_to_flat_cube)

           
    def smooth_mask(self):
        """
        Apply a Gaussian blur, using sigma = 4 in the velocity direction (seems to work best), to the uncorrected cube.
        The mode 'nearest' seems to give the best results.
        :return: (ndarray) mask to apply to the un-clipped cube
        """
        sigma = 1.5 * self.bmaj / self.cellsize
        smooth_cube = ndimage.uniform_filter(self.flat_cube, size=[sigma, sigma,4], mode='constant')  # mode='nearest'
        newrms= self.rms_estimate(smooth_cube) 
        self.cliplevel=newrms*self.rmsfac   
        mask=(smooth_cube > self.cliplevel)
        return mask      
       
    def get_header_coord_arrays(self,hdr,cube_or_mom):
       try:
           cd1=hdr['CDELT1']
           cd2=hdr['CDELT2'] 
           
       except:
           cd1=hdr['CD1_1']
           cd2=hdr['CD2_2']        
       
       x1=((np.arange(0,hdr['NAXIS1'])-(hdr['CRPIX1']-1))*cd1) + hdr['CRVAL1']
       y1=((np.arange(0,hdr['NAXIS2'])-(hdr['CRPIX2']-1))*cd2) + hdr['CRVAL2']
       
       try:    
           cd3=hdr['CDELT3']
       except:    
           cd3=hdr['CD3_3']
           
               
       if hdr['CTYPE3'] =='VRAD':     
           v1=((np.arange(0,hdr['NAXIS3'])-(hdr['CRPIX3']-1))*cd3) + hdr['CRVAL3']
           if hdr['CUNIT3']=='m/s':
                v1/=1e3
                cd3/=1e3        
       else:
           f1=(((np.arange(0,hdr['NAXIS3'])-(hdr['CRPIX3']-1))*cd3) + hdr['CRVAL3'])*u.Hz
           restfreq = hdr['RESTFRQ']*u.Hz
           v1=f1.to(u.km/u.s, equivalencies=u.doppler_radio(restfreq))
           v1=v1.value
           cd3= v1[1]-v1[0]
                 
       return x1,y1,v1,np.abs(cd1*3600),cd3
           

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
        #params = {'mathtext.default': 'regular'}
        #matplotlib.rcParams.update(params)
        matplotlib.rcParams['axes.labelsize'] = 20*mult
        
           
    def rms_estimate(self,cube):
        quarterx=np.array(self.xcoord.size/4.).astype(np.int)
        quartery=np.array(self.ycoord.size/4.).astype(np.int)
        return np.nanstd(cube[quarterx*1:3*quarterx,1*quartery:3*quartery,self.ignore_firstlast_chans+2:self.ignore_firstlast_chans+5])
        
                    
        
    def read_in_a_cube(self,path):
        hdulist=fits.open(path)
        hdr=hdulist[0].header
        cube = np.squeeze(hdulist[0].data.T) #squeeze to remove singular stokes axis if present
        return cube, hdr
        
    def read_primary_cube(self,cube):
        
        ### read in cube ###
        datacube,hdr = self.read_in_a_cube(cube)
        
        try:
           self.bmaj=hdr['BMAJ']*3600.
           self.bmin=hdr['BMIN']*3600.
           self.bpa=hdr['BPA']*3600.
        except:
           self.bmaj=np.median(hdulist[1].data['BMAJ'])
           self.bmin=np.median(hdulist[1].data['BMIN'])
           self.bpa=np.median(hdulist[1].data['BPA'])
           
        try:
            self.galname=hdr['OBJECT']
        except:
            self.galname="Galaxy"
            
        try:
            self.bunit=hdr['BUNIT']
        except:
            self.bunit="Unknown"
                          
        self.xcoord,self.ycoord,self.vcoord,self.cellsize,self.dv = self.get_header_coord_arrays(hdr,"cube")
        
        try:
            self.obj_ra=hdr['OBSRA']
            self.obj_dec=hdr['OBSDEC']
        except:
            self.obj_ra=np.median(self.xcoord)
            self.obj_dec=np.median(self.ycoord)
            
            
        
        if self.dv < 0:
            datacube = np.flip(datacube,axis=2)
            self.dv*=(-1)
            self.vcoord = np.flip(self.vcoord)

        datacube[~np.isfinite(datacube)]=0.0
        
        return datacube
    
    def prepare_cubes(self):
        self.mask=self.smooth_mask()

        if self.ignore_firstlast_chans != 0:
            self.mask[:,:,0:self.ignore_firstlast_chans]=False
            self.mask[:,:,(-1)*self.ignore_firstlast_chans:-1]=False
            
        self.clip_cube()
        
        self.xc=(self.xcoord_trim-self.obj_ra)*(-1) * 3600.
        self.yc=(self.ycoord_trim-self.obj_dec) * 3600.
        
        if self.gal_distance == None:
            self.gal_distance = self.vsys/70.
            print("Warning! Estimating galaxy distance using Hubble expansion. Set `gal_distance` if this is not appropriate.")
    
    def make_all(self,pdf=False,fits=False):
        self.set_rc_params()

        fig = plt.figure( figsize=(7*3,14))

        gs0 = gridspec.GridSpec(2, 1, figure=fig)
        gs0.tight_layout(fig)
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[0], wspace=0.0, hspace=0.0)

        ax1 = fig.add_subplot(gs00[0, 0])
        ax2 = fig.add_subplot(gs00[0,1], sharey=ax1)
        ax3 = fig.add_subplot(gs00[0, 2], sharey=ax1)
        textaxes = fig.add_subplot(gs00[0, 3])
        textaxes.axis('off')


        gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=0.3, hspace=0.0)

        ax4 = fig.add_subplot(gs01[0, 0])
        ax5 = fig.add_subplot(gs01[0, 1])




        #inner_lower = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[1])

        #ax4 = plt.Subplot(fig, inner_lower[0])
        #ax5 = plt.Subplot(fig, inner_lower[1])
                #
        # ax5 = fig.add_subplot(gs[1, -1])
        plt.tight_layout()
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        gs0.tight_layout(fig, rect=[0, 0, 1, 0.97])



        #fig,axes=plt.subplots(2,3,figsize=(7*3,14))
        self.make_moments(axes=np.array([ax1,ax2,ax3]),fits=fits)
        self.make_pvd(axes=ax4,fits=fits)
        self.make_spec(axes=ax5,fits=fits)


        ###### make summary box
        
        rjustnum=35

        c = ICRS(self.obj_ra*u.degree, self.obj_dec*u.degree)

        # import ipdb
        # ipdb.set_trace()
        thetext = (self.galname)+'\n \n'
        thetext += (("RA: "+c.ra.to_string(u.hour, sep=':')))+'\n'
        thetext += ("Dec: "+c.dec.to_string(u.degree, sep=':', alwayssign=True))+'\n \n'
        thetext += ("Vsys: "+str(int(self.vsys))+" km/s")+'\n'
        thetext += ("Dist: "+str(int(self.gal_distance))+" Mpc")+'\n'
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

    
    # def make_all(self,pdf=False,fits=False):
    #     self.set_rc_params()
    #
    #     fig = plt.figure(constrained_layout=True,figsize=(7*3,14))
    #     gs = fig.add_gridspec(2, 4)
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    #     ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    #     ax4 = fig.add_subplot(gs[1,0:-1])
    #     ax5 = fig.add_subplot(gs[1, -1])
    #     textaxes = fig.add_subplot(gs[0, -1])
    #     textaxes.axis('off')
    #
    #
    #     plt.setp(ax2.get_yticklabels(), visible=False)
    #     plt.setp(ax3.get_yticklabels(), visible=False)
    #     gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    #
    #
    #
    #     # fig,axes=plt.subplots(2,3,figsize=(7*3,14))
    #     self.make_moments(axes=np.array([ax1,ax2,ax3]),fits=fits)
    #     self.make_pvd(axes=ax4,fits=fits)
    #     ax4.set_title(self.galname)
    #     self.make_spec(axes=ax5,fits=fits)
    #
    #
    #     ###### make summary box
    #
    #
    #     c = ICRS(self.obj_ra*u.degree, self.obj_dec*u.degree)
    #
    #     thetext = self.galname
    #     thetext += "\n \nRA: "+c.ra.to_string(u.hour, sep=':')
    #     thetext += "\nDec: "+c.dec.to_string(u.degree, sep=':', alwayssign=True)
    #     thetext += "\n \nVsys: "+str(int(self.vsys))+" km s$^{-1}$"
    #     thetext += "\nDist: "+str(int(self.gal_distance))+" Mpc"
    #     if self.gal_distance == self.vsys/70.:
    #         thetext+="\n(Estimated from Vsys)"
    #
    #
    #
    #     at2 = AnchoredText(thetext,
    #                        loc='upper right', prop=dict(size=30), frameon=False,
    #                        bbox_transform=textaxes.transAxes
    #                        )
    #     textaxes.add_artist(at2)
    #
    #
    #
    #     if pdf:
    #         plt.savefig(self.galname+"_allplots.pdf", bbox_inches = 'tight')
    #     else:
    #         plt.show()
            
    
    def make_moments(self,axes=None,mom=[0,1,2],pdf=False,fits=False):
        mom=np.array(mom)
        self.fits=fits
        
        if np.any(self.xc) == None:
            self.prepare_cubes()
        
        self.set_rc_params()
       
        nplots=mom.size
        if np.any(axes) == None:
            fig,axes=plt.subplots(1,nplots,sharey=True,figsize=(7*nplots,7), gridspec_kw = {'wspace':0, 'hspace':0})
            outsideaxis=0
        else:
            outsideaxis=1
            
            
        if nplots == 1:
            axes=np.array([axes])
        
        for i in range(0,nplots):
            if mom[i] == 0:
                self.mom0(axes[i],first=not i)
            if mom[i] == 1:
                self.mom1(axes[i],first=not i)
            if mom[i] == 2:
                self.mom2(axes[i],first=not i)

        
        if self.gal_distance != None:
            self.scalebar(axes[i]) # plot onto last axes

                    
        if pdf:
            plt.savefig(self.galname+"_moment"+"".join(mom.astype(np.str))+".pdf", bbox_inches = 'tight')
        else:
            if not outsideaxis: plt.show()
        
    # def scalebar_old(self,ax):
    #     barlength_pc = np.ceil((np.abs(self.xc[-1]-self.xc[0])*4.84*self.gal_distance)/1000.)*100
    #     barlength_arc=  barlength_pc/(4.84*self.gal_distance)
    #
    #     barcen=[self.xc[-1] - (barlength_arc)*1.0 , self.yc[0] + (barlength_arc)*0.5]
    #
    #     ax.add_patch(Rectangle((barcen[0]-((barlength_arc/2) + barlength_arc*0.2),self.yc[0] + (barlength_arc)*0.45), barlength_arc+ barlength_arc*0.4, (barlength_arc)*0.5,
    #                  edgecolor='none',
    #                  facecolor='white',
    #                  linewidth=1.5))
    #
    #     if np.log10(barlength_pc) > 3:
    #         ax.annotate((barlength_pc/1e3).astype(np.str)+ " kpc",xy=(barcen[0],barcen[1]+(self.yc[1]-self.yc[0])),c='k', ha='center',size=18)
    #     else:
    #         ax.annotate(barlength_pc.astype(np.int).astype(np.str)+ " pc",xy=(barcen[0],barcen[1]+(self.yc[1]-self.yc[0])),c='k', ha='center',size=18)
    #
    #     ax.plot([barcen[0]-barlength_arc/2.,barcen[0]+barlength_arc/2.],[barcen[1],barcen[1]],'k')
    #
    
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
            mask_cumsum=np.cumsum(self.mask.sum(axis=0).sum(axis=0))
            w_low,=np.where(mask_cumsum/np.max(mask_cumsum) < 0.02)
            w_high,=np.where(mask_cumsum/np.max(mask_cumsum) > 0.98)
            
            if w_low==[]: w_low=np.array([0])
            if w_high==[]: w_high=np.array([self.vcoord.size])

            self.chans2do=[np.clip(np.max(w_low)-2,0,self.vcoord.size),np.clip(np.min(w_high)+2,0,self.vcoord.size)]
    
        if self.vsys == None:
            # use the cubeto try and guess the vsys
            self.vsys=((self.pbcorr_cube*self.mask).sum(axis=0).sum(axis=0)*self.vcoord).sum()/((self.pbcorr_cube*self.mask).sum(axis=0).sum(axis=0)).sum()
        
        if self.imagesize != None:
            if np.array(self.imagesize).size == 1:
                self.imagesize=[self.imagesize,self.imagesize]
            
            wx,=np.where((np.abs((self.xcoord-self.obj_ra)*3600.) <= self.imagesize[0]))
            wy,=np.where((np.abs((self.ycoord-self.obj_dec)*3600.) <= self.imagesize[1]))
            self.spatial_trim=[np.min(wx),np.max(wx),np.min(wy),np.max(wy)]        
        
        if self.spatial_trim == None:
            
            mom0=(self.mask).sum(axis=2)
            mom0[mom0>0]=1
            
            cumulative_x = np.cumsum(mom0.sum(axis=1),dtype=np.float)
            cumulative_x /= np.max(cumulative_x)
            cumulative_y = np.cumsum(mom0.sum(axis=0),dtype=np.float)
            cumulative_y /= np.max(cumulative_y)
            
            wx_low,=np.where(cumulative_x < 0.02)
            wx_high,=np.where(cumulative_x > 0.98)
            
            wy_low,=np.where(cumulative_y < 0.02)
            wy_high,=np.where(cumulative_y > 0.98)

            
            beam_in_pix = np.int(np.ceil(self.bmaj/self.cellsize))
            
            
            self.spatial_trim = [np.clip(np.max(wx_low) - 2*beam_in_pix,0,self.xcoord.size),np.clip(np.min(wx_high) + 2*beam_in_pix,0,self.xcoord.size)\
                                , np.clip(np.max(wy_low) - 2*beam_in_pix,0,self.ycoord.size), np.clip(np.min(wy_high) + 2*beam_in_pix,0,self.ycoord.size)]
            
            
        self.flat_cube_trim=self.flat_cube[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]]
        self.pbcorr_cube_trim=self.pbcorr_cube[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]]
        self.mask_trim=self.mask[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]] 
        
        self.xcoord_trim=self.xcoord[self.spatial_trim[0]:self.spatial_trim[1]]
        self.ycoord_trim=self.ycoord[self.spatial_trim[2]:self.spatial_trim[3]]
        self.vcoord_trim=self.vcoord[self.chans2do[0]:self.chans2do[1]]  
            
            #self.spatial_trim
            
    

    def colorbar(self,mappable,ticks=None):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cb=fig.colorbar(mappable, cax=cax,ticks=ticks,orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position('top')
        return cb         
            
    #
    
    # def add_beam_old(self,ax):
    #     beampos=(self.xc[0] + (self.bmaj)*1.5, self.yc[0] + (self.bmaj)*1.5)
    #
    #     ax.add_patch(Ellipse(beampos, width=self.bmaj, height=self.bmin, angle=self.bpa,
    #                  edgecolor='black',
    #                  facecolor='none',
    #                  linewidth=1.5))
        
        
    def add_beam(self,ax):
        """
        Draw an ellipse of width=0.1, height=0.15 in data coordinates
        """
        
        ae = AnchoredEllipse(ax.transData, width=self.bmaj, height=self.bmin, angle=self.bpa,
                             loc='lower left', pad=0.5, borderpad=0.4,
                             frameon=False)                   
        ae.ellipse.set_edgecolor('black')
        ae.ellipse.set_facecolor('none')
        ae.ellipse.set_linewidth(1.5)
        ax.add_artist(ae)
        
            
        
    def mom0(self,ax1,first=True):
        mom0=(self.pbcorr_cube_trim*self.mask_trim).sum(axis=2)*self.dv
        
        
        oldcmp = cm.get_cmap("YlOrBr", 512)
        newcmp = ListedColormap(oldcmp(np.linspace(0.15, 1, 256)))
        
        
        

        im1=ax1.contourf(self.xc,self.yc,mom0.T,levels=np.linspace(np.nanmin(mom0[mom0 > 0]),np.nanmax(mom0),10),cmap=newcmp)
        
        ax1.set_xlabel('RA offset (")')
        if first: ax1.set_ylabel('Dec offset (")')
        
        
        vticks=np.linspace(np.nanmin(mom0[mom0 > 0]),np.nanmax(mom0),5)
        
        cb=self.colorbar(im1,ticks=vticks)
        
        if self.bunit == "Jy/beam":
            cb.set_label("Integrated Intensity (Jy beam$^{-1}$ km s$^{-1}$)")
        if self.bunit == "K":
            cb.set_label("Integrated Intensity (K km s$^{-1}$)")
            
            
        self.add_beam(ax1)

        
        ax1.set_aspect('equal')
        
        if self.fits:
            self.write_fits(mom0.T,0)
        
        
    def mom1(self,ax1,first=True):

        mom0=(self.pbcorr_cube_trim*self.mask_trim).sum(axis=2)
        mom1=mom0.copy()*np.nan
        mom1[mom0 != 0.0] = (((self.pbcorr_cube_trim*self.mask_trim)*self.vcoord_trim).sum(axis=2))[mom0 != 0.0]/mom0[mom0 != 0.0]
        
        im1=ax1.contourf(self.xc,self.yc,mom1.T-self.vsys,levels=self.vcoord_trim-self.vsys,cmap=sauron)
        
        ax1.set_xlabel('RA offset (")')
        if first: ax1.set_ylabel('Dec offset (")')
        
        maxv=np.max(np.abs(mom1.T-self.vsys))

        vticks=np.linspace((-1)*np.ceil(np.max(np.abs(self.vcoord_trim-self.vsys))/10.)*10.,np.ceil(np.max(np.abs(self.vcoord_trim-self.vsys))/10.)*10.,5)#(np.arange(0,(self.vcoord_trim.size),np.floor(self.vcoord_trim.size/10.)+1)*self.dv)
                
        cb=self.colorbar(im1,ticks=vticks)
        cb.set_label("Velocity (km s$^{-1}$)")
        self.add_beam(ax1)
        
        ax1.set_aspect('equal')
        
        if self.fits:
            self.write_fits(mom1.T,1)

        
    def mom2(self,ax1,first=True):
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
        cb.set_label("Obs. Vel. Disp (km s$^{-1}$)")

        
        self.add_beam(ax1)
        
        # fig.colorbar(im1, ax=ax1,ticks=
        ax1.set_aspect('equal')
        
        if self.fits:
            self.write_fits(mom2.T,2)
      
      
    def write_fits(self,array,whichmoment):
        if self.fits == True:
            filename=self.galname+"_mom"+"".join(np.array([whichmoment]).astype(np.str))+".fits"
        else:
            filename=self.fits+"_mom"+"".join(np.array([whichmoment]).astype(np.str))+".fits"
            
        newhdu = fits.PrimaryHDU(array)

        newhdu.header['CRPIX1']=1
        newhdu.header['CRVAL1']=self.xcoord_trim[0]
        newhdu.header['CDELT1']=self.xcoord_trim[1]-self.xcoord_trim[0]
        newhdu.header['CTYPE1']='RA---SIN'
        newhdu.header['CUNIT1']='deg'
        newhdu.header['CRPIX2']=1
        newhdu.header['CRVAL2']=self.ycoord_trim[0]
        newhdu.header['CDELT2']=self.ycoord_trim[1]-self.ycoord_trim[0]
        newhdu.header['CTYPE2']='DEC--SIN'
        newhdu.header['CUNIT2']='deg'
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
                
        centpix_x=np.where(np.isclose(self.xc,0.0,atol=self.cellsize/2.))[0]
        centpix_y=np.where(np.isclose(self.yc,0.0,atol=self.cellsize/2.))[0]
        
        rotcube= rotateImage(self.pbcorr_cube_trim*self.mask_trim,180-self.posang,[centpix_x[0],centpix_y[0]])

        
        pvd=rotcube[np.array(rotcube.shape[1]//2-self.pvdthick).astype(np.int):np.array(rotcube.shape[1]//2+self.pvdthick).astype(np.int),:,:].sum(axis=0)
        

        
        xx = self.yc * np.cos(np.deg2rad(self.posang)) #- self.yc * np.sin(np.deg2rad(self.posang))
        yy = self.yc * np.sin(np.deg2rad(self.posang)) #+ self.yc * np.cos(np.deg2rad(self.posang))
        
        if self.posang > 180:
            loc1="upper left"
            loc2="lower right"
        else:
            loc1="upper right"
            loc2="lower left"

        pvdaxis=(-1)*np.sign(self.yc)*np.sqrt(xx*xx + yy*yy)
        # import ipdb
        # ipdb.set_trace()
        
        
        # pvdaxis=self.xc*np.sin(np.deg2rad(self.posang)) + self.yc*np.cos(np.deg2rad(self.posang))
        oldcmp = cm.get_cmap("YlOrBr", 512)
        newcmp = ListedColormap(oldcmp(np.linspace(0.15, 1, 256)))
        
        axes.contourf(pvdaxis,self.vcoord_trim,pvd.T,levels=np.linspace(self.cliplevel,np.nanmax(pvd),10),cmap=newcmp)
        axes.contour(pvdaxis,self.vcoord_trim,pvd.T,levels=np.linspace(self.cliplevel,np.nanmax(pvd),10),colors='black')
        
        axes.set_xlabel('Offset (")')
        axes.set_ylabel('Velocity (km s$^{-1}$)')
        
        secax = axes.secondary_yaxis('right', functions=(self.vsystrans, self.vsystrans_inv))
        secax.set_ylabel(r'V$_{\rm offset}$ (km s$^{-1}$)')
        
        

        anchored_text = AnchoredText("PA: "+"".join(np.array([self.posang]).astype(np.str))+"$^{\circ}$", loc=loc1,frameon=False)
        axes.add_artist(anchored_text)
        
        if self.gal_distance != None:
            self.scalebar(axes,loc=loc2)
        
        if self.fits:
            self.write_pvd_fits(pvdaxis,self.vcoord_trim,pvd.T)
        
        if pdf:
            plt.savefig(self.galname+"_pvd.pdf", bbox_inches = 'tight')
        else:
            if not outsideaxis: plt.show()
    
    def make_spec(self,axes=None,fits=False,pdf=False):
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
        
        if self.bunit == "Jy/beam":
            
            spec*=1/self.beam_area()
            spec_mask*=1/self.beam_area()
            
            if spec_mask.max() < 1:
                spec*=1e3
                spec_mask*=1e3
                ylab="Flux Density (mJy)"
            else:
                ylab="Flux Density (Jy)"
        if self.bunit == "K":
            ylab="Tempearture (K)"
            
            
        
        
        axes.step(self.vcoord,spec,c='lightgrey',linestyle='--')
        axes.step(self.vcoord_trim,spec_mask,c='k')
        axes.set_xlabel('Velocity (km s$^{-1}$)')
        axes.set_ylabel(ylab)
        


            
            
        secax = axes.secondary_xaxis('top', functions=(self.vsystrans, self.vsystrans_inv))
        secax.set_xlabel(r'V$_{\rm offset}$ (km s$^{-1}$)')
        
        if pdf:
            plt.savefig(self.galname+"_spec.pdf", bbox_inches = 'tight')
        else:
            if not outsideaxis: plt.show()
        
            
        
            