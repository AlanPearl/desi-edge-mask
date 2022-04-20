import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class DesiEdgeMasker:
    def __init__(self, ra, dec, bins_per_dim=10_000,
                 ra_lims=None, dec_lims=None,
                 make_plots=False):
        if len(np.shape(bins_per_dim)) < 1:
            bins_per_dim = (bins_per_dim, bins_per_dim)
        
        if ra_lims is None:
            ra_lims = (-3, 325)
        if dec_lims is None:
            dec_lims = (-70, 90)
            
        self.ra_lims, self.dec_lims = ra_lims, dec_lims
        self.make_plots = make_plots
    
        test2 = np.zeros((101, 101), dtype="int")
        x, y = np.ogrid[0:101, 0:101]
    
        # get the x and y center points of our image
        center_x = test2.shape[0] // 2
        center_y = test2.shape[1] // 2
    
        # create a circle mask which is centered in the middle of the image, and with radius 10 pixels
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 50 ** 2
    
        test2[circle_mask] = 1
    
        mask = np.isfinite(ra) & np.isfinite(dec)
        ra, dec = ra[mask], dec[mask]
    
        # Create "2-D histogram" of footprint
        bins = (np.linspace(np.min(dec),
                            np.max(dec), 1000),
                np.linspace(np.min((ra - 300) % 360),
                            np.max((ra - 300) % 360), 1000))
        h_uneq, xedges, yedges = np.histogram2d(
            dec, (ra - 300) % 360,
            bins=bins)
    
        if make_plots:
            # Plot histogram
            plt.figure(figsize=(12, 12))
            plt.imshow(h_uneq, vmax=3)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.show()
    
        bins = (np.linspace(*dec_lims, bins_per_dim[0]),
                np.linspace(*ra_lims, bins_per_dim[1]))
        h, xedges, yedges = np.histogram2d(
            dec,
            (ra - 298) % 360 * np.cos(
                np.pi / 180 * dec),
            bins=bins)
    
        # Histogram scale
        self.x_scale = x_scale = (np.diff(dec_lims)[0]) / (h.shape[1] - 1)
        self.y_scale = y_scale = (np.diff(ra_lims)[0]) / (h.shape[1] - 1)
    
        # These are your buffers (size of kernel)....0.285 degrees is the edge buffer we used
        dim_x = 0.285 / x_scale
        dim_y = 0.285 / y_scale
    
        # Making elliptical kernel for buffer (same way as above)
    
        test = np.zeros((int(np.round(dim_x * 2) + 1), int(np.round(dim_y * 2) + 1)), dtype="int")
        x, y = np.ogrid[0:int(np.round(dim_x * 2) + 1), 0:int(np.round(dim_y * 2) + 1)]
    
        # get the x and y center points of our image
        center_x = test.shape[0] / 2 - 1
        center_y = test.shape[1] / 2 - 1
    
        circle_mask = (x - center_x) ** 2 / (dim_x ** 2) + (y - center_y) ** 2 / (dim_y ** 2) <= 1

        test[circle_mask] = 1
    
        if make_plots:
            plt.imshow(test)
            plt.show()

        # Change histogram to ones and zeros only
        h0 = np.clip(h, 0, 1)

        # Close holes
        kernel = test2.astype("uint8")
        closing = cv.morphologyEx(h0, cv.MORPH_CLOSE, kernel)

        if make_plots:
            plt.figure(figsize=(12, 12))
            plt.imshow(closing)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.show()
    
        # Remove edges
        kernel2 = test.astype("uint8")
        self.eroded = cv.erode(closing, kernel2) != 0
    
        if make_plots:
            plt.figure(figsize=(12, 12))
            plt.imshow(self.eroded)
            plt.plot(1000, 100, 'o')
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

    # Convert RA/DEC to histogram indices
    def get_edge_mask(self, ra, dec):
        ra_indices = np.round(((ra - 298) % 360 * np.cos(np.pi / 180 * dec) - self.ra_lims[0]) / self.y_scale).astype(int)
        dec_indices = np.round((dec - self.dec_lims[0]) / self.x_scale).astype(int)

        # Apply mask
        edge_mask = self.eroded[dec_indices, ra_indices]

        if self.make_plots:
            # Sanity check
            ra2, dec2 = ra[edge_mask == 0], dec[edge_mask == 0]
            h_uneq2, xedges, yedges = np.histogram2d(dec2, (ra2 - 298) % 360, bins=(
                np.linspace(np.min(dec2), np.max(dec2), 1000),
                np.linspace(np.min((ra2 - 298) % 360), np.max((ra2 - 298) % 360), 1000)))
            plt.figure(figsize=(12, 12))
            plt.imshow(h_uneq2, vmax=1)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

        return edge_mask
