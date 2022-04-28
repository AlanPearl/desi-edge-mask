import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class DesiEdgeMasker:
    def __init__(self, ra, dec, edge_buffer=0.285,
                 bins_per_dim=2000, ra_zero=298,
                 fill_hole_radius_pixels=20,
                 ra_lims=None, dec_lims=None,
                 make_debugging_plots=False):
        """
        Given the ra/dec of spatial data, this object can mask objects
        near the edge of the data.

        Note: Runtime and memory increase quadratically with `bins_per_dim`

        Tip: If this algorithm is finding edges that shouldn't be
        there in the middle of your data, either increase `bins_per_dim`
        or decrease `fill_hole_radius_pixels` (and vice versa if it
        connects data which should be unconnected)
            If you simply want to increase resolution, then increase
        `bins_per_dim` AND `fill_hole_radius_pixels` proportionally.

        :param ra: np.ndarray[float]
            Right ascension of the data (in degrees)
        :param dec: np.ndarray[float]
            Declination of the data (in degrees)
        :param edge_buffer: float
            Distance from edge to mask
        :param bins_per_dim: int | tuple[int]
            Number of pixels in ra/dec to bin data into
        :param ra_zero: float
            RA zeropoint (this should never go through data)
            TODO: calculate a good value automatically by default?
        :param fill_hole_radius_pixels: int | tuple[int]
            ~Number of pixels to allow between data without an edge
            between them
        :param ra_lims: Optional[tuple[float]]
            Bounds on RA (reasonable defaults inferred from data)
        :param dec_lims: Optional[tuple[float]]
            Bounds on dec (reasonable defaults inferred from data)
        :param make_debugging_plots: bool
            Set to true to see plots of each step for debugging
        """
        # Parse input variables
        # =====================
        ra, dec = np.asarray(ra), np.asarray(dec)
        ra_eq = (ra - ra_zero) % 360 * np.cos(np.pi / 180 * dec)

        # Convert scalar input into a 2-tuple
        if len(np.shape(bins_per_dim)) < 1:
            bins_per_dim = (bins_per_dim,) * 2
        if len(np.shape(fill_hole_radius_pixels)) < 1:
            fill_hole_radius_pixels = (fill_hole_radius_pixels,) * 2

        if ra_lims is None:
            # ra_eq_lims = (-3, 325)
            ra_eq_lims = np.min(ra_eq) - 2*edge_buffer, np.max(ra_eq) + 2*edge_buffer
        else:
            assert (ra_lims[0] - ra_zero) // 360 == (ra_lims[1] - ra_zero) // 360, \
                "Invalid ra_lims for this ra_zero value"
            ra_eq_lims = tuple((x - ra_zero) % 360 for x in ra_lims)
        if dec_lims is None:
            # dec_lims = (-70, 90)
            dec_lims = np.min(dec) - 2*edge_buffer, np.max(dec) + 2*edge_buffer

        self.ra_zero = ra_zero
        self.ra_eq_lims, self.dec_lims = ra_eq_lims, dec_lims
        self.make_debugging_plots = make_debugging_plots

        # Use these input variables to alter Misha's code below
        # =====================================================

        # Making a better ellipse....which will be used later on to fill
        # in holes. You may need to adjust its size based on your data.
        fill_hole_gridsize = fill_hole_radius_pixels * 2 + 1
        test2 = np.zeros((fill_hole_gridsize, fill_hole_gridsize), dtype="int")
        x, y = np.ogrid[0:fill_hole_gridsize, 0:fill_hole_gridsize]
    
        # get the x and y center points of our image
        center_x = (test2.shape[0] - 1) / 2
        center_y = (test2.shape[1] - 1) / 2
    
        # create a circle mask which is centered in the middle of the
        # image, and with radius 10 pixels
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= fill_hole_radius_pixels ** 2
    
        test2[circle_mask] = 1
    
        mask = np.isfinite(ra) & np.isfinite(dec)
        ra, dec = ra[mask], dec[mask]

        if make_debugging_plots:
            # Create "2-D histogram" of footprint
            bins = (np.linspace(np.min(dec),
                                np.max(dec), 1000),
                    np.linspace(np.min((ra - ra_zero) % 360),
                                np.max((ra - ra_zero) % 360), 1000))
            h_uneq, xedges, yedges = np.histogram2d(
                dec, (ra - ra_zero) % 360, bins=bins)

            # Plot histogram
            plt.figure(figsize=(12, 12))
            plt.imshow(h_uneq, vmax=h_uneq.max(initial=0) / 2)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.show()
    
        bins = (np.linspace(*dec_lims, bins_per_dim[0]),
                np.linspace(*ra_eq_lims, bins_per_dim[1]))
        h, xedges, yedges = np.histogram2d(
            dec, ra_eq, bins=bins)
    
        # Histogram scale
        self.x_scale = x_scale = (np.diff(dec_lims)[0]) / h.shape[0]
        self.y_scale = y_scale = (np.diff(ra_eq_lims)[0]) / h.shape[1]
    
        # These are your buffers (size of kernel)....
        # 0.285 degrees is the edge buffer we used
        dim_x = edge_buffer / x_scale
        dim_y = edge_buffer / y_scale
    
        # Making elliptical kernel for buffer (same way as above)
        test = np.zeros((int(np.round(dim_x * 2) + 1), int(np.round(dim_y * 2) + 1)), dtype="int")
        x, y = np.ogrid[0:int(np.round(dim_x * 2) + 1), 0:int(np.round(dim_y * 2) + 1)]
    
        # get the x and y center points of our image
        center_x = (test.shape[0] - 1) / 2
        center_y = (test.shape[1] - 1) / 2
    
        circle_mask = (x - center_x) ** 2 / (dim_x ** 2) + (y - center_y) ** 2 / (dim_y ** 2) <= 1

        test[circle_mask] = 1
    
        if make_debugging_plots:
            plt.imshow(test)
            plt.show()

        # Change histogram to ones and zeros only
        h0 = np.clip(h, 0, 1)

        # Close holes
        kernel = test2.astype("uint8")
        closing = cv.morphologyEx(h0, cv.MORPH_CLOSE, kernel)

        if make_debugging_plots:
            plt.figure(figsize=(12, 12))
            plt.imshow(closing)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.show()
    
        # Remove edges
        kernel2 = test.astype("uint8")
        self.eroded = cv.erode(closing, kernel2) != 0
    
        if make_debugging_plots:
            plt.figure(figsize=(12, 12))
            plt.imshow(self.eroded)
            # plt.plot(1000, 100, 'o')
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

    def get_edge_mask(self, ra, dec, make_debugging_plots=None, invert=False):
        if make_debugging_plots is None:
            make_debugging_plots = self.make_debugging_plots
        ra_eq = (ra - self.ra_zero) % 360 * np.cos(np.pi / 180 * dec)
        # Convert RA/DEC to histogram indices
        ra_indices = np.round((ra_eq - self.ra_eq_lims[0]) / self.y_scale).astype(int)
        dec_indices = np.round((dec - self.dec_lims[0]) / self.x_scale).astype(int)

        # Apply mask
        edge_mask = self.eroded[dec_indices, ra_indices]

        if make_debugging_plots:
            # Sanity check
            ra2, dec2 = ra[edge_mask == 0], dec[edge_mask == 0]
            h_uneq2, xedges, yedges = np.histogram2d(dec2, (ra2 - self.ra_zero) % 360, bins=(
                np.linspace(np.min(dec2), np.max(dec2), 1000),
                np.linspace(np.min((ra2 - self.ra_zero) % 360), np.max((ra2 - self.ra_zero) % 360), 1000)))
            plt.figure(figsize=(12, 12))
            plt.imshow(h_uneq2, vmax=h_uneq2.max(initial=0) / 2)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

        return ~edge_mask if invert else edge_mask
