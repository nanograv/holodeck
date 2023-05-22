"""
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import kalepy as kale


class Corner:

    _LIMITS_STRETCH = 0.1

    def __init__(self, ndim, origin='tl', rotate=True, axes=None, labels=None, limits=None, **kwargs):

        origin = kale.plot._parse_origin(origin)

        # -- Construct figure and axes
        if axes is None:
            fig, axes = kale.plot._figax(ndim, **kwargs)
            self.fig = fig
            if origin[0] == 1:
                axes = axes[::-1]
            if origin[1] == 1:
                axes = axes.T[::-1].T
        else:
            try:
                self.fig = axes[0, 0].figure
            except Exception as err:
                raise err

        self.origin = origin
        self.axes = axes

        last = ndim - 1
        if labels is None:
            labels = [''] * ndim

        for (ii, jj), ax in np.ndenumerate(axes):
            # Set upper-right plots to invisible
            if jj > ii:
                ax.set_visible(False)
                continue

            ax.grid(True)

            # Bottom row
            if ii == last:
                if rotate and (jj == last):
                    ax.set_ylabel(labels[jj])   # currently this is being reset to empty later, that's okay
                else:
                    ax.set_xlabel(labels[jj])

                # If vertical origin is the top
                if origin[0] == 1:
                    ax.xaxis.set_label_position('top')
                    ax.xaxis.set_ticks_position('top')

            # Non-bottom row
            else:
                ax.set_xlabel('')
                for tlab in ax.xaxis.get_ticklabels():
                    tlab.set_visible(False)

            # First column
            if jj == 0:
                # Not-first rows
                if ii != 0:
                    ax.set_ylabel(labels[ii])

                # If horizontal origin is the right
                if origin[1] == 1:
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')

            # Not-first columns
            else:
                # if (jj != last) or (not rotate):
                ax.set_ylabel('')
                for tlab in ax.yaxis.get_ticklabels():
                    tlab.set_visible(False)

            # Diagonals
            if ii == jj:
                # not top-left
                if (ii != 0) and (origin[1] == 0):
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')
                else:
                    ax.yaxis.set_label_position('left')
                    ax.yaxis.set_ticks_position('left')

        # If axes limits are given, set axes to them
        if limits is not None:
            limit_flag = False
            kale.plot._set_corner_axes_extrema(self.axes, limits, rotate)
        # Otherwise, prepare to calculate limits during plotting
        else:
            limits = [None] * ndim
            limit_flag = True

        # --- Store key parameters
        self.ndim = ndim
        self._rotate = rotate
        self._labels = labels
        self._limits = limits
        self._limit_flag = limit_flag

        return

    def plot(self, data, edges=None, weights=None, quantiles=None,
             color=None, cmap=None, limit=None, dist1d={}, dist2d={}):

        if limit is None:
            limit = self._limit_flag

        # ---- Sanitize
        if np.ndim(data) != 2:
            err = "`data` (shape: {}) must be 2D with shape (parameters, data-points)!".format(
                np.shape(data))
            raise ValueError(err)

        axes = self.axes
        size = np.shape(data)[0]
        shp = np.shape(axes)
        if (np.ndim(axes) != 2) or (shp[0] != shp[1]) or (shp[0] != size):
            raise ValueError("`axes` (shape: {}) does not match data dimension {}!".format(shp, size))

        # ---- Set parameters
        last = size - 1
        rotate = self._rotate

        # Set default color or cmap as needed
        color, cmap = kale.plot._parse_color_cmap(ax=axes[0][0], color=color, cmap=cmap)

        edges = kale.utils.parse_edges(data, edges=edges)
        quantiles, _ = kale.plot._default_quantiles(quantiles=quantiles)

        #
        # Draw / Plot Data
        # ----------------------------------

        # ---- Draw 1D Histograms & Carpets
        limits = [None] * size      # variable to store the data extrema
        for jj, ax in enumerate(axes.diagonal()):
            rot = (rotate and (jj == last))
            self.dist1d(
                ax, edges[jj], data[jj], weights=weights, quantiles=quantiles, rotate=rot,
                color=color, **dist1d
            )
            limits[jj] = kale.utils.minmax(data[jj], stretch=self._LIMITS_STRETCH)

        # ---- Draw 2D Histograms and Contours
        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue
            self.dist2d(
                ax, [edges[jj], edges[ii]], [data[jj], data[ii]], weights=weights,
                color=color, cmap=cmap, quantiles=quantiles, **dist2d
            )

        # If we are setting the axes limits dynamically
        if limit:
            # Update any stored values
            for ii in range(self.ndim):
                self._limits[ii] = kale.utils.minmax(limits[ii], prev=self._limits[ii])

            # Set axes to limits
            kale.plot._set_corner_axes_extrema(self.axes, self._limits, self._rotate)

        return

    def dist1d(self, ax, edge, data, color=None, **kwargs):
        """Wrapper for `kalepy.plot.dist1d` that sets default parameters appropriate for 1D data.
        """
        # Set default parameters
        kwargs.setdefault('density', False)
        kwargs.setdefault('confidence', False)
        kwargs.setdefault('carpet', True)
        kwargs.setdefault('hist', True)
        # This is identical to `kalepy.plot.dist1d` (just used for naming convenience)
        rv = kale.plot.dist1d(data, ax=ax, edges=edge, color=color, **kwargs)
        return rv

    def dist2d(
        self, ax, edges, data, weights=None, quantiles=None, sigmas=None,
        color=None, cmap=None, smooth=None, upsample=None, pad=True, outline=True,
        median=False, scatter=True, contour=True, hist=True, mask_dense=None, mask_below=True, mask_alpha=0.9
    ):

        # Set default color or cmap as needed
        color, cmap = kale.plot._parse_color_cmap(ax=ax, color=color, cmap=cmap)

        # Use `scatter` as the limiting-number of scatter-points
        #    To disable scatter, `scatter` will be set to `None`
        scatter = kale.plot._scatter_limit(scatter, "scatter")

        # Default: if either hist or contour is being plotted, mask over high-density scatter points
        if mask_dense is None:
            mask_dense = (scatter is not None) and (hist or contour)

        # Calculate histogram
        edges = kale.utils.parse_edges(data, edges=edges)
        hh, *_ = np.histogram2d(*data, bins=edges, weights=weights, density=True)

        _, levels, quantiles = kale.plot._dfm_levels(hh, quantiles=quantiles, sigmas=sigmas)
        if mask_below is True:
            mask_below = levels.min()

        # ---- Draw components
        # ------------------------------------
        handle = None

        # ---- Draw Scatter Points
        if (scatter is not None):
            handle = kale.plot.draw_scatter(ax, *data, color=color, zorder=5, limit=scatter)

        # ---- Draw Median Lines (cross-hairs style)
        if median:
            for dd, func in zip(data, [ax.axvline, ax.axhline]):
                # Calculate value
                if weights is None:
                    med = np.median(dd)
                else:
                    med = kale.utils.quantiles(dd, percs=0.5, weights=weights)

                # Load path_effects
                out_pe = kale.plot._get_outline_effects() if outline else None
                # Draw
                func(med, color=color, alpha=0.25, lw=1.0, zorder=40, path_effects=out_pe)

        cents, hh_prep = kale.plot._prep_hist(edges, hh, smooth, upsample, pad)

        # ---- Draw 2D Histogram
        if hist:
            _ee, _hh, handle = kale.plot.draw_hist2d(
                ax, edges, hh, mask_below=mask_below, cmap=cmap, zorder=10, shading='auto',
            )

        # ---- Draw Contours
        if contour:
            contour_cmap = cmap.reversed()
            # Narrow the range of contour colors relative to full `cmap`
            dd = 0.7 / 2
            nq = len(quantiles)
            if nq < 4:
                dd = nq*0.08
            contour_cmap = kale.plot._cut_colormap(contour_cmap, 0.5 - dd, 0.5 + dd)

            _ee, _hh, _handle = _contour2d(
                ax, cents, hh_prep, levels=levels, cmap=contour_cmap, zorder=20, outline=outline,
            )

            if handle is None:
                hi = 1 if len(_handle.collections) > 0 else 0
                handle = _handle.collections[hi]
                # for some reason the above handle is not showing up on legends... create a dummy line
                # to get a better handle
                col = handle.get_edgecolor()
                handle, = ax.plot([], [], color=col)

        # Mask dense scatter-points
        if mask_dense:
            # NOTE: levels need to be recalculated here!
            _, levels, quantiles = kale.plot._dfm_levels(hh_prep, quantiles=quantiles)
            span = [levels.min(), hh_prep.max()]
            mask_cmap = mpl.colors.ListedColormap('white')
            # Draw
            ax.contourf(*cents, hh_prep, span, cmap=mask_cmap, antialiased=True, zorder=9, alpha=mask_alpha)

        return handle

    def legend(self, handles, labels, index=None,
               loc=None, fancybox=False, borderaxespad=0, **kwargs):
        """
        """
        fig = self.fig

        # Set Bounding Box Location
        # ------------------------------------
        bbox = kwargs.pop('bbox', None)
        bbox = kwargs.pop('bbox_to_anchor', bbox)
        if bbox is None:
            if index is None:
                size = self.ndim
                if size in [2, 3, 4]:
                    index = (0, -1)
                    loc = 'lower left'
                elif size == 1:
                    index = (0, 0)
                    loc = 'upper right'
                elif size % 2 == 0:
                    index = size // 2
                    index = (1, index)
                    loc = 'lower left'
                else:
                    index = (size // 2) + 1
                    loc = 'lower left'
                    index = (size-index-1, index)

            bbox = self.axes[index].get_position()
            bbox = (bbox.x0, bbox.y0)
            kwargs['bbox_to_anchor'] = bbox
            kwargs.setdefault('bbox_transform', fig.transFigure)

        # Set other defaults
        leg = fig.legend(handles, labels, fancybox=fancybox,
                         borderaxespad=borderaxespad, loc=loc, **kwargs)
        return leg

    def target(self, targets, upper_limits=None, lower_limits=None, lw=1.0, fill_alpha=0.1, **kwargs):
        size = self.ndim
        axes = self.axes
        last = size - 1
        # labs = self._labels
        extr = self._limits

        # ---- check / sanitize arguments
        if len(targets) != size:
            err = "`targets` (shape: {}) must be shaped ({},)!".format(np.shape(targets), size)
            raise ValueError(err)

        if lower_limits is None:
            lower_limits = [None] * size
        if len(lower_limits) != size:
            err = "`lower_limits` (shape: {}) must be shaped ({},)!".format(np.shape(lower_limits), size)
            raise ValueError(err)

        if upper_limits is None:
            upper_limits = [None] * size
        if len(upper_limits) != size:
            err = "`upper_limits` (shape: {}) must be shaped ({},)!".format(np.shape(upper_limits), size)
            raise ValueError(err)

        # ---- configure settings
        kwargs.setdefault('color', 'red')
        kwargs.setdefault('alpha', 0.50)
        kwargs.setdefault('zorder', 20)
        line_kw = dict()
        line_kw.update(kwargs)
        line_kw['lw'] = lw
        span_kw = dict()
        span_kw.update(kwargs)
        span_kw['alpha'] = fill_alpha

        # ---- draw 1D targets and limits
        for jj, ax in enumerate(axes.diagonal()):
            if (self._rotate and (jj == last)):
                func = ax.axhline
                func_up = lambda xx: ax.axhspan(extr[jj][0], xx, **span_kw)
                func_lo = lambda xx: ax.axhspan(xx, extr[jj][1], **span_kw)
            else:
                func = ax.axvline
                func_up = lambda xx: ax.axvspan(extr[jj][0], xx, **span_kw)
                func_lo = lambda xx: ax.axvspan(xx, extr[jj][1], **span_kw)

            if targets[jj] is not None:
                func(targets[jj], **line_kw)
            if upper_limits[jj] is not None:
                func_up(upper_limits[jj])
            if lower_limits[jj] is not None:
                func_lo(lower_limits[jj])

        # ---- draw 2D targets and limits
        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue
            for kk, func, func_lim in zip([ii, jj], [ax.axhline, ax.axvline], [ax.axhspan, ax.axvspan]):
                if targets[kk] is not None:
                    func(targets[kk], **line_kw)
                if upper_limits[kk] is not None:
                    func_lim(extr[kk][0], upper_limits[kk], **span_kw)
                if lower_limits[kk] is not None:
                    func_lim(lower_limits[kk], extr[kk][0], **span_kw)

        return


def _contour2d(ax, edges, hist, levels, outline=True, **kwargs):

    LW = 1.5

    alpha = kwargs.setdefault('alpha', 0.8)
    lw = kwargs.pop('linewidths', kwargs.pop('lw', LW))
    kwargs.setdefault('linestyles', kwargs.pop('ls', '-'))
    kwargs.setdefault('zorder', 10)

    # ---- Draw contours
    cont = ax.contour(*edges, hist, levels=levels, linewidths=lw, **kwargs)

    # ---- Add Outline path effect to contours
    if (outline is True):
        outline = kale.plot._get_outline_effects(2*lw, alpha=1 - np.sqrt(1 - alpha))
        plt.setp(cont.collections, path_effects=outline)

    return edges, hist, cont
