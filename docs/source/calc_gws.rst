================================================
Getting Started: Calculating Gravitational Waves
================================================

.. contents:: :local:

Basics
======

The chirp-mass is defined as:

.. math::
   \mathcal{M} \equiv \frac{\left(m_1 m_2\right)^{3/5}}{M^{1/5}} = M \frac{q^{3/5}}{\left(1 + q\right)^{6/5}}.

for a total mass :math:`M = m_1 + m_2`, and mass-ratio :math:`q \equiv m_2 / m_1 \leq 1`.

The 'hardening time-scale' is the time it takes a binary to 'harden'(i.e. come closer together), such that the frequency increases by an e-folding. This is defined as,

.. math::
   \tau_f \equiv \frac{dt}{d\ln f_r} = \frac{f_r}{df_r/dt},

where :math:`f_r` denotes the rest-frame GW frequency which is twice the
orbital frequency. This can be related to observer-frame frequency as,
:math:`f_r = (1+z) f`, where :math:`z` is the redshift to the source.
The hardening timescale is determined by whatever physical processes are
serving to merge the binary (e.g. dynamical friction, stellar
scattering, gas torques, gravitational waves). The hardening timescale
assuming the binary is circular, and is driven purely due to
gravitational waves (GWs) is,

.. math::
    \tau_\textrm{GW,circ}= \frac{5}{96}\left(\frac{G\mathcal{M}}{c^3}\right)^{-5/3} \left(2 \pi f_r\right)^{-8/3}.
    :name: eq:time_hard_gw

Total GW power emitted for a circular binary is,

.. math::
    L_\textrm{GW}= \frac{32}{5 G c^5} \left(G\mathcal{M}\right)^{10/3} \left( 2\pi f_r\right)^{10/3}.

Alternatively, the GW strain (sky and polarization averaged) from a
circular binary can be written as [Sesana2008]_.

.. math::
    h_{s,circ}(f_r) = \frac{8}{10^{1/2}} \frac{\left(G\mathcal{M}\right)^{5/3}}{c^4 \, d_L}
        \left(2 \pi f_r\right)^{2/3},

for a luminosity distance :math:`d_L`.


Gravitational Wave Background (GWB)
===================================

The most accurate way to calculate the GWB characteristic strain is to
sum up the strains from all binaries in the universe within a
logarithmic frequency interval [Sesana2008]_:

.. math::
    h_c^2(f) = \int_0^\infty \!\! dz \; \frac{d^2 N}{dz \, d\ln f_r} \; h_s^2

where :math:`N` is the number of binaries as a function of redshift and
(rest-frame) frequency. Because :math:`N` is a number of binaries, it is
always an integer number. In practice, binary populations are generally
simulated for a small fraction of the universe (e.g. the volume of a
cosmological simulation). In this case, we started from a comoving
number-density of sources :math:`n_c \equiv \frac{dN}{dV_c} = \frac{N}{V_\textrm{sim}}`,
where here :math:`V_\textrm{sim}` is the comoving volume of the simulation (or
data set). We can relate the number and number density together using
the chain rule as [Sesana2008]_ (Eq.6),

.. math::
    \frac{d^2 N}{dz \, d\ln f_r} = \frac{d n_c}{dz} \frac{dt}{d\ln f_r} \frac{dz}{dt} \frac{d V_c}{dz}.

The second term on the right-hand side is the hardening timescale
(:math:`\tau_f`). This makes sense because the longer binaries spend in
a given interval of frequency, the more binaries we expect to find in
that interval. Note that, when performing calculations using the
right-hand side of this expression, there is generally nothing that
enforces that the derived number of binaries in the universe is an
integer (important later). The third and fourth terms are two
cosmographic expressions that define the rate of evolution of the
Universe, and the amount of comoving volume as a function of redshift
[Hogg1999]_

.. math::
    \frac{dz}{dt} = & \, H_0 \left(1+z\right) \, E(z), \\
           \frac{d V_c}{dz} = & \, 4\pi \frac{c}{H_0} \frac{d_c^2}{E(z)},

where luminosity distance is related to comoving distance as :math:`d_L = d_c \, (1+z)`.
Putting this together we have,

.. math::
   h_c^2(f) = \int_0^\infty \!\! dz \; \frac{dn_c}{dz} \, h_s^2  \, 4\pi c \, d_c^2 \cdot \left(1+z\right) \, \tau_f.


Discretization
==============

For a calculation using a finite volume, and discrete time-steps, this
turns from an integral into a summation. One summation is over redshift.
There is another summation over binaries. This is because, to calculate
a (comoving) number density in a finite volume, we will add up all of
the individual binaries, and then divide by the simulation volume. Thus,
in practice, we can write:

.. math::
    h_c^2(f) = \sum_\textrm{redshift} \; \sum_\textrm{binaries} \; h_s^2 \;
        \frac{4\pi \, c \, d_c^2 \cdot \left(1 + z\right)}{V_\textrm{sim}} \; \tau_f.

This means, that at each redshift (or simulation snapshot), we sum over
all binaries, adding up the GW strain (:math:`h_s`) from each binary,
multiplied by some factor (everything after :math:`h_s^2`) which we
label as :math:`\Lambda`. The term :math:`\Lambda` is the conversion
factor from one simulation binary, into the number of binaries in a full
universe.

Recall, however, that this calculation does not enforce the integral
nature of binaries. At high frequencies, :math:`\Lambda < 1` which
represents a 'fraction'of a binary. What this means is that the
'expectation value' for the number of binaries in the Universe, is a
fraction, means that there is some probability of there being a binary
(or not being a binary). To handle this, instead of using each binary's
:math:`\Lambda` value directly, we can draw from a Poisson distribution
centered at this expectation value, i.e. :math:`\mathcal{P}(\Lambda)`.
So we can write,

.. math::
    h_c^2(f) = & \sum_\textrm{redshift} \; \sum_\textrm{binaries} \; h_s^2 \cdot \mathcal{P}(\Lambda), \\
           \Lambda \equiv & \frac{4\pi \, c \, d_c^2 \cdot \left(1 + z\right)}{V_\textrm{sim}} \; \tau_f.

In the case that binary are evolving purely due to GW emission, we can
replace :math:`\tau_f` with :math:numref:`eq:time_hard_gw` and write:

.. math::

    \Lambda \equiv \frac{1}{V_\textrm{sim}} \; \frac{5 \pi c}{24}
        \, \frac{d_c^2}{\left(1+z\right)^{5/3}}
        \, \left( \frac{G \mathcal{M}}{c^3} \right)^{-5/3}
        \, \left(2\pi f\right)^{-8/3}.

Note that the observed GW frequencies :math:`f` are arbitrarily chosen:
the locations at which to 'sample' the GWB. Typically, for pulsar timing
arrays, these are chosen based on Nyquist sampling for a given
observational duration :math:`T \sim 15 \, \textrm{yr}` and cadence
:math:`\Delta t \sim 2 \, \textrm{week}`, such that such that
:math:`f = \left[1/T, 2/T, 3/T, \, \ldots \, , 1/\left(2 \Delta t\right)\right]`.


References
==========

See `full annotated bibliography here <biblio>`_.

.. [Hogg1999] Hogg 1999.
.. [Sesana2008] Sesana+2008.
