# calculations for New Horizons Parallax paper

# https://www.numerical.recipes/book.html
# 3rd edition in C++, section 15.4 General Linear Least Squares pp. 788-799
# Numerical Recipes: The Art of Scientific Computing, Third Edition,
# by W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery.
# Version 3.04 (2011).
# Text is Copyright c 1988-2007 by Cambridge University Press.
# Computer source code is Copyright c 1987-2007 by Numerical Recipes Software.
# Hardcover ISBN 978-0-521-88068-8 published by Cambridge University Press

from numpy import (array, asfarray, eye, matmul, pi, cos, sin, sqrt,
                   zeros, empty, newaxis, where, arcsin, arctan2)
# scipy.linalg generally slightly better than numpy.linalg
from scipy.linalg import svd, inv, solve, pinv
from numpy.ma import MaskedArray

# These are required if you call gaia_source_id() or get_astrometry()
# from astroquery.simbad import Simbad
# from astroquery.gaia import Gaia
Simbad = Gaia = None

j2000 = 2451545.0    # Julian date of J2000 epoch
yr = 365.25  # days per Julian year (Gaia unit of time for proper motions)
pc = 3600. * 180./pi  # au/pc = 206264.80625 arcsec/radian
au = 149597870.7  # km/au (exactly)
kms2auyr = 86400*yr / au  # (s/yr) / (km/au) = (au/yr)/(km/s)


# Gaia DR3 source_id:
# 5853498713190525696 for Proxima Cen
# 3864972938605115520 for Wolf 359
def gaia_source_id(name):
    table = Simbad.query_objectids(name)
    names = []
    for row in table:
        name = row['ID']
        if "gaia" in name.lower():
            names.append(name)
    return names


# Gaia gaia_source column names:
# source_id
# ref_epoch (julian years) 2015.5 for DR2, 2016.0 for DR3
# ra (deg) barycentric right ascension in ICRS (or BCRS)
# dec (deg) barycentric declination in ICRS
# parallax (mas)
# pmra (mas/year) proper motion in right ascension direction per Julian year
# pmdec (mas/year) proper motion in declination direction per Julian year
# radial_velocity (km/s)
# ra_error (mas) standard deviation in the ra direction, not in ra angle
# dec_error (mas)
# parallax_error (mas)
# pmra_error (mas/year)
# pmdec_error (mas/year)
# radial_velocity_error (km/s)
# correlation coefficients (covariance = corr*error_i*error_j)
# [ra]  ra_dec_corr  ra_parallax_corr  ra_pmra_corr  ra_pmdec_corr
#       [dec]  dec_parallax_corr  dec_pmra_corr  dec_pmdec_corr
#              [parallax]  parallax_pmra_corr  parallax_pmdec_corr
#                          [pmra]  pmra_pmdec_corr
#                                  [pmdec]
# Note Proxima Cen and Wolf 359 EDR3 are same source_id and values as DR3.
def get_astrometry(*ids, catalog="gaiadr3"):
    cols = ("source_id, ref_epoch, ra, dec, parallax, pmra, pmdec, " +
            "radial_velocity, ra_error, dec_error, parallax_error, " +
            "pmra_error, pmdec_error, radial_velocity_error, " +
            "ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, " +
            "dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, " +
            "parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr")
    query = "SELECT TOP 10 {} FROM {}.gaia_source WHERE source_id = ".format(
        cols, catalog)
    data = []
    for id in ids:
        raw = GaiaCoordinates(Gaia.launch_job(query+str(id)).get_results()[0])
        data.append(raw)
    return data


class GaiaCoordinates(object):
    def __init__(self, job_result):
        self.raw = raw = dict(job_result)
        for key in raw:
            if isinstance(raw[key], MaskedArray):
                raw[key] = None  # easier to recognize than MaskedArray
        self.vec0 = array([raw[nm] for nm in
                          ["ra", "dec", "parallax", "pmra", "pmdec"]])
        self.cov0 = self.covariance_matrix(raw)
        self.set_epoch()

    @staticmethod
    def covariance_matrix(record):
        """form 5x5 covariance matrix for (ra, dec, parallax, pmra, pmdec)"""
        names = ["ra", "dec", "parallax", "pmra", "pmdec"]
        std = array([record[nm + "_error"] for nm in names])
        cov = eye(5)
        for i, nmi in enumerate(names):
            for j, nmj in enumerate(names):
                if j == i:
                    continue
                pair = nmi + "_" + nmj if i < j else nmj + "_" + nmi
                cov[i, j] = record[pair + "_corr"]
        return std[:, newaxis] * cov * std

    def djd_from(self, jd_or_jy=None):
        """return input epoch minus ref_epoch in days"""
        if jd_or_jy is None:
            return 0.0

        def jy2jd(jy):
            return 365.25 * (jy - 2000.0) + j2000

        jd0 = jy2jd(self.raw["ref_epoch"])
        jd = jd_or_jy if jd_or_jy > 9999 else jy2jd(jd_or_jy)
        return jd - jd0

    def set_epoch(self, jd_or_jy=None):
        dt = self.djd_from(jd_or_jy) / 365.25  # (year)
        # vec is (ra, dec, parallax, pmra, pmdec)
        vec = self.vec0.copy()  # be sure not to clobber self.vec0
        # Compute tshift matrix, matmul(tshift, vec0) --> vec at jd_or_jy
        tshift = eye(5)
        tshift[0, 3] = dt
        tshift[1, 4] = dt
        # Gaia precision is so high that there can be a significant
        # difference between stepping around the great circle in the
        # (pmra, pmdec) direction and stepping in a rectangular (ra, dec)
        # coordinate system.  Prefer stepping around the great circle.
        vec[:2] = _ra_dec_stepper(vec[0], vec[1], vec[3]*dt, vec[4]*dt)
        deg2mas = 3600000.0
        ra, dec, parallax = vec[:3]
        parallax /= deg2mas
        ra, dec, parallax = [v*pi/180. for v in (ra, dec, parallax)]
        cd, sd = cos(dec), sin(dec)
        ca, sa = cos(ra), sin(ra)
        self.dt = dt
        self.vec = vec
        tshift[0, 3] = dt  # covariances use angle on sky like pmra
        self.cov = cov = matmul(tshift, matmul(self.cov0, tshift.T))
        # Also compute position p and its covariance pcov
        # p = (cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec)) / parallax
        #     where p is in parsecs if parallax is in arcsec
        #     or alternatively p is in AU if parallax is in radians
        pdir = array([cd*ca, cd*sa, sd])
        self.p = p = pdir / parallax
        # dpddec = (-sd*ca, -sd*sa, cd) / parallax
        # dpdra = (-sa, ca, 0) / parallax  # dra is on sky not change in ra
        # form matrix of partial derivatives dx/dra, etc.
        radir = array([-sa, ca, 0.])
        decdir = array([-sd*ca, -sd*sa, cd])
        der = array([radir, decdir, -p]).T / parallax
        cov = cov[:3, :3]  # (ra, dec, parallax) covariances in mas
        radian = pi/180. / 3600000.  # radians/mas
        # convert covariances to rad**2, then transform to p = (x, y, z)
        self.pcov = pcov = matmul(der, matmul(cov*radian**2, der.T))
        # Do svd decomposition on pcov.
        # (Same as eigen-decomposition for positive definite matrices.)
        self.paxes, pstd, _ = svd(pcov)
        self.pstd = sqrt(pstd)
        # (Ignored third return value always paxes.T for symmetric matrix.)
        # paxes[:,i] axis corresponds to psv[i], psv sorted largest first
        # paxes[:,0] always very nearly direction of p, and pstd[0] is
        # at least tens of thousands of times larger than pstd[1 or 2].
        # For Proxima and Wolf, tranverse errors in p are under 0.001 AU,
        # while distance errors are tens of AU.
        # That is, the error ellipsoid for p is of order 100,000 times
        # larger in the radial direction than in the transverse direction.
        # Note also that the proper motion errors cause the transverse
        # errors to grow quite rapidly as you move away from the Sun.
        # Form the projection matrix that takes IRCS xyz to (ra, dec, p)
        # directions.
        self.proj = array([radir, decdir, pdir])


class NHObservation(object):
    def __init__(self, *ra_dec_errs):
        """each argument is ((ra, dec), (ra_std, dec_std, ra_dec_corr))"""
        has_errs = True if asfarray(ra_dec_errs[0][1]).shape else False
        wgt = zeros((2, 2))
        ra_dec = zeros(2) if has_errs else []
        nobs = len(ra_dec_errs)
        raw_radec, raw_errors = [], []
        for radec in ra_dec_errs:
            if has_errs:
                radec, (ra_err, dec_err, ra_dec_corr) = radec
                raw_radec.append(radec)
                raw_errors.append((ra_err, dec_err, ra_dec_corr))
                std = array([ra_err, dec_err])
                cov = array([[1., ra_dec_corr], [ra_dec_corr, 1.]])
                cov = std[:, newaxis] * cov * std
                voc = inv(cov)
                ra_dec += matmul(voc, asfarray(radec))
                wgt += voc
            else:
                ra_dec.append(radec)
        if has_errs:
            cov = inv(wgt)
            ra_dec = matmul(cov, ra_dec)
        else:
            raw_radec = ra_dec
            radec = asfarray(ra_dec)
            ra_dec = radec.mean(axis=0)
            radec -= ra_dec  # deviations from mean
            radec[:, 0] *= cos(ra_dec[1] * pi/180.)  # correct dra to on sky
            # ... so all deviations are angles on the sky
            # Use unbiased estimator for covariance matrix,
            # sum of outer product of deviations divided by N-1.
            cov = (radec[:, newaxis] * radec[..., newaxis]
                   ).sum(axis=0) / (nobs - 1)
            # Covariance of the mean has an additional divide by nobs
            cov /= nobs
        self.raw_radec = raw_radec
        self.raw_errors = raw_errors
        self.ra_dec = ra_dec
        self.ra_dec_cov = cov
        ra, dec = ra_dec
        pi180 = pi / 180.
        cd, sd = cos(dec * pi180), sin(dec * pi180)
        ca, sa = cos(ra * pi180), sin(ra * pi180)
        er = array([cd*ca, cd*sa, sd])  # unit vector, radial direction
        eradec = array([[-sa, ca, 0.],  # unit vector, ra direction
                        [-sd*ca, -sd*sa, cd]])  # unit vector, dec direction
        cov = matmul(eradec.T, matmul(cov, eradec)) * pi180**2
        self.er = er
        self.er_cov = cov
        # compute projection matrix perpendicular to er
        self.er_perp = eye(3) - er[:, None] * er
        # save rotation matrix, matmul(er_rot, (vx, vy, vz)) = (vra, vdec, vr)
        self.er_rot = array([eradec[0], eradec[1], er])


def step_ra_dec(ra, dec, dra, ddec):
    # (ra, dec) in degrees, (dra, ddec) in mas on the sky
    dra, ddec = dra / 3600000., ddec / 3600000.  # convert to degrees
    return ra + dra/cos(dec * pi/180.), dec + ddec


def step_great_circle(ra, dec, dra, ddec):
    # (ra, dec) in degrees, (dra, ddec) in mas on the sky
    # Take (dra, ddec) to be a direction on the sky and move around that
    # great circle by the length of the step.
    dra, ddec = dra / 3600000., ddec / 3600000.  # convert to degrees
    pi180 = pi / 180.  # convert to radians
    dra, ddec = dra * pi180, ddec * pi180
    ra, dec = ra * pi180, dec * pi180
    cd, sd = cos(dec), sin(dec)
    ca, sa = cos(ra), sin(ra)
    er = array([cd*ca, cd*sa, sd])  # radial direction
    era = array([-sa, ca, 0.])  # ra direction
    edec = array([-sd*ca, -sd*sa, cd])  # dec direction
    dang = sqrt(dra**2 + ddec**2)  # magnitude of step angle
    if dang == 0.:
        dang = 1.e-30
    # move around great circle by dang to reach new radial direction
    er = er*cos(dang) + (era*dra + edec*ddec)*sin(dang)/dang
    ra = arctan2(er[1], er[0])/pi180
    if ra < 0.:
        ra += 360.
    return ra, arcsin(er[2])/pi180


# Gaia precision is so high that there can be a significant
# difference between stepping around the great circle in the
# (pmra, pmdec) direction and stepping in a rectangular (ra, dec)
# coordinate system.  Prefer stepping around the great circle.
_ra_dec_stepper = step_great_circle


def to_ra_dec(xyz, andr=False):
    """convert {..., 3} array of xyz values to ra, dec, dist"""
    r = sqrt((xyz**2).sum(axis=-1))
    dec = arcsin(xyz[..., 2] / r) * 180./pi
    ra = arctan2(xyz[..., 1], xyz[..., 0]) * 180./pi
    ra = where(ra < 0., ra + 360., ra)
    return (ra, dec, r) if andr else (ra, dec)


def to_xyz(ra, dec=None, r=None):
    """convert (ra, dec) or {..., 2} ra_dec to {..., 3} xyz"""
    ra = asfarray(ra)
    if dec is None:
        if ra.shape[-1] == 3:
            r = ra[..., 2]
        ra, dec = ra[..., 0], ra[..., 1]
    deg2rad = pi / 180.
    ra, dec = deg2rad * ra, deg2rad * dec
    xyz = empty(ra.shape + (3,))
    xy = cos(dec)
    xyz[..., 0], xyz[..., 1], xyz[..., 2] = xy*cos(ra), xy*sin(ra), sin(dec)
    if r is not None:
        xyz *= asfarray(r)[..., newaxis]
    return xyz


# proxima, wolf = get_astrometry(5853498713190525696, 3864972938605115520)
proxima = GaiaCoordinates(dict(
    SOURCE_ID=5853498713190525696,
    ref_epoch=2016.0,
    ra=217.39232147200883,
    dec=-62.67607511676666,
    parallax=768.0665391873573,
    pmra=-3781.741008265163,
    pmdec=769.4650146478623,
    radial_velocity=-21.942726,  # Simbad gives -20.578199 [0.004684]
    ra_error=0.023999203,
    dec_error=0.03443618,
    parallax_error=0.049872905,
    pmra_error=0.031386077,
    pmdec_error=0.050524533,
    radial_velocity_error=0.21612652,
    ra_dec_corr=0.37388447,
    ra_parallax_corr=0.056153428,
    ra_pmra_corr=-0.30604824,
    ra_pmdec_corr=-0.07928604,
    dec_parallax_corr=0.15966518,
    dec_pmra_corr=-0.07302318,
    dec_pmdec_corr=-0.20184441,
    parallax_pmra_corr=-0.11339641,
    parallax_pmdec_corr=-0.095663965,
    pmra_pmdec_corr=0.6296853))
wolf = GaiaCoordinates(dict(
    SOURCE_ID=3864972938605115520,
    ref_epoch=2016.0,
    ra=164.10319030755974,
    dec=7.002726940984864,
    parallax=415.17941567802137,
    pmra=-3866.3382751436793,
    pmdec=-2699.214987679166,
    radial_velocity=None,  # Simbad gives 19.57 [0.0005]
    ra_error=0.06683743,
    dec_error=0.051524777,
    parallax_error=0.06837086,
    pmra_error=0.08130645,
    pmdec_error=0.06910815,
    radial_velocity_error=None,
    ra_dec_corr=0.08985967,
    ra_parallax_corr=-0.38118768,
    ra_pmra_corr=0.07363863,
    ra_pmdec_corr=-0.016567856,
    dec_parallax_corr=-0.2219509,
    dec_pmra_corr=0.007211521,
    dec_pmdec_corr=0.057815764,
    parallax_pmra_corr=-0.006037742,
    parallax_pmdec_corr=-0.026854547,
    pmra_pmdec_corr=-0.16432397))

# Adjust Gaia data to epoch of New Horizons observations.
proxima.set_epoch(2458961.9214508)
wolf.set_epoch(2458962.8230569)
# Note the time from Proxima to Wolf observation (yr).
dt_wp = wolf.dt - proxima.dt

# Use Simbad radial velocities, converting (km/s) --> (au/yr)
proxima.raw["radial_velocity"] = -20.578199 * kms2auyr
proxima.raw["radial_velocity_error"] = 0.004684 * kms2auyr
wolf.raw["radial_velocity"] = 19.5700 * kms2auyr
wolf.raw["radial_velocity_error"] = 0.0005 * kms2auyr

# New Horizons observations - aggregate into single (ra, dec) and cov
# from Marc Buie.  Use deviations from the mean to estimate covariances.
maspx = 4080.0  # mas/pixel
proxima_nh = NHObservation(
    (217.362885635, -62.676382283),  # lor_0449855932.fits
    (217.362917492, -62.676324665),  # lor_0449855931.fits
    (217.363443584, -62.676359090))  # lor_0449855930.fits
wolf_nh = NHObservation(
    (164.094305581, 7.001002589),  # lor_0449933837.fits
    (164.094306594, 7.001021465),  # lor_0449933832.fits
    (164.094286808, 7.001023757))  # lor_0449933827.fits
# proxima ra_dec = 217.363082, -62.676355
#            std     0.000235    0.000019  corr -0.3903
#    wolf ra_dec = 164.0942997,   7.0010159
#            std     0.0000091    0.0000094  corr -0.5442

# New Horizons positions according to JPL Horizons
# - what is coordinate origin here? solar system barycenter?
nh_at_prox = to_xyz(289.125167, -20.283167, 46.8555688617415)
nh_at_wolf = to_xyz(289.121042, -20.283028, 46.8478566629295)
# Numbers in DSN column of table 5 of paper differ:
nh_dsn = to_xyz(287.914, -20.730, 47.115)
# Direct query of JPL Horizons
# Target Body:  New Horizons (spacecraft) [NH New_Horizons]
# Observer Location:  Solar System Barycenter (SSB) [code: 500]
#                                RA        DEC      dRA*cosD  d(DEC)/dt
# 2020-Apr-22 10:06:53.349     287.87127 -20.44346  0.157249  0.022397
#        delta  deldot:        47.1099608471106  13.8858416
# 2020-Apr-23 07:45:12.116     287.87228 -20.44333  0.157201  0.022391
#                              47.1171911151307  13.8856583
jpl_nh_p = to_xyz(287.87127, -20.44346, 47.1099608471106)
jpl_nh_w = to_xyz(287.87228, -20.44333, 47.1171911151307)
jpl_nh_x = 0.5*(jpl_nh_p + jpl_nh_w)

# Note that New Horizons moved between the two observations, which
# is counter to the assumption that there is a single NH position vector.
# With three stars, we would have enough data to estimate the NH velocity
# as well as its position, but with only two, we need an estimate of
# the displacement dr between observations.
dr_wp = nh_at_wolf - nh_at_prox  # (magnitude 0.008337 au)
v_nh = dr_wp / dt_wp  # NH velocity (magnitude 16.010 km/s)
# The 0.008337 change in NH position is to be compared to the distance to
# Wolf 359.  Since the Gaia errors in transverse position of Wolf 359
# are of order 0.001 au, the NH motion between observations is potentially
# significant.  However, the errors in the NH angle measurements are of
# order one microradian, which is about 0.5 au in the transverse position
# of Wolf 359, so the 0.008337 au of spacecraft motion in the day between
# the two observations will not make any significant difference.


def position_solver(*stars, options=""):
    """Solve for spacecraft position given several star observations.

    One argument per star observation.  Each argument is (gaia, observation).
    The gaia object is either a position as a 1D ndarray, or an object with
    gaia.p and gaia.pcov members (like a GaiaCoordinates instance), while
    the observation object is either a unit direction vector as a 1D ndarray,
    or an object with observation.er and observation.er_cov members (like a
    NHObservation instance.

    Returns (x, urms, chi2, xcov, xaxes, xstd), or just (x, urms) if the
    observations are specified as 1D ndarrays.

    Options, if specified are one or more letters:
    "q" - equal weighting (like Kaplan), implies "1"
          default is 1/p**2 on first pass, pcov weighting thereafter
    "1" - stop after first pass
    """
    equal_weighting = "q" in options
    stop_1 = equal_weighting or ("1" in options)
    no_uvdvu = "0" in options
    no_pcov = "x" in options
    p, pcov, d, dcov, q = [], [], [], [], []
    for gaia, obs in stars:
        try:
            p.append(asfarray(gaia))
            if len(pcov):
                raise ValueError("all stars must have same p format")
        except TypeError:
            if len(pcov) < len(p):
                raise ValueError("all stars must be same p format")
            p.append(gaia.p)
            pcov.append(gaia.pcov)
        try:
            er = asfarray(obs)
            if len(dcov):
                raise ValueError("all stars must have same d format")
        except TypeError:
            if len(dcov) < len(d):
                raise ValueError("all stars must be same d format")
            er = obs.er
            dcov.append(obs.er_cov)
        d.append(er)
        q.append(eye(3) - er[:, newaxis] * er)  # projection normal to er
    p = asfarray(p)
    p2 = (p**2).sum(axis=-1)
    if pcov:
        pcov = asfarray(pcov)
    d = asfarray(d)
    if dcov:
        dcov = asfarray(dcov)
    q = asfarray(q)
    if equal_weighting:
        mat = q.sum(axis=0)
        rhs = matmul(q, p[..., newaxis]).sum(axis=0)[:, 0]
    else:
        qop2 = q / p2[:, newaxis, newaxis]
        mat = qop2.sum(axis=0)
        rhs = matmul(qop2, p[..., newaxis]).sum(axis=0)[:, 0]

    # Make first estimate of NH position x.
    x = solve(mat, rhs)
    nstar = p.shape[0]
    r = p - x
    u = matmul(q, r[..., newaxis])  # component of p-x normal to d
    urms = sqrt((u**2).sum() / nstar)  # rms distance of x from LOSs d (au)
    if not len(dcov):
        return x, urms  # Just return estimated x and rms LOS distances.
    if no_pcov or not len(pcov):
        pcov = zeros((nstar, 3, 3))

    # If covariances given, compute minimum chi2 value of x.
    for ipass in range(10):  # should converge in fewer than 10 passes
        r2 = (r**2).sum(axis=-1)[:, newaxis, newaxis]
        cov = pcov + r2*dcov
        cov = matmul(q, matmul(cov, q))  # covariance of u_i
        voc = array([pinv(m, rtol=1.e-8) for m in cov])  # cov pseudo-inverses
        vocu = matmul(voc, u)
        xvdx = matmul(vocu.transpose(0, 2, 1), matmul(dcov, vocu))
        wgt = voc if no_uvdvu else (voc - xvdx * eye(3))
        mat = wgt.sum(axis=0)
        xcov = inv(mat)
        xaxes, xstd, _ = svd(xcov)
        xstd = sqrt(xstd)
        if stop_1:
            break
        rhs = matmul(wgt, p[..., newaxis]).sum(axis=0)[:, 0]
        xold, x = x, matmul(xcov, rhs)
        if max(abs(x - xold)) < 0.01 * xstd[0]:
            break
        r = p - x
        u = matmul(q, r[..., newaxis])
    else:
        raise RuntimeError("chi2 convergence failed, {}, {}".format(
            max(abs(x - xold)), xstd[0]))
    chi2 = (u * vocu).sum()
    urms = sqrt((u**2).sum() / nstar)  # rms distance of x from LOSs d (au)
    return x, urms, chi2, xcov, xaxes, xstd


class Garbage(object):
    def __init__(self, **kwargs):
        self.add(**kwargs)

    def add(self, **kwargs):
        self.__dict__.update(**kwargs)


def display_prep(from_wolf=False):
    x, urms, chi2, xcov, xaxes, xstd = position_solver(
        (proxima, proxima_nh), (wolf, wolf_nh))  # chi2 NH loc
    y, yrms, ychi2, ycov, yaxes, ystd = position_solver(
        (proxima, proxima_nh), (wolf, wolf_nh), options="q")  # Kaplan NH loc
    # nh_x = 0.5*(nh_at_prox + nh_at_wolf)  # true NH position from JPL
    nh_x = nh_dsn  # true NH position from JPL
    if from_wolf:
        origin, nh, other, othernh = wolf, wolf_nh, proxima, proxima_nh
    else:
        origin, nh, other, othernh = proxima, proxima_nh, wolf, wolf_nh
    p0, p0cov, p0proj = origin.p, origin.pcov, origin.proj
    p1, p1cov, p1proj = other.p, other.pcov, other.proj
    d0, d0cov, d0proj = nh.er, nh.er_cov, nh.er_perp
    d1, d1cov, d1proj = othernh.er, othernh.er_cov, othernh.er_perp
    # rotate p0proj 180 degress about tilted dec axis
    proj = p0proj * [[-1.], [1.], [-1.]]
    # Drawing will be the view from p0 of:
    # 1. light pixel grid centered on SS barycenter
    #      technically, scaled to plane of SS barycenter
    # 2. point at SS barycenter (negative Gaia position of p0) = (0, 0)
    # 3. point at projection of line in NH observed direction of p0 thru p0
    # 4. line in NH observed direction of p1 passing thru p1
    sol0 = matmul(proj, 0 - p0)  # SS barycenter seen from p0 (0, 0, |p0|)
    r0 = sol0[2]
    r0nh = sqrt(sum((p0 - nh_x)**2))
    d00 = -r0nh*matmul(proj, d0)  # scaled minus NH observed direction of p0
    nh0t = matmul(proj, nh_x - p0)  # true NH seen from p0
    nh0c = matmul(proj, x - p0)  # chi2 NH seen from p0
    nh0k = matmul(proj, y - p0)  # Kaplan NH seen from p0
    p10 = matmul(proj, p1 - p0)  # p1 seen from p0
    d10 = matmul(proj, d1)  # NH observed direction to p1 in p0 coords
    pixel = r0 * (pi/180. / 3600.) * 4.08  # pixel size in AU at r0
    return Garbage(
        x=x, urms=urms, chi2=chi2, xcov=xcov, xaxes=xaxes, xstd=xstd,
        y=y, yrms=yrms, ychi2=ychi2, ycov=ycov, yaxes=yaxes, ystd=ystd,
        nh_x=nh_x,
        p0=p0, p0cov=p0cov, p0proj=p0proj, p1=p1, p1cov=p1cov, p1proj=p1proj,
        d0=d0, d0cov=d0cov, d0proj=d0proj, d1=d1, d1cov=d1cov, d1proj=d1proj,
        sol0=sol0, r0=r0, r0nh=r0nh, d00=d00, nh0t=nh0t, nh0c=nh0c,
        nh0k=nh0k, p10=p10, d10=d10, pixel=pixel)


# g = display_prep()
# r1 = sqrt(sum(g.p1**2))
# xy0 = g.p10-(r1+50)*g.d10
# xy1 = g.p10-(r1+10)*g.d10
# plt.rcParams['axes.prop_cycle']  <-- cols from printed array
#    cols[0] = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
# clf()
# plot([x,x],y[::10],lw=0.5,c="gray"); plot(x[::10],[y,y],lw=0.5,c="gray")
# scatter([0, g.nh0t[0]], [0, g.nh0t[1]], marker="x", c="k")
# scatter([g.nh0k[0]], [g.nh0k[1]], marker="+", c="r")
# scatter([g.nh0c[0]], [g.nh0c[1]], marker="+", c="b")
# scatter([g.d00[0]], [g.d00[1]], marker=".", c=[cols[0]])
# plot([xy0[0],xy1[0]], [xy0[1], xy1[1]])
# xylim(-55,5,-5,55)
# xylim(-43.0125, -40.1618, 4.3536, 6.6158)
# savefig(...)
