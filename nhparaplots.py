from datetime import datetime

from numpy import (array, arange, asfarray, pi, cos, sin, sqrt, arctan2,
                   where, matmul, cross, interp)
from scipy.linalg import svd
from matplotlib import rc
from matplotlib.pyplot import (clf, axes, axis, plot, text, arrow, savefig,
                               figure)

rc('lines', linewidth=2)

# https://ssd.jpl.nasa.gov/astro_par.html
# obliquity of ecliptic at J2000 = 84381.41100 as  (IAU 1976 value 84381.448)
# J2000.0 = JD 2451545.0
# ICRF = J2000, ECLIPJ2000

# Chapront 2002 gives 23 26 21.41100 at J2000 = 23.439280833
# IAU 1976 gives 23 26 21.448 at J2000 = 23.4392911
obliquity = 23.4392911 * pi / 180.0  # use IAU 1976 value here
_coblq, _soblq = cos(obliquity), sin(obliquity)
# _oblq @ xyz_equ --> xyz_ecl
# xeq = xecl                   xecl = xeq
# yeq = c*yeq - s*zecl         yecl = c*yeq + s*zeq
# zeq = s*zecl + c*zecl        zecl = -s*yeq + c*zeq
_oblq = array([[1., 0., 0.], [0., _coblq, _soblq], [0., -_soblq, _coblq]])

# J2000 orbital shapes
# planet    a (au)            e       incl (deg)    plon (deg)    nlon (deg)
# Mercury   0.38709927   0.20563593   7.00497902   77.45779628   48.33076593
# Venus     0.72333566   0.00677672   3.39467605  131.60246718   76.67984255
# Earth     1.00000261   0.01671123  -0.00001531  102.93768193    0.0
# Mars      1.52371034   0.09339410   1.84969142  -23.94362959   49.55953891
# Jupiter   5.20288700   0.04838624   1.30439695   14.72847983  100.47390909
# Saturn    9.53667594   0.05386179   2.48599187   92.59887831  113.66242448
# Uranus   19.18916464   0.04725744   0.77263783  170.95427630   74.01692503
# Neptune  30.06992276   0.00859048   1.77004347   44.96476227  131.78422574
# Pluto    39.48211675   0.24882730  17.14001206  224.06891629  110.30393684
orbital_params = dict(
    mercury=[0.38709927, 0.20563593, 7.00497902, 77.45779628, 48.33076593],
    venus=[0.72333566, 0.00677672, 3.39467605, 131.60246718, 76.67984255],
    earth=[1.00000261, 0.01671123, -0.00001531, 102.93768193, 0.0],
    mars=[1.52371034, 0.09339410, 1.84969142, -23.94362959, 49.55953891],
    jupiter=[5.20288700, 0.04838624, 1.30439695, 14.72847983, 100.47390909],
    saturn=[9.53667594, 0.05386179, 2.48599187, 92.59887831, 113.66242448],
    uranus=[19.18916464, 0.04725744, 0.77263783, 170.95427630, 74.01692503],
    neptune=[30.06992276, 0.00859048, 1.77004347, 44.96476227, 131.78422574],
    pluto=[39.48211675, 0.24882730, 17.14001206, 224.06891629, 110.30393684])
for p in orbital_params:
    orbital_params[p] = asfarray(orbital_params[p])
    orbital_params[p][2:] *= pi / 180.


def get_orbit(planet, npts=256):
    a, e, incl, plon, nlon = orbital_params[planet]
    ee = 2.*pi * arange(npts) / (npts - 1.)
    cee, see = cos(ee), sin(ee)
    x, y = cee - e,  sqrt(1. - e**2) * see
    aper = plon - nlon  # argument of perihelion
    cw, sw = cos(aper), sin(aper)
    cn, sn = cos(nlon), sin(nlon)
    ci, si = cos(incl), sin(incl)
    cisn, cicn = ci*sn, ci*cn
    return array([(cw*cn - sw*cisn)*x - (sw*cn + cw*cisn)*y,
                  (cw*sn + sw*cicn)*x - (sw*sn - cw*cicn)*y,
                  (sw*x + cw*y)*si]) * a


def equ2ecl(xyz, y=None, z=None):
    xyz = asfarray(xyz)
    if y is not None:
        xyz = xyz[..., None] + asfarray([0., 0., 0.])
        xyz[..., 1], xyz[..., 2] = y, z
    return matmul(_oblq, xyz[..., None])[..., 0]


def ecl2equ(xyz, y=None, z=None):
    xyz = asfarray(xyz)
    if y is not None:
        xyz = xyz[..., None] + asfarray([0., 0., 0.])
        xyz[..., 1], xyz[..., 2] = y, z
    return matmul(_oblq.T, xyz[..., None])[..., 0]


def radec2xyz(ra, dec=None, r=None):
    if dec is None:
        ra = asfarray(ra)
        if ra.shape[0] == 3:
            ra, dec, r = ra
        else:
            ra, dec = ra
    else:
        dec = asfarray(dec)
        if r is not None:
            r = asfarray(r)
    ra, dec = ra * pi / 180.0, dec * pi / 180.0
    ca, sa = cos(ra), sin(ra)
    cd, sd = cos(dec), sin(dec)
    xyz = asfarray([ca*cd, sa*cd, sd])
    return xyz if r is None else r * xyz


def xyz2radec(x, y=None, z=None, with_r=False):
    if y is None:
        x, y, z = asfarray(x)
    elif y is None:
        y = asfarray(y)
    elif z is None:
        z = asfarray(z)
    ra = arctan2(y, x + 1.2345e-99) * 180./pi
    ra = where(ra < 0., ra + 360., ra)
    r2 = x**2 + y**2
    dec = arctan2(z, sqrt(r2) + 1.2345e-99) * 180./pi
    if not with_r:
        return ra, dec
    return ra, dec, sqrt(r2 + z**2)


def read_horizons(filename):
    month2iso = dict(Jan="01", Feb="02", Mar="03", Apr="04",
                     May="05", Jun="06", Jul="07", Aug="08",
                     Sep="09", Oct="10", Nov="11", Dec="12")
    phase = 0
    header, jd, columns = None, [], []
    for line in open(filename):
        line = line.strip()
        if phase == 0:
            phase = 1 if line.startswith("**********") else 0
            continue
        elif phase == 1:
            if line.startswith("Date__(UT)__HR:MN"):
                header = line.split()
                phase = 2
            continue
        elif phase == 2:
            if line == "$$SOE":
                phase = 3
            continue
        elif phase == 3 and line == "$$EOE":
            break
        cols = line.split()
        date, cols = " ".join(cols[:2]), cols[2:]  # separate date
        columns.append([float(v) for v in cols])
        date = date.split("-", 2)
        date[1] = month2iso[date[1]]
        date = datetime.fromisoformat("-".join(date) + "+00:00")
        # Convert POSIX time (seconds since 1970-01-01 00:00) to JD
        jd.append(date.timestamp()/86400. + 2440587.5)
    return header, array(jd), array(columns)


jpl_nh_hdr, jpl_nh_jd, jpl_nh_cols = read_horizons("nhjpl_traj.txt")
jpl_nh_ra_dec_r = jpl_nh_cols[:, [0, 1, 6]]
# New Horizons positions from 2006 to 2024 from JPL Horizons
# Initially in ICRS J2000 equatorial coordinates, convert to J2000 ecliptic
nh_xyz = radec2xyz(jpl_nh_ra_dec_r.T)  # 3x219
nh_xyz = equ2ecl(nh_xyz.T).T

# Position of NH at proxima and wolf observations (JPL horizons)
# nh_prox = equ2ecl([14.39949805, -41.52427211, -16.24295586])
# nh_wolf = equ2ecl([14.39415179, -41.51851105, -16.24017574])
nh_prox = equ2ecl([13.54653191, -42.01288633, -16.45470347])
nh_wolf = equ2ecl([13.54936315, -42.01913102, -16.45712871])

# 256 points on orbits of each outer planet (at J2000)
# These are J2000 ecliptic coordinates.
jup_xyz = get_orbit("jupiter")
sat_xyz = get_orbit("saturn")
ura_xyz = get_orbit("uranus")
nep_xyz = get_orbit("neptune")
plu_xyz = get_orbit("pluto")

# old values with just first three images of each star
# prox_p = array([-97953.67951680758, -74843.75608285567, -238585.345246937])
# wolf_p = array([-474237.1701433132, 135103.1850675353, 60541.43185132139])
# prox_d = array([-0.364828764412182, -0.2785603518906408, -0.888427882842520])
# wolf_d = array([-0.9545434996533468, 0.2720115447504638,0.12188694266412427])
prox_p = array([-97953.68289082,  -74843.75300592, -238585.34482693])
wolf_p = array([-474237.16953007,  135103.18821108,   60541.42963995])
prox_d = array([-0.36482883, -0.27856106, -0.88842763])
wolf_d = array([-0.95454352,  0.27201153,  0.12188681])
prox_p, wolf_p = equ2ecl(prox_p), equ2ecl(wolf_p)
prox_d, wolf_d = equ2ecl(prox_d), equ2ecl(wolf_d)
# nh is about 2 degrees above ecliptic
# proxima is about 45 degrees below ecliptic, wolf nearly in ecliptic (0.2 deg)
# angle between wolf and proxima is 80.55 degrees
# Find points on the nh-prox and nh-wolf lines in the solar system:
prox_q = prox_p - prox_d * prox_d.dot(prox_p)
wolf_q = wolf_p - wolf_d * wolf_d.dot(wolf_p)

# xyz weighted by 1/r**2
wxyz = array([13.65470968, -41.98857720, -16.20123976])
wxyz_chi2 = 45.441396815063264  # just raw 1/r**2 weights, but scales cov
wxyz_cov = array([[0.01737435, -0.00611055, -0.00121004],
                  [-0.00611055, 0.00533358, 0.00244089],
                  [-0.00121004, 0.00244089, 0.00358894]])
wxyz = equ2ecl(wxyz)

# raw observations
prox_raw = array([[-0.36482939, -0.36482994, -0.36482696],
                  [-0.27855885, -0.27855959, -0.27856262],
                  [-0.88842810, -0.88842764, -0.88842791]])
wolf_raw = array([[-0.95454356, -0.95454352, -0.95454342],
                  [0.27201145, 0.27201143, 0.27201175],
                  [0.12188671, 0.12188704, 0.12188708]])
prox_raw, wolf_raw = equ2ecl(*prox_raw), equ2ecl(*wolf_raw)

cb6_rdylbu = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
sns_rdylbu = ['#e34933', '#fca55d', '#fee99d', '#e9f6e8', '#a3d3e6', '#588cc0']

# x, xcov = n_star_solve([proxima.p, wolf.p], [p_dbar, w_dbar])
x2 = array([13.68164369, -41.82340672, -16.19656273])
x2cov = array([[1.03768713e+11, 1.71672039e+10, 6.92428360e+10],
               [1.71672039e+10, 7.08565668e+10, 4.65404606e+10],
               [6.92428360e+10, 4.65404606e+10, 2.08960579e+11]])
x2 = equ2ecl(x2)
x2cov = matmul(_oblq, matmul(x2cov, _oblq.T))
# x, xcov = n_star_solve([proxima.p]*6+[wolf.p]*6,
#                        list(proxima_nh.raw_xyz)+list(wolf_nh.raw_xyz))
x6 = array([13.68164206, -41.82340774, -16.19656615])
x6cov = array([[1.72947855e+10, 2.86120065e+09, 1.15404727e+10],
               [2.86120065e+09, 1.18094278e+10, 7.75674343e+09],
               [1.15404727e+10, 7.75674343e+09, 3.48267632e+10]])
x6 = equ2ecl(x6)
x6cov = matmul(_oblq, matmul(x6cov, _oblq.T))


def vernal_up(xyz):
    return -xyz[1], xyz[0]


def pts_on_line(q, d, s1, s2):
    p1, p2 = q + s1*d, q + s2*d
    return vernal_up(array([p1, p2]).T)


def arrow_along(q, d, s1, s2):
    p1, p2 = q + s1*d, q + s2*d
    return vernal_up(p1) + vernal_up(p2 - p1)


# Create 2x3 matrix taking point in 3D space into u-plane perpendicular to d.
def uspace_rot(d, up=(0., 0., 1.)):
    d, up = asfarray(d), asfarray(up)
    right = cross(d, up)
    right /= sqrt(sum(right**2))
    up = cross(right, d / sqrt(sum(d**2)))
    return array([right, up])


# Line thru p1 in direction d1, line thru p2 in direction d2.
def closest_pts(p1, d1, p2, d2):
    p1, d1, p2, d2 = map(asfarray, [p1, d1, p2, d2])
    # Find d3 normal to both lines, such that d1, d2, d3 right handed
    d3 = cross(d1, d2)
    # p1 + s*d1 = p2 + t*d2  at intersection point in plane normal to d3
    # s * cross(d1, d2) = cross(p2 - p1, d2)
    # s = cross(p2 - p1, d2).dot(d3) / d3.dot(d3)
    # t = cross(p2 - p1, d1).dot(d3) / d3.dot(d3)
    # q1 = p1 + d1 * cross(p2 - p1, d2).dot(rd3)
    d3 /= d3.dot(d3)
    p21 = p2 - p1
    s, t = cross(p21, d2).dot(d3), cross(p21, d1).dot(d3)
    q1, q2 = p1 + s*d1, p2 + t*d2
    return q1, q2, abs(d3.dot(q2 - q1)) / sqrt(d3.dot(d3))


# Lines are skew by about 1/3 AU.
prox_0, wolf_0, pw_skew = closest_pts(prox_p, prox_d, wolf_p, wolf_d)
# distances to proxima and wolf (from SS barycenter)
rprox = sqrt(sum(prox_p**2))
rwolf = sqrt(sum(wolf_p**2))

rot_prox, rot_wolf = uspace_rot(prox_d), uspace_rot(wolf_d)
arrow_prox = arrow_along(prox_q, prox_d, -92, 47)
arrow_wolf = arrow_along(wolf_q, wolf_d, -70, 18)
seg_prox = pts_on_line(prox_q, prox_d, -92, 51.5)
seg_wolf = pts_on_line(wolf_q, wolf_d, -70, 21)
los_dot = array([0., 0.])
prox_line = matmul(rot_prox, (wolf_q + array([[-70.], [21.]])*wolf_d
                              - prox_q)[..., None])[..., 0].T
wolf_line = matmul(rot_wolf, (prox_q + array([[-92.], [51.5]])*prox_d
                              - wolf_q)[..., None])[..., 0].T
prox_nhpos = matmul(rot_prox, nh_xyz - prox_q[:, None])
wolf_nhpos = matmul(rot_wolf, nh_xyz - wolf_q[:, None])


def get_ticks(x, y, length=1):
    xy = array([x, y])
    dxy = xy[:, 1:] - xy[:, :-1]
    perp = 0. * xy
    perp[:, 1:] = dxy
    perp[:, :-1] += dxy
    perp[:, 1:-1] *= 0.5
    perp = perp[::-1] * array([[-1.], [1.]])
    perp /= sqrt(perp[0]**2 + perp[1]**2)
    # perp = unit vectors perpendicular to (x, y)
    ticks = xy + 0.5*length*array([-perp, perp])
    # endpts x xycoord x npts --> xycoord x endpts x npts
    return ticks.transpose(1, 0, 2)


nh_xy = vernal_up(nh_xyz)
nh_ticks = get_ticks(*nh_xy, 1.4)
# 2454102.0 is 2007-01-01 12:00+00:00
tickt = 2454102.0 + 365.25 * arange(17)
ticki = interp(tickt, jpl_nh_jd, arange(jpl_nh_jd.size))
tickf = ticki % 1.0
ticki = (ticki // 1.0).astype(int)
nh_ticks = (1. - tickf)*nh_ticks[..., ticki] + tickf*nh_ticks[..., ticki+1]


# nhfig1 caption:
#
# Orbits of the outer planets and New Horizons (NH) viewed from the
# ecliptic north pole.  The dotted line is zero right ascension.  The
# P and W rays show the directions to Proxima Cen and Wolf 359,
# respectively, as measured in NH images on 2020-04-23.  Each ray passes
# through the exact 3D position of its star taken from the Gaia archive,
# so the spacecraft must have been at their intersection point on that
# date.  Proxima Cen is about 45 degrees south of the ecliptic, so the
# P ray is pointing into the page at that angle.  Wolf 359 is nearly in
# the ecliptic plane, and the spacecraft is within 2 degrees.
def fig1(prog=3, save=False, name="nhfig1.png", dpi=300):
    fig_1 = figure(1)
    clf()
    fig_1.set_figwidth(9.77)
    fig_1.set_figheight(7.45)
    fig_1.set_dpi(100.)
    axes(aspect="equal").set_axis_off()
    # axis((-50.768, 69.110, -36.685, None))  does weird stuff here
    plot(*vernal_up(jup_xyz), c="0.7", lw=1)
    plot(*vernal_up(sat_xyz), c="0.7", lw=1)
    plot(*vernal_up(ura_xyz), c="0.7", lw=1)
    plot(*vernal_up(nep_xyz), c="0.7", lw=1)
    plot(*vernal_up(plu_xyz), c="0.7", lw=1)
    arrow(0, 0, 0, 51, color="0.45", head_width=3, head_length=4.5,
          zorder=2)
    for y in [0, 10, 20, 30, 40, 50]:
        plot([-1.8, 0], [y, y], c="0.45", lw=1)
        text(-2.5, y, str(y), va="center", ha="right", c="0.45", size=10)
    text(1, 48, "x (au)", va="center", ha="left", c="0.45", size=12)
    if prog > 2:
        plot(*nh_ticks, c="k", lw=0.75)
        plot(*nh_xy, c="k", solid_capstyle="round")
        px, py = vernal_up(nh_xyz[:, -1])
        text(px+1, py, "NH", size="large", va="center", ha="left", c="k")
    plot(0, 0, "o", ms=5, c="orange")
    if prog > 0:
        plot(*seg_prox, c="#d73027")
        text(35, 22.2, "P", size="large", va="center", ha="center",
             c="#d73027")
        ang = arctan2(*(prox_d[:2]*[1, -1])) * 180./pi
        text(46, 2.5, "Proxima Cen", size=10, c="#d73027", rotation=ang)
        c, s = cos(ang * pi/180.), sin(ang * pi/180.)
        arrow(59.8, 2.7, 2.7*c, 2.7*s, width=0.1, head_width=1.2,
              color="#d73027")
    if prog > 1:
        plot(*seg_wolf, c="#4575b4")
        text(46.2, 27, "W", size="large", va="center", ha="center",
             c="#4575b4")
        ang = arctan2(*(wolf_d[:2]*[-1, 1])) * 180./pi
        text(33.1, -27.3, "Wolf 359", size=10, c="#4575b4", rotation=ang)
        c, s = -cos(ang * pi/180.), -sin(ang * pi/180.)
        arrow(33.5, -28, 2.7*c, 2.7*s, width=0.1, head_width=1.2,
              color="#4575b4")
    if prog > 1:
        text(12, 52, "New Horizons 2020-04-23", size=14, c="k")
    axis((-50.768, 69.110, -36.685, None))
    if save:
        if prog < 3:
            name = name.split(".")
            name = ".".join([name[0] + "-{}".format(prog), name[1]])
        savefig(name, dpi=dpi, facecolor="w")


def circle_pts(x, y, r, npts=256):
    th = arange(npts) * 2.*pi / (npts - 1.)
    return x + r*cos(th), y + r*sin(th)


def ellipsoid(xcen, ycen, rot, xcov, sigma_d=1.e-6, npts=256):
    # project xcov into plane by rot, then decompose into 2x2 rot and sig**2
    rot, sig2, _ = svd(matmul(rot, matmul(xcov, rot.T)))
    # rot is [major_axis, minor_axis].T
    xy = matmul(rot, array(circle_pts(0., 0., 1., npts)).T[..., None])[..., 0]
    xy *= sqrt(sig2) * sigma_d  # stretch into ellipse
    xy = matmul(rot.T, xy[..., None])[..., 0] + array([xcen, ycen])
    return xy.T


def fig2(save=False, draw=False,  name="nhfig2.png", dpi=300):
    fig_2 = figure(2, layout=None)
    fig_2.set_figwidth(10)
    fig_2.set_figheight(5)
    fig_2.set_dpi(100.)
    fig_2.clear()
    gs = fig_2.add_gridspec(nrows=1, ncols=2, left=0.025, right=0.975,
                            top=0.95, bottom=0.05, hspace=0.05, wspace=0.1)
    ax0, ax1 = fig_2.add_subplot(gs[0, 0]), fig_2.add_subplot(gs[0, 1])
    for ax in ax0, ax1:
        ax.clear()
        ax.set_aspect("equal")
        ax.set_axis_off()
    # point to zoom around in ax0: (0, 0), in ax1: (-0.2, 0)
    ax0.axis([-0.30, 0.30, -0.30, 0.30])
    ax1.axis([-0.50, 0.10, -0.30, 0.30])
    for r, n, rot, q, d, _0, line, raw, c1, c2, ax in [
            [rprox, 8, rot_prox, prox_q, prox_d, wolf_0, prox_line, prox_raw,
             "#d73027", "#4575b4", ax0],
            [rwolf, 5, rot_wolf, wolf_q, wolf_d, prox_0, wolf_line, wolf_raw,
             "#4575b4", "#d73027", ax1]]:
        is_ax0 = r < 300000.0
        # axes
        xy = matmul(rot, nh_xyz[:, 169:183]-q[:, None])
        ticks = get_ticks(*xy, 0.03)
        xytk = (asfarray(ticks)[:, :, 7] - asfarray(xy)[:, None, 7]) * 1.6
        x0, y0 = matmul(rot, 0.5*(nh_prox + nh_wolf) - q)
        # ax.arrow(x0, y0, 0,0.54, width=0.002, head_width=0.040, color="0.75")
        ax.plot([x0, x0], [y0-0.5, y0+0.5], c="0.75", lw=2)
        for tick in arange(-3, 3):
            tick = 0.1 * (tick if is_ax0 else tick + 1)
            if tick == 0.0:
                continue
            ax.plot([x0-0.01, x0+0.01], [y0+tick, y0+tick], c="0.75", lw=2)
            ax.text(x0-0.02, y0+tick, "{:4.1f}".format(tick),
                    c="0.75", size=10, va="center", ha="right")
        ax.text(x0-0.02, y0+(0.15 if is_ax0 else 0.35), "(au)",
                c="0.75", size=10, va="center", ha="right")
        # ax.arrow(x0, y0,-0.54,0, width=0.002, head_width=0.040, color="0.75")
        ax.plot([x0-0.5, x0+0.5], [y0, y0], c="0.75", lw=2)
        for tick in 0.1*arange(-4, 5):
            if tick == 0.0:
                continue
            ax.plot([x0+tick, x0+tick], [y0-0.01, y0+0.01], c="0.75", lw=2)
        # arcsec circles and lines
        arcsec = r * pi / 180. / 3600
        for rr in arange(1, n) * 0.1 * arcsec:
            ax.plot(*circle_pts(0., 0., rr), c=c1, lw=0.75, dashes=[0.75, 2])
        x, y = line
        dx, dy = y[0] - y[1], x[1] - x[0]
        rr = sqrt(dx**2 + dy**2)
        dx, dy = dx/rr, dy/rr
        arcsec = (rprox + rwolf - r) * pi / 180. / 3600
        for rr in arange(n-13, 13-n) * 0.1 * arcsec:
            if rr:
                ax.plot(x+rr*dx, y+rr*dy, c=c2, lw=0.75, dashes=[0.75, 2])
        # NH trajectory (right to left in both views)
        # Ticks are exactly 30 days apart at 00:00:00 UTC
        # 2020-02-06, 2020-03-07, 2020-04-06, 2020-05-06, 2020-06-05
        nh = array([x0, y0])
        ax.plot(*ticks, c="k", lw=0.75)
        ax.plot(*xy, c="k")
        ax.plot(*(xytk + nh[:, None]), c="k")
        ax.plot(*nh, "o", ms=5, c="k")
        # gray dashed shortest distance line between P and W
        p0, p1 = los_dot, matmul(rot, _0-q)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", c="0.5")
        kaplan = 0.5*(p0 + p1)
        ax.plot(*kaplan, marker="o", ms=5, c="0.5")  # dot at midpoint
        weighted = matmul(rot, x6-q)  # wxyz-q
        ax.plot(*weighted, marker="*", ms=16, c="k")
        ax.plot(*ellipsoid(*weighted, rot, x6cov, 1.e-6), c="k")
        ax.plot(*ellipsoid(*weighted, rot, x6cov, 2.e-6), c="k")
        # P and W themselves with dots at closest approach
        ax.plot(*p0, "o", ms=5, c=c1)
        ax.plot(*line, c=c2)
        ax.plot(*p1, "o", ms=5, c=c2)
        if draw:
            # raw positions of P or W
            p0, p1, p2 = matmul(rot, r*(raw - d).T).T
            for p in [p0, p1, p2]:
                ax.plot(*p, marker="+", ms=12, mew=1, c=c1)
        # annotations
        p0 = matmul(rot, _0 - q)
        p1 = line[:, 1] - line[:, 0]
        p1 /= sqrt(sum(p1**2)) * 40
        if is_ax0:  # this is proxima subplot
            ax.text(-0.28, 0.22, "NH", size="large", c="k")
            # ax.text(0.020, -0.022, "P", size="large", c=c1)
            ax.text(0.008, -0.01, "P", size="large", c=c1)
            ax.text(0.03, -0.16, "W", size="large", c=c2)
            p0 += [0.045, -0.014]
        else:  # this is wolf subplot
            ax.text(-0.49, -0.075, "NH", size="large", c="k")
            ax.text(0.01, -0.012, "W", size="large", c=c1)
            ax.text(-0.113, 0.092, "P", size="large", c=c2)
            p0 += [-0.025, 0.003]
        ax.arrow(*p0, *p1, width=0.0007, head_width=0.0084, color=c2)
    if save:
        savefig(name, dpi=dpi, facecolor="w")
